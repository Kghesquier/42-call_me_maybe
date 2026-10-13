"""Constrained decoding: make the LLM write a schema-valid function call.

Principle
---------
We never let the model write the JSON on its own. The structure
(braces, keys, commas, quotes) is *forced*: we append those tokens
ourselves. The model is only asked to fill the free parts:

* the function name   -> must be one of the names in the definitions;
* each argument value -> must match the argument type (number, integer,
  string, boolean).

For each free token we:
1. ask the model for the logits of the next token;
2. compute the set of tokens that keep the output valid;
3. set every other logit to -inf;
4. take the argmax (greedy decoding).

Because invalid tokens can never be picked, the final JSON is always
valid and matches the schema, whatever the model "wanted" to write.
"""

import json
import math
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.models import SUPPORTED_TYPES, FunctionCall, FunctionDefinition
from src.prompt import build_prompt
from src.vocabulary import Vocabulary

IdArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

# --- number grammar -------------------------------------------------------
NUMBER_CHARS = frozenset("0123456789.-+eE")
NUMBER_PREFIX = re.compile(r"-?(\d+(\.\d*)?([eE][+-]?\d{0,3})?)?")
NUMBER_FULL = re.compile(r"-?\d+(\.\d+)?([eE][+-]?\d{1,3})?")
INTEGER_PREFIX = re.compile(r"-?\d*")
INTEGER_FULL = re.compile(r"-?\d+")
# First characters of a token that may end a number (", " or "}" ...).
NUMBER_END_CHARS = frozenset(",} \n\t")

# --- JSON string grammar (small state machine) ----------------------------
STATE_NORMAL = 0      # inside the string, no pending escape
STATE_ESCAPE = 1      # just read a backslash
STATE_HEX_BASE = 10   # 10 + n : inside \uXXXX, n hex digits still needed
ESCAPE_CHARS = '"\\/bfnrtu'
HEX_CHARS = "0123456789abcdefABCDEF"


class DecodingError(Exception):
    """Raised when no valid token can be produced (should not happen)."""


def advance_string_state(state: int, text: str) -> Tuple[Optional[int], int]:
    """Feed ``text`` to the JSON-string state machine.

    Args:
        state: Current state (see the ``STATE_*`` constants).
        text: Characters of the candidate token.

    Returns:
        ``(new_state, close_index)``. ``new_state`` is ``None`` when the
        text would make the string invalid. ``close_index`` is the
        position of the closing quote in ``text`` or ``-1`` if the string
        is still open.
    """
    for index, char in enumerate(text):
        if state == STATE_NORMAL:
            if char == '"':
                return state, index
            if char == "\\":
                state = STATE_ESCAPE
            elif ord(char) < 0x20:
                return None, -1          # raw control chars are illegal
        elif state == STATE_ESCAPE:
            if char == "u":
                state = STATE_HEX_BASE + 4
            elif char in ESCAPE_CHARS:
                state = STATE_NORMAL
            else:
                return None, -1          # e.g. "\d" is not valid JSON
        else:
            if char not in HEX_CHARS:
                return None, -1
            remaining = state - STATE_HEX_BASE - 1
            state = STATE_NORMAL if remaining == 0 else state - 1
    return state, -1


def repair_string(content: str) -> str:
    """Turn a truncated JSON string body into a valid Python string.

    Used only when the length limit is reached in the middle of an escape.

    Args:
        content: Raw JSON string body (without quotes).

    Returns:
        The decoded string value.
    """
    while content:
        try:
            value = json.loads(f'"{content}"')
            return str(value)
        except json.JSONDecodeError:
            content = content[:-1]
    return ""


def repair_number(content: str, integer: bool) -> str:
    """Make sure a generated number is complete (``"12."`` -> ``"12"``).

    Args:
        content: Characters generated so far.
        integer: Whether only integers are allowed.

    Returns:
        A string matching the full number grammar.
    """
    pattern = INTEGER_FULL if integer else NUMBER_FULL
    while content and not pattern.fullmatch(content):
        content = content[:-1]
    return content or "0"


class ConstrainedDecoder(BaseModel):
    """Generate schema-valid function calls with a small LLM.

    Attributes:
        llm: Object exposing ``encode`` and ``get_logits_from_input_ids``
            (the ``Small_LLM_Model`` of ``llm_sdk``).
        vocab: Decoded vocabulary of the model.
        max_string_tokens: Safety limit for string arguments.
        max_number_chars: Safety limit for numeric arguments.
        verbose: Print every chosen token (visualization of generation).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Any
    vocab: Vocabulary
    max_string_tokens: int = 64
    max_number_chars: int = 32
    verbose: bool = False

    _plain_ids: IdArray = PrivateAttr()
    _special_ids: IdArray = PrivateAttr()
    _number_ids: IdArray = PrivateAttr()
    _number_end_ids: IdArray = PrivateAttr()
    _number_end_set: Set[int] = PrivateAttr()
    _by_first_char: Dict[str, IdArray] = PrivateAttr()
    _choice_cache: Dict[Tuple[str, ...], IdArray] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Pre-sort the vocabulary into token families (done once).

        Checking 150k tokens in Python at every step would be far too
        slow, so tokens are classified once:

        * plain   : no quote, no backslash, no control char -> always
          valid inside a string;
        * special : contains a quote or a backslash -> must be checked
          with the state machine;
        * number  : only digits / sign / dot / exponent chars;
        * end     : can terminate a number (starts with ``,`` ``}``...).
        """
        plain: List[int] = []
        special: List[int] = []
        number: List[int] = []
        number_end: List[int] = []
        first: Dict[str, List[int]] = {}
        watched = set(ESCAPE_CHARS + HEX_CHARS)
        for token_id, text in enumerate(self.vocab.token_texts):
            if not text:
                continue
            if '"' in text or "\\" in text:
                special.append(token_id)
            elif all(ord(char) >= 0x20 for char in text):
                plain.append(token_id)
            if all(char in NUMBER_CHARS for char in text):
                number.append(token_id)
            if text[0] in NUMBER_END_CHARS:
                number_end.append(token_id)
            if text[0] in watched:
                first.setdefault(text[0], []).append(token_id)
        self._plain_ids = np.array(plain, dtype=np.int64)
        self._special_ids = np.array(special, dtype=np.int64)
        self._number_ids = np.array(number, dtype=np.int64)
        self._number_end_ids = np.array(number_end, dtype=np.int64)
        self._number_end_set = set(number_end)
        self._by_first_char = {
            char: np.array(ids, dtype=np.int64) for char, ids in first.items()
        }
        self._choice_cache = {}

    # ------------------------------------------------------------------
    # low-level helpers
    # ------------------------------------------------------------------
    def _encode(self, text: str) -> List[int]:
        """Tokenize text with the SDK and return a flat list of ids.

        Args:
            text: Text to encode.

        Returns:
            Token ids.
        """
        raw = self.llm.encode(text)
        values: Any = raw.tolist() if hasattr(raw, "tolist") else raw
        while values and isinstance(values[0], list):
            values = values[0]
        return [int(value) for value in values]

    def _next_logits(self, ids: List[int]) -> FloatArray:
        """Run the model and get the logits of the next token.

        Args:
            ids: Whole context (prompt + what was generated so far).

        Returns:
            One logit per token id.
        """
        raw = self.llm.get_logits_from_input_ids(ids)
        return np.asarray(raw, dtype=np.float64).reshape(-1)

    def _select(self, logits: FloatArray, allowed: IdArray) -> int:
        """Mask forbidden tokens with -inf and return the best allowed one.

        Args:
            logits: Raw logits from the model.
            allowed: Ids of the tokens that keep the output valid.

        Returns:
            The chosen token id.

        Raises:
            DecodingError: If no token is allowed.
        """
        allowed = allowed[allowed < logits.shape[0]]
        if allowed.size == 0:
            raise DecodingError("no valid token available at this step")
        masked = np.full(logits.shape, -np.inf)
        masked[allowed] = logits[allowed]
        token_id = int(np.argmax(masked))
        if self.verbose:
            print(
                f"    [{allowed.size:>6} allowed] -> "
                f"{self.vocab.token_texts[token_id]!r}",
                file=sys.stderr,
            )
        return token_id

    # ------------------------------------------------------------------
    # value generators (each one appends what it generates to ``ids``)
    # ------------------------------------------------------------------
    def _choice_candidates(self, options: Sequence[str]) -> IdArray:
        """Tokens that appear somewhere inside one of the options.

        Args:
            options: The allowed strings.

        Returns:
            Ids of tokens worth testing (cached per option set).
        """
        key = tuple(options)
        if key not in self._choice_cache:
            ids = [
                token_id
                for token_id, text in enumerate(self.vocab.token_texts)
                if text and any(text in option for option in options)
            ]
            self._choice_cache[key] = np.array(ids, dtype=np.int64)
        return self._choice_cache[key]

    def _choose(self, ids: List[int], options: Sequence[str]) -> str:
        """Let the model pick one string among ``options``.

        A token is allowed only if the text generated so far plus that
        token is still the beginning of at least one option. As soon as
        a single option remains, the rest is forced without calling the
        model (saves time).

        Args:
            ids: Context ids, extended in place.
            options: Allowed strings; none may be a prefix of another.

        Returns:
            The chosen option.
        """
        texts = self.vocab.token_texts
        candidates = self._choice_candidates(options)
        generated = ""
        while True:
            remaining = [o for o in options if o.startswith(generated)]
            if generated in remaining:
                return generated
            if len(remaining) == 1:
                ids.extend(self._encode(remaining[0][len(generated):]))
                return remaining[0]
            valid = np.array(
                [
                    token_id for token_id in candidates.tolist()
                    if any(o.startswith(generated + texts[token_id])
                           for o in remaining)
                ],
                dtype=np.int64,
            )
            token_id = self._select(self._next_logits(ids), valid)
            ids.append(token_id)
            generated += texts[token_id]

    def _generate_string(self, ids: List[int]) -> str:
        """Generate the body of a JSON string (opening quote already there).

        The closing quote is not appended to ``ids``: the caller adds it
        with the next forced segment.

        Args:
            ids: Context ids, extended in place.

        Returns:
            The decoded string value.
        """
        texts = self.vocab.token_texts
        empty = np.array([], dtype=np.int64)
        content = ""
        state = STATE_NORMAL
        for _ in range(self.max_string_tokens):
            logits = self._next_logits(ids)
            if state == STATE_NORMAL:
                always, pool = self._plain_ids, self._special_ids
            else:
                starts = ESCAPE_CHARS if state == STATE_ESCAPE else HEX_CHARS
                always = empty
                pool = np.concatenate(
                    [self._by_first_char.get(c, empty) for c in starts]
                )
            checked = [
                token_id for token_id in pool.tolist()
                if advance_string_state(state, texts[token_id])[0]
                is not None
            ]
            allowed = np.concatenate(
                [always, np.array(checked, dtype=np.int64)]
            )
            token_id = self._select(logits, allowed)
            text = texts[token_id]
            new_state, close = advance_string_state(state, text)
            if close >= 0:
                return repair_string(content + text[:close])
            content += text
            state = new_state if new_state is not None else STATE_NORMAL
            ids.append(token_id)
        return repair_string(content)

    def _generate_number(self, ids: List[int], integer: bool) -> float:
        """Generate a number (or an integer).

        Allowed tokens are those made only of number characters that keep
        the text a valid number prefix. Once the number is complete, the
        model may also pick a "terminator" token (``,`` / ``}``...) to
        say it is done; that token is not kept.

        Args:
            ids: Context ids, extended in place.
            integer: Restrict to integers.

        Returns:
            The value (an int is returned for integers).
        """
        texts = self.vocab.token_texts
        prefix = INTEGER_PREFIX if integer else NUMBER_PREFIX
        full = INTEGER_FULL if integer else NUMBER_FULL
        content = ""
        while len(content) < self.max_number_chars:
            logits = self._next_logits(ids)
            valid = [
                token_id for token_id in self._number_ids.tolist()
                if prefix.fullmatch(content + texts[token_id])
            ]
            allowed = np.array(valid, dtype=np.int64)
            if full.fullmatch(content):
                allowed = np.concatenate([allowed, self._number_end_ids])
            if allowed.size == 0:
                break
            token_id = self._select(logits, allowed)
            if token_id in self._number_end_set:
                break
            content += texts[token_id]
            ids.append(token_id)
        content = repair_number(content, integer)
        if integer:
            return int(content)
        value = float(content)
        if not math.isfinite(value):
            value = float(re.split(r"[eE]", content)[0])
        return value

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def decode_call(
        self, request: str, functions: List[FunctionDefinition]
    ) -> FunctionCall:
        """Produce the function call for one natural-language request.

        The context grows like this (forced parts in quotes)::

            <prompt> '{"name": "' <NAME> '", "parameters": {"a": ' <NUM>
            ', "b": ' <NUM> '}}'

        Args:
            request: The user prompt.
            functions: Available functions.

        Returns:
            The validated function call.
        """
        ids = self._encode(build_prompt(request, functions))
        ids.extend(self._encode('{"name": "'))
        # The closing quote makes options prefix-free ("fn_a" vs "fn_ab").
        chosen = self._choose(ids, [f.name + '"' for f in functions])[:-1]
        function = next(f for f in functions if f.name == chosen)
        pending = ', "parameters": {'
        parameters: Dict[str, Any] = {}
        for index, (name, spec) in enumerate(function.parameters.items()):
            key = (", " if index else "") + json.dumps(name) + ": "
            kind = spec.type if spec.type in SUPPORTED_TYPES else "string"
            if kind == "string":
                ids.extend(self._encode(pending + key + '"'))
                parameters[name] = self._generate_string(ids)
                pending = '"'
            elif kind == "boolean":
                ids.extend(self._encode(pending + key))
                parameters[name] = self._choose(ids, ["true", "false"]) \
                    == "true"
                pending = ""
            else:
                ids.extend(self._encode(pending + key))
                parameters[name] = self._generate_number(
                    ids, integer=(kind == "integer")
                )
                pending = ""
        return FunctionCall(
            prompt=request, name=function.name, parameters=parameters
        )
