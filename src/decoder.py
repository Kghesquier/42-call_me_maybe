"""Constrained decoding: the LLM writes a function call that is always valid.

The JSON structure (braces, keys, commas, quotes) is written by us: we
append those tokens to the context ourselves. The model only generates
the free parts:

* the function name  -> must be one of the defined names;
* each argument value -> must match the argument type.

For every generated token:
1. the model gives one logit per token of the vocabulary;
2. we compute the list of tokens that keep the output valid;
3. every other logit is set to -inf;
4. we take the token with the highest remaining logit (argmax).
"""

import re
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel

from src.models import FunctionCall, FunctionDefinition
from src.prompt import build_prompt

# What a number may look like while it is being written...
NUMBER_PREFIX = re.compile(r"-?(\d+(\.\d*)?)?")
INTEGER_PREFIX = re.compile(r"-?\d*")
# ...and when it is complete.
NUMBER_FULL = re.compile(r"-?\d+(\.\d+)?")
INTEGER_FULL = re.compile(r"-?\d+")

MAX_STRING_TOKENS = 50
MAX_NUMBER_CHARS = 20


class ConstrainedDecoder(BaseModel):
    """Generate function calls token by token with a masked LLM.

    Attributes:
        llm: The ``Small_LLM_Model`` from ``llm_sdk``.
        tokens: ``tokens[i]`` is the text of token ``i`` (see vocabulary).
        string_ids: Tokens allowed inside a string.
        number_ids: Tokens made only of digits, ``.`` and ``-``.
        end_ids: Tokens that may end a number (start with ``,`` or ``}``).
    """

    llm: Any
    tokens: List[str]
    string_ids: List[int] = []
    number_ids: List[int] = []
    end_ids: List[int] = []

    def model_post_init(self, __context: Any) -> None:
        """Sort the vocabulary once, so each step only filters a few ids."""
        for i, text in enumerate(self.tokens):
            if not text:
                continue
            # No backslash (no JSON escapes to handle) and no control
            # characters such as newlines.
            if "\\" not in text and all(ord(c) >= 32 for c in text):
                self.string_ids.append(i)
            if all(c in "0123456789.-" for c in text):
                self.number_ids.append(i)
            if text[0] in ",}":
                self.end_ids.append(i)

    def encode(self, text: str) -> List[int]:
        """Tokenize text with the SDK.

        Args:
            text: Text to encode.

        Returns:
            The token ids (the SDK returns a 2-D tensor ``[[ids]]``).
        """
        return [int(i) for i in self.llm.encode(text)[0].tolist()]

    def pick(self, ids: List[int], allowed: List[int]) -> int:
        """Run the model and return the best token among ``allowed``.

        Args:
            ids: The whole context (prompt + what is generated so far).
            allowed: Ids of the tokens that keep the output valid.

        Returns:
            The chosen token id.

        Raises:
            ValueError: If no token is allowed.
        """
        if not allowed:
            raise ValueError("no valid token at this step")
        logits = np.array(self.llm.get_logits_from_input_ids(ids))
        masked = np.full(len(logits), -np.inf)
        masked[allowed] = logits[allowed]
        return int(np.argmax(masked))

    def choose(self, ids: List[int], options: List[str]) -> str:
        """Let the model pick one string among ``options``.

        A token is allowed if (text so far + token) is still the start of
        an option. When only one option is left, it is completed without
        calling the model. No option may be a prefix of another one.

        Args:
            ids: Context ids, extended in place.
            options: The allowed strings.

        Returns:
            The chosen option.
        """
        text = ""
        while True:
            left = [o for o in options if o.startswith(text)]
            if len(left) == 1:
                ids.extend(self.encode(left[0][len(text):]))
                return left[0]
            allowed = [
                i for i, t in enumerate(self.tokens)
                if t and any(o.startswith(text + t) for o in left)
            ]
            token_id = self.pick(ids, allowed)
            ids.append(token_id)
            text += self.tokens[token_id]

    def generate_string(self, ids: List[int]) -> str:
        """Generate a string value (the opening quote is already there).

        The string ends with the first token that contains a ``"``; only
        the part before that quote is kept. That token is not added to
        ``ids``: the caller adds the closing quote itself.

        Args:
            ids: Context ids, extended in place.

        Returns:
            The string value.
        """
        value = ""
        for _ in range(MAX_STRING_TOKENS):
            token_id = self.pick(ids, self.string_ids)
            text = self.tokens[token_id]
            if '"' in text:
                return value + text.split('"')[0]
            value += text
            ids.append(token_id)
        return value

    def generate_number(self, ids: List[int], integer: bool) -> float:
        """Generate a number.

        A token is allowed if (text so far + token) still looks like the
        start of a number. Once the number is complete, the tokens
        starting with ``,`` or ``}`` are allowed too: picking one of them
        means the model has finished (that token is not kept).

        Args:
            ids: Context ids, extended in place.
            integer: Only allow integers.

        Returns:
            The value (an int when ``integer`` is True).
        """
        prefix = INTEGER_PREFIX if integer else NUMBER_PREFIX
        full = INTEGER_FULL if integer else NUMBER_FULL
        text = ""
        while len(text) < MAX_NUMBER_CHARS:
            allowed = [
                i for i in self.number_ids
                if prefix.fullmatch(text + self.tokens[i])
            ]
            if full.fullmatch(text):
                allowed += self.end_ids
            token_id = self.pick(ids, allowed)
            if token_id in self.end_ids:
                break
            text += self.tokens[token_id]
            ids.append(token_id)
        # Only reached on the length limit: fix an unfinished "12." or "-".
        text = text.rstrip(".")
        if text in ("", "-"):
            text = "0"
        return int(text) if integer else float(text)

    def decode_call(
        self, request: str, functions: List[FunctionDefinition]
    ) -> FunctionCall:
        """Produce the function call for one prompt.

        The context grows like this (parts in quotes are forced)::

            <prompt> '{"name": "' NAME '", "parameters": {"a": ' NUMBER
            ', "b": ' NUMBER '}}'

        Args:
            request: The user prompt.
            functions: The available functions.

        Returns:
            The function call.
        """
        ids = self.encode(build_prompt(request, functions))
        ids += self.encode('{"name": "')
        # The closing quote stops "fn_add" from being a prefix of
        # "fn_add_numbers".
        name = self.choose(ids, [f.name + '"' for f in functions])[:-1]
        function = next(f for f in functions if f.name == name)

        parameters: Dict[str, Any] = {}
        forced = ', "parameters": {'
        for index, (arg, spec) in enumerate(function.parameters.items()):
            if index > 0:
                forced += ", "
            forced += f'"{arg}": '
            if spec.type == "number" or spec.type == "integer":
                ids += self.encode(forced)
                parameters[arg] = self.generate_number(
                    ids, integer=(spec.type == "integer")
                )
                forced = ""
            elif spec.type == "boolean":
                ids += self.encode(forced)
                parameters[arg] = self.choose(ids, ["true", "false"]) \
                    == "true"
                forced = ""
            elif spec.enum:
                ids += self.encode(forced + '"')
                options = [value + '"' for value in spec.enum]
                parameters[arg] = self.choose(ids, options)[:-1]
                forced = ""
            else:  # "string" and any unknown type
                ids += self.encode(forced + '"')
                parameters[arg] = self.generate_string(ids)
                forced = '"'
        return FunctionCall(
            prompt=request, name=function.name, parameters=parameters
        )
