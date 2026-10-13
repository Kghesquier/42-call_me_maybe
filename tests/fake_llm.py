"""A tiny stand-in for ``llm_sdk.Small_LLM_Model`` used by the tests.

It exposes the same public methods (``encode``, ``decode``,
``get_logits_from_input_ids``, ``get_path_to_vocab_file``) and writes a
``vocab.json`` in the same byte-level format as Qwen's.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from src.vocabulary import bytes_to_unicode

SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>"]
MULTI_CHAR_TOKENS = [
    '{"', '"name', '": "', "fn_", "add", "_numbers", "greet", '", ',
    '"}', "}}", " 2", "\\\\", "\\d", "\\", "12", '"hello', "hello",
    ', "', "parameters", '": {"', "true", "false", "Ġ", "é", "\n\n",
    "</think>", "<think>", "\n", '""', '\\"', "u00e9",
]


class FakeLLM:
    """Deterministic fake model.

    Args:
        workdir: Directory where vocab.json is written.
        targets: Optional mapping request -> JSON text the model "wants"
            to write. Without it the logits are random.
        seed: Seed for random logits.
    """

    def __init__(self, workdir: Path,
                 targets: Optional[Dict[str, str]] = None,
                 seed: int = 0) -> None:
        """Build the vocabulary and write it to disk."""
        texts: List[str] = [chr(c) for c in range(32, 127)]
        texts += [t for t in MULTI_CHAR_TOKENS if t not in texts]
        self.texts = texts
        self.ids: Dict[str, int] = {t: i for i, t in enumerate(texts)}
        for offset, special in enumerate(SPECIAL_TOKENS):
            self.ids[special] = len(texts) + offset
        self.size = len(texts) + len(SPECIAL_TOKENS) + 3  # padding ids
        table = bytes_to_unicode()
        vocab = {
            "".join(table[b] for b in text.encode("utf-8")): index
            for index, text in enumerate(texts)
        }
        self.vocab_path = workdir / "vocab.json"
        self.vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
        self.targets = targets or {}
        self.rng = random.Random(seed)
        self.calls = 0

    def encode(self, text: str) -> List[List[int]]:
        """Greedy longest-match tokenization (2-D like the real SDK)."""
        result: List[int] = []
        position = 0
        keys = sorted(self.ids, key=len, reverse=True)
        while position < len(text):
            for key in keys:
                if text.startswith(key, position):
                    result.append(self.ids[key])
                    position += len(key)
                    break
            else:
                position += 1  # unknown char: dropped
        return [result]

    def decode(self, ids: List[int]) -> str:
        """Inverse of encode (special tokens included)."""
        reverse = {index: text for text, index in self.ids.items()}
        return "".join(reverse.get(i, "") for i in ids)

    def get_path_to_vocab_file(self) -> str:
        """Return the path of the generated vocab.json."""
        return str(self.vocab_path)

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """Random logits, or logits pushing towards the target text."""
        self.calls += 1
        logits = [self.rng.uniform(-1.0, 1.0) for _ in range(self.size)]
        context = self.decode(input_ids)
        request = context.split("<|im_start|>user\n")[-1]
        request = request.split("<|im_end|>")[0]
        target = self.targets.get(request)
        if target is None:
            return logits
        generated = context.split("</think>\n\n")[-1]
        if not target.startswith(generated):
            return logits
        needed = target[len(generated):]
        for index, text in enumerate(self.texts):
            if text and needed.startswith(text):
                logits[index] = 10.0 + len(text)
        return logits
