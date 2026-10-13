"""Vocabulary loading: map every token id to the text it represents.

Qwen uses a byte-level BPE tokenizer (same scheme as GPT-2). In
``vocab.json`` each token is written with a special alphabet where every
byte is replaced by a printable unicode character (for example a space
becomes ``Ġ`` and a newline becomes ``Ċ``). To know which characters a
token really produces we must undo that mapping.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel


def bytes_to_unicode() -> Dict[int, str]:
    """Build the GPT-2 byte -> printable character table.

    Returns:
        A dict mapping each of the 256 byte values to one character.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = printable[:]
    code_points = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            code_points.append(256 + shift)
            shift += 1
    return {b: chr(c) for b, c in zip(byte_values, code_points)}


def decode_token(token: str, table: Dict[str, int]) -> str:
    """Convert one vocab.json token into the text it produces.

    Args:
        token: Token as written in vocab.json.
        table: Inverse of :func:`bytes_to_unicode`.

    Returns:
        The decoded text, or ``""`` when the token is not a complete UTF-8
        sequence (such tokens are never allowed by the decoder).
    """
    try:
        raw = bytes(table[char] for char in token)
        return raw.decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return ""


class Vocabulary(BaseModel):
    """Token id -> decoded text, for every token of the model.

    Attributes:
        token_texts: ``token_texts[i]`` is the text of token ``i``; empty
            for ids that are unknown or not decodable.
    """

    token_texts: List[str]

    @classmethod
    def from_file(cls, path: str) -> "Vocabulary":
        """Load ``vocab.json`` (or a full ``tokenizer.json``).

        Args:
            path: Path returned by ``get_path_to_vocab_file``.

        Returns:
            The loaded vocabulary.

        Raises:
            ValueError: If the file does not look like a vocabulary.
            OSError: If the file cannot be read.
        """
        with Path(path).open("r", encoding="utf-8") as handle:
            data: Any = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("model"), dict):
            data = data["model"].get("vocab")
        if not isinstance(data, dict) or not data:
            raise ValueError(f"{path} is not a token -> id vocabulary")
        table = {char: byte for byte, char in bytes_to_unicode().items()}
        size = max(int(index) for index in data.values()) + 1
        texts = [""] * size
        for token, index in data.items():
            texts[int(index)] = decode_token(str(token), table)
        return cls(token_texts=texts)

    def __len__(self) -> int:
        """Return the number of token ids covered."""
        return len(self.token_texts)
