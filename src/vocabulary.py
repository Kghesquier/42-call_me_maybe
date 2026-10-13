"""Load the model vocabulary: token id -> text of the token.

Qwen uses byte-level BPE (like GPT-2). In vocab.json every byte is
written as a printable character: a space is ``Ġ``, a newline is ``Ċ``.
We undo this mapping to know the real text of every token.
"""

import json
from typing import Dict, List


def bytes_to_unicode() -> Dict[int, str]:
    """Build the GPT-2 table: byte value -> printable character.

    Printable bytes keep their own character; the others (space,
    control chars...) are moved to the characters starting at 256.

    Returns:
        A dict with the 256 byte values as keys.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table: Dict[int, str] = {}
    shift = 0
    for byte in range(256):
        if byte in printable:
            table[byte] = chr(byte)
        else:
            table[byte] = chr(256 + shift)
            shift += 1
    return table


def load_vocabulary(path: str) -> List[str]:
    """Read vocab.json and decode every token.

    Args:
        path: Path given by ``get_path_to_vocab_file()``.

    Returns:
        ``tokens[i]`` is the text of token ``i``. It is ``""`` when the
        token is not valid UTF-8 on its own (such tokens are never used).
    """
    with open(path, "r", encoding="utf-8") as handle:
        vocab: Dict[str, int] = json.load(handle)
    char_to_byte = {char: byte for byte, char in bytes_to_unicode().items()}
    tokens = [""] * (max(vocab.values()) + 1)
    for token, token_id in vocab.items():
        try:
            raw = bytes(char_to_byte[char] for char in token)
            tokens[token_id] = raw.decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            pass
    return tokens
