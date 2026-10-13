"""Tests for the vocabulary, the grammars and the constrained decoder."""

import json
from pathlib import Path
from typing import Dict, List

import pytest

from src.decoder import (
    STATE_ESCAPE,
    STATE_NORMAL,
    ConstrainedDecoder,
    advance_string_state,
    repair_number,
    repair_string,
)
from src.models import FunctionDefinition
from src.vocabulary import Vocabulary, bytes_to_unicode, decode_token
from tests.fake_llm import FakeLLM

FUNCTIONS: List[FunctionDefinition] = [
    FunctionDefinition.model_validate(raw) for raw in [
        {"name": "fn_add_numbers", "description": "Add two numbers.",
         "parameters": {"a": {"type": "number"}, "b": {"type": "number"}}},
        {"name": "fn_greet", "description": "Greet someone.",
         "parameters": {"name": {"type": "string"}}},
        {"name": "fn_substitute_string_with_regex",
         "description": "Regex replace.",
         "parameters": {"source_string": {"type": "string"},
                        "regex": {"type": "string"},
                        "replacement": {"type": "string"}}},
        {"name": "fn_is_even", "description": "Check parity.",
         "parameters": {"n": {"type": "integer"},
                        "strict": {"type": "boolean"}}},
        {"name": "fn_add", "description": "Prefix of another name.",
         "parameters": {}},
    ]
]


def make_decoder(tmp_path: Path, targets: Dict[str, str],
                 seed: int = 0) -> ConstrainedDecoder:
    """Build a decoder on top of the fake model."""
    llm = FakeLLM(tmp_path, targets=targets, seed=seed)
    vocab = Vocabulary.from_file(llm.get_path_to_vocab_file())
    return ConstrainedDecoder(llm=llm, vocab=vocab)


def test_byte_level_decoding() -> None:
    """``Ġ`` stands for a space in vocab.json."""
    table = {c: b for b, c in bytes_to_unicode().items()}
    assert decode_token("Ġhello", table) == " hello"
    assert decode_token("Ċ", table) == "\n"


def test_string_state_machine() -> None:
    """Escapes and closing quotes are tracked correctly."""
    assert advance_string_state(STATE_NORMAL, "abc") == (STATE_NORMAL, -1)
    assert advance_string_state(STATE_NORMAL, 'ab"x') == (STATE_NORMAL, 2)
    assert advance_string_state(STATE_NORMAL, "\\")[0] == STATE_ESCAPE
    assert advance_string_state(STATE_ESCAPE, "d")[0] is None
    assert advance_string_state(STATE_ESCAPE, '"x') == (STATE_NORMAL, -1)
    assert advance_string_state(STATE_NORMAL, "\\u00e9") == (
        STATE_NORMAL, -1)
    assert advance_string_state(STATE_NORMAL, "\\u00g")[0] is None
    assert advance_string_state(STATE_NORMAL, "a\nb")[0] is None


def test_repairs() -> None:
    """Truncated values are turned into valid ones."""
    assert repair_string("abc\\") == "abc"
    assert repair_string("\\\\d+") == "\\d+"
    assert repair_number("12.", integer=False) == "12"
    assert repair_number("-", integer=False) == "0"
    assert repair_number("3e", integer=False) == "3"


@pytest.mark.parametrize("request_text, target, name, parameters", [
    ("What is the sum of 2 and 12?",
     '{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 12}}',
     "fn_add_numbers", {"a": 2.0, "b": 12.0}),
    ("Greet shrek",
     '{"name": "fn_greet", "parameters": {"name": "shrek"}}',
     "fn_greet", {"name": "shrek"}),
    ("Replace digits in 'a1' with X",
     '{"name": "fn_substitute_string_with_regex", "parameters": '
     '{"source_string": "a1", "regex": "\\\\d+", "replacement": "X"}}',
     "fn_substitute_string_with_regex",
     {"source_string": "a1", "regex": "\\d+", "replacement": "X"}),
    ("Is -12 even, strictly?",
     '{"name": "fn_is_even", "parameters": {"n": -12, "strict": true}}',
     "fn_is_even", {"n": -12, "strict": True}),
    ("Call fn_add",
     '{"name": "fn_add", "parameters": {}}', "fn_add", {}),
])
def test_guided_model_gets_expected_call(
    tmp_path: Path, request_text: str, target: str, name: str,
    parameters: Dict[str, object],
) -> None:
    """A model that 'knows' the answer is decoded into that exact call."""
    decoder = make_decoder(tmp_path, {request_text: target})
    call = decoder.decode_call(request_text, FUNCTIONS)
    assert call.name == name
    assert call.parameters == parameters
    for key, value in parameters.items():
        assert type(call.parameters[key]) is type(value)


@pytest.mark.parametrize("seed", range(25))
def test_random_model_always_gives_valid_schema(
    tmp_path: Path, seed: int
) -> None:
    """Even with random logits, the output always matches the schema."""
    decoder = make_decoder(tmp_path, {}, seed=seed)
    call = decoder.decode_call("anything", FUNCTIONS)
    function = next(f for f in FUNCTIONS if f.name == call.name)
    assert set(call.parameters) == set(function.parameters)
    expected = {"number": float, "integer": int, "string": str,
                "boolean": bool}
    for key, spec in function.parameters.items():
        assert type(call.parameters[key]) is expected[spec.type]
    text = json.dumps(call.model_dump(), allow_nan=False)
    assert json.loads(text)["name"] == call.name


def test_single_function_needs_no_model_call(tmp_path: Path) -> None:
    """With one possible function the name is forced for free."""
    decoder = make_decoder(tmp_path, {})
    only = [FunctionDefinition(name="fn_ping", parameters={})]
    call = decoder.decode_call("ping", only)
    assert call.name == "fn_ping"
    assert decoder.llm.calls == 0
