"""Tests for input/output handling and for the command-line program."""

import json
import sys
from pathlib import Path
from typing import Any, List

import pytest

import src.__main__ as entry
from src.io_utils import InputError, load_functions, load_prompts
from tests.fake_llm import FakeLLM

DATA = Path(__file__).resolve().parent.parent / "data" / "input"


def test_missing_file(tmp_path: Path) -> None:
    """A missing file gives a clear InputError."""
    with pytest.raises(InputError, match="not found"):
        load_functions(tmp_path / "nope.json")


def test_invalid_json(tmp_path: Path) -> None:
    """Broken JSON gives the line and column of the problem."""
    bad = tmp_path / "bad.json"
    bad.write_text('[{"prompt": "hi",}]', encoding="utf-8")
    with pytest.raises(InputError, match="not valid JSON"):
        load_prompts(bad)


def test_wrong_structure(tmp_path: Path) -> None:
    """Valid JSON with the wrong shape is rejected."""
    bad = tmp_path / "bad.json"
    bad.write_text('{"name": "fn"}', encoding="utf-8")
    with pytest.raises(InputError, match="array"):
        load_functions(bad)
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(InputError, match="no function"):
        load_functions(bad)
    bad.write_text('[{"description": "no name"}]', encoding="utf-8")
    with pytest.raises(InputError, match="invalid"):
        load_functions(bad)


def test_bad_prompt_entries_are_skipped(tmp_path: Path) -> None:
    """Invalid prompt entries are skipped, valid ones kept."""
    path = tmp_path / "prompts.json"
    path.write_text('[{"prompt": "ok"}, {"x": 1}, "plain string"]',
                    encoding="utf-8")
    prompts = load_prompts(path)
    assert [p.prompt for p in prompts] == ["ok", "plain string"]


def run_main(monkeypatch: pytest.MonkeyPatch, argv: List[str],
             llm: Any = None) -> int:
    """Run the entry point and return its exit code."""
    monkeypatch.setattr(sys, "argv", ["src", *argv])
    if llm is not None:
        monkeypatch.setattr(entry, "load_model", lambda name: llm)
    with pytest.raises(SystemExit) as info:
        entry.main()
    return int(info.value.code or 0)


def test_main_end_to_end(tmp_path: Path,
                         monkeypatch: pytest.MonkeyPatch) -> None:
    """The real input files produce a valid output file."""
    output = tmp_path / "out" / "results.json"
    code = run_main(monkeypatch, [
        "--functions_definition", str(DATA / "functions_definition.json"),
        "--input", str(DATA / "function_calling_tests.json"),
        "--output", str(output),
    ], llm=FakeLLM(tmp_path))
    assert code == 0
    results = json.loads(output.read_text(encoding="utf-8"))
    prompts = json.loads(
        (DATA / "function_calling_tests.json").read_text(encoding="utf-8"))
    assert len(results) == len(prompts)
    for result in results:
        assert set(result) == {"prompt", "name", "parameters"}


def test_main_reports_errors(tmp_path: Path,
                             monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture[str]) -> None:
    """Missing files end with exit code 1 and a message, no traceback."""
    code = run_main(monkeypatch, ["--input", str(tmp_path / "x.json")])
    assert code == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "Traceback" not in err
