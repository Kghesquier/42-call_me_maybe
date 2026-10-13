"""Read the input files and write the output file.

Every problem is turned into a ``ValueError`` with a clear message;
``__main__`` prints it and exits without a traceback.
"""

import json
from pathlib import Path
from typing import Any, List

from pydantic import ValidationError

from src.models import FunctionCall, FunctionDefinition, PromptItem


def read_json(path: Path) -> Any:
    """Load a JSON file.

    Args:
        path: File to read.

    Returns:
        The parsed JSON value.

    Raises:
        ValueError: If the file is missing, unreadable or invalid JSON.
    """
    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def load_functions(path: Path) -> List[FunctionDefinition]:
    """Load and validate functions_definition.json.

    Args:
        path: Path of the file.

    Returns:
        The function definitions.

    Raises:
        ValueError: If the file is invalid or contains no function.
    """
    data = read_json(path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must be a non-empty JSON array")
    try:
        functions = [FunctionDefinition.model_validate(f) for f in data]
    except ValidationError as exc:
        raise ValueError(f"invalid function in {path}: {exc}") from exc
    names = [f.name for f in functions]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate function names in {path}")
    return functions


def load_prompts(path: Path) -> List[PromptItem]:
    """Load and validate function_calling_tests.json.

    Args:
        path: Path of the file.

    Returns:
        The prompts.

    Raises:
        ValueError: If the file is invalid.
    """
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    try:
        return [PromptItem.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ValueError(f"invalid prompt in {path}: {exc}") from exc


def write_results(path: Path, results: List[FunctionCall]) -> None:
    """Write the results as a JSON array (creates the directory).

    Args:
        path: Output file.
        results: The function calls.

    Raises:
        ValueError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump([r.model_dump() for r in results], handle, indent=2)
    except OSError as exc:
        raise ValueError(f"cannot write {path}: {exc}") from exc
