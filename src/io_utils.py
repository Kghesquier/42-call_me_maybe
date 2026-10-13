"""Reading the input files and writing the output file, safely."""

import json
import sys
from pathlib import Path
from typing import Any, List

from pydantic import ValidationError

from src.models import FunctionCall, FunctionDefinition, PromptItem


class InputError(Exception):
    """Raised for any problem with the user-provided files.

    The message is meant to be shown to the user as-is.
    """


def read_json(path: Path, label: str) -> Any:
    """Load a JSON file and turn every failure into an InputError.

    Args:
        path: File to read.
        label: Human-readable name of the file, used in error messages.

    Returns:
        The parsed JSON value.

    Raises:
        InputError: If the file is missing, unreadable or not valid JSON.
    """
    if not path.exists():
        raise InputError(f"{label} file not found: {path}")
    if not path.is_file():
        raise InputError(f"{label} path is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{label} file is not valid JSON ({path}, line {exc.lineno}, "
            f"column {exc.colno}): {exc.msg}"
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"cannot read {label} file {path}: {exc}") from exc


def _short_validation_message(exc: ValidationError) -> str:
    """Summarize a pydantic error in one line.

    Args:
        exc: The validation error.

    Returns:
        A compact description of the first problems found.
    """
    parts = []
    for error in exc.errors()[:3]:
        location = ".".join(str(item) for item in error["loc"]) or "value"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


def load_functions(path: Path) -> List[FunctionDefinition]:
    """Load and validate the function definitions.

    Args:
        path: Path of functions_definition.json.

    Returns:
        The validated function definitions.

    Raises:
        InputError: If the file is invalid or defines no usable function.
    """
    data = read_json(path, "functions definition")
    if not isinstance(data, list):
        raise InputError(
            "functions definition file must contain a JSON array"
        )
    functions: List[FunctionDefinition] = []
    for index, raw in enumerate(data):
        try:
            functions.append(FunctionDefinition.model_validate(raw))
        except ValidationError as exc:
            raise InputError(
                f"function #{index} in {path} is invalid: "
                f"{_short_validation_message(exc)}"
            ) from exc
    if not functions:
        raise InputError(f"no function defined in {path}")
    names = [function.name for function in functions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise InputError(
            f"duplicate function names in {path}: {', '.join(duplicates)}"
        )
    return functions


def load_prompts(path: Path) -> List[PromptItem]:
    """Load the prompts to process; invalid entries are skipped.

    Args:
        path: Path of function_calling_tests.json.

    Returns:
        The valid prompts, in file order.

    Raises:
        InputError: If the file itself is unusable.
    """
    data = read_json(path, "input")
    if not isinstance(data, list):
        raise InputError("input file must contain a JSON array")
    prompts: List[PromptItem] = []
    for index, raw in enumerate(data):
        if isinstance(raw, str):
            raw = {"prompt": raw}
        try:
            prompts.append(PromptItem.model_validate(raw))
        except ValidationError as exc:
            print(
                f"Warning: skipping input entry #{index}: "
                f"{_short_validation_message(exc)}",
                file=sys.stderr,
            )
    return prompts


def write_results(path: Path, results: List[FunctionCall]) -> None:
    """Write the function calls as a JSON array.

    Args:
        path: Output file; parent directories are created if needed.
        results: The calls to write.

    Raises:
        InputError: If the file cannot be written.
    """
    payload = [result.model_dump() for result in results]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            # allow_nan=False: NaN/Infinity are not valid JSON.
            json.dump(payload, handle, indent=2, ensure_ascii=False,
                      allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot write output file {path}: {exc}") from exc
