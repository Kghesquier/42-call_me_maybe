"""Entry point: ``uv run python -m src [options]``."""

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

from src.decoder import ConstrainedDecoder
from src.io_utils import (
    InputError,
    load_functions,
    load_prompts,
    write_results,
)
from src.models import SUPPORTED_TYPES, FunctionCall, FunctionDefinition
from src.vocabulary import Vocabulary

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments to parse (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function "
                    "calls using constrained decoding.",
    )
    parser.add_argument("--functions_definition", type=Path,
                        default=DEFAULT_FUNCTIONS,
                        help="JSON file describing the functions")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="JSON file with the prompts")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="where to write the results")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Hugging Face model id (default: Qwen3-0.6B)")
    parser.add_argument("--verbose", action="store_true",
                        help="show every generated token")
    return parser.parse_args(argv)


def warn_unsupported_types(functions: List[FunctionDefinition]) -> None:
    """Tell the user about argument types that will be treated as strings.

    Args:
        functions: The loaded definitions.
    """
    for function in functions:
        for name, spec in function.parameters.items():
            if spec.type not in SUPPORTED_TYPES:
                print(f"Warning: {function.name}.{name} has unsupported "
                      f"type '{spec.type}', generated as a string",
                      file=sys.stderr)


def load_model(model_name: str) -> Any:
    """Import the SDK and load the LLM.

    Args:
        model_name: Hugging Face model id.

    Returns:
        A ``Small_LLM_Model`` instance.

    Raises:
        InputError: If the SDK or the model cannot be loaded.
    """
    try:
        sdk = importlib.import_module("llm_sdk")
        model_class = getattr(sdk, "Small_LLM_Model")
    except (ImportError, AttributeError) as exc:
        raise InputError(
            f"cannot import llm_sdk ({exc}); run 'uv sync' first"
        ) from exc
    try:
        return model_class(model_name)
    except Exception as exc:  # the SDK may raise many different errors
        raise InputError(
            f"cannot load model '{model_name}': {exc}"
        ) from exc


def run(args: argparse.Namespace) -> int:
    """Run the whole pipeline.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code (0 on success).
    """
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)
    warn_unsupported_types(functions)
    print(f"Loaded {len(functions)} functions and {len(prompts)} prompts",
          file=sys.stderr)

    llm = load_model(args.model)
    try:
        vocab = Vocabulary.from_file(llm.get_path_to_vocab_file())
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot load the vocabulary: {exc}") from exc
    decoder = ConstrainedDecoder(llm=llm, vocab=vocab, verbose=args.verbose)

    results: List[FunctionCall] = []
    start = time.perf_counter()
    for index, item in enumerate(prompts, start=1):
        print(f"[{index}/{len(prompts)}] {item.prompt}", file=sys.stderr)
        try:
            call = decoder.decode_call(item.prompt, functions)
        except Exception as exc:  # one bad prompt must not stop the run
            print(f"  error: {exc}", file=sys.stderr)
            continue
        print(f"  -> {call.name}({call.parameters})", file=sys.stderr)
        results.append(call)

    write_results(args.output, results)
    elapsed = time.perf_counter() - start
    print(f"Wrote {len(results)} results to {args.output} "
          f"in {elapsed:.1f}s", file=sys.stderr)
    return 0


def main() -> None:
    """Run the program and convert every failure into a clean message."""
    try:
        code = run(parse_args())
    except InputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        code = 130
    except Exception as exc:  # last safety net: never show a traceback
        print(f"Unexpected error: {exc}", file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
