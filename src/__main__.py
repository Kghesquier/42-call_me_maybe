"""Entry point: ``uv run python -m src [options]``."""

import argparse
import sys
from pathlib import Path
from typing import Any, List

from src.decoder import ConstrainedDecoder
from src.io_utils import load_functions, load_prompts, write_results
from src.models import FunctionCall
from src.vocabulary import load_vocabulary


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(prog="python -m src")
    parser.add_argument(
        "--functions_definition", type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/output/function_calling_results.json"),
    )
    return parser.parse_args()


def load_model() -> Any:
    """Load Qwen/Qwen3-0.6B through the SDK.

    Returns:
        A ``Small_LLM_Model``.

    Raises:
        ValueError: If the SDK or the model cannot be loaded.
    """
    try:
        from llm_sdk import Small_LLM_Model
        return Small_LLM_Model("Qwen/Qwen3-0.6B")
    except Exception as exc:
        raise ValueError(f"cannot load the model: {exc}") from exc


def run(args: argparse.Namespace) -> None:
    """Process every prompt and write the output file.

    Args:
        args: The command-line options.
    """
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    llm = load_model()
    tokens = load_vocabulary(llm.get_path_to_vocab_file())
    decoder = ConstrainedDecoder(llm=llm, tokens=tokens)

    results: List[FunctionCall] = []
    for index, item in enumerate(prompts, start=1):
        print(f"[{index}/{len(prompts)}] {item.prompt}")
        try:
            call = decoder.decode_call(item.prompt, functions)
        except Exception as exc:  # one bad prompt must not stop the run
            print(f"  error: {exc}", file=sys.stderr)
            continue
        print(f"  -> {call.name} {call.parameters}")
        results.append(call)

    write_results(args.output, results)
    print(f"Results written to {args.output}")


def main() -> None:
    """Run the program; print a clear message instead of crashing."""
    try:
        run(parse_args())
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
