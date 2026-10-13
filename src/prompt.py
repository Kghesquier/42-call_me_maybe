"""Build the text given to the model before the JSON is generated."""

from typing import List

from src.models import FunctionDefinition


def build_prompt(request: str, functions: List[FunctionDefinition]) -> str:
    """Create the Qwen3 chat prompt listing the available functions.

    The empty ``<think></think>`` block turns off Qwen3's reasoning mode
    so that the model answers directly.

    Args:
        request: The user's natural-language prompt.
        functions: All functions the model may call.

    Returns:
        The prompt text.
    """
    lines = []
    for f in functions:
        args = ", ".join(f"{n}: {p.type}" for n, p in f.parameters.items())
        lines.append(f"- {f.name}({args}): {f.description}")
    catalog = "\n".join(lines)
    return (
        "<|im_start|>system\n"
        "Translate the user request into ONE function call written as "
        'JSON: {"name": ..., "parameters": {...}}. Pick the function '
        "whose description matches the request and copy the argument "
        "values from the request.\n"
        f"Available functions:\n{catalog}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
