"""Build the text given to the model before constrained generation."""

from typing import List

from src.models import FunctionDefinition

SYSTEM_RULES = (
    "You translate a user request into ONE function call, written as "
    'JSON: {"name": <function name>, "parameters": {<arguments>}}.\n'
    "Pick the function whose description matches the request. "
    "Copy argument values exactly from the request, without the "
    "surrounding quotes. Numbers are written as plain digits. Inside "
    "JSON strings a backslash is written \\\\ (so the regex \\d+ is "
    'written "\\\\d+").'
)


def describe_function(function: FunctionDefinition) -> str:
    """Render one function as a short signature line.

    Args:
        function: The function to describe.

    Returns:
        A line such as ``- fn_add(a: number, b: number): Add two numbers``.
    """
    arguments = ", ".join(
        f"{name}: {spec.type}" for name, spec in function.parameters.items()
    )
    return f"- {function.name}({arguments}): {function.description}"


def build_prompt(request: str, functions: List[FunctionDefinition]) -> str:
    """Create the full chat prompt for Qwen3.

    The assistant turn is opened with an empty ``<think>`` block, which
    tells Qwen3 to skip its reasoning mode and answer directly. The JSON
    itself is then written by the constrained decoder.

    Args:
        request: The user's natural-language prompt.
        functions: All functions the model may call.

    Returns:
        The prompt text, ready to be encoded.
    """
    catalog = "\n".join(describe_function(f) for f in functions)
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_RULES}\nAvailable functions:\n{catalog}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
