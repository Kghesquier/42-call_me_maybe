"""Pydantic models for the input files and the output file."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ParameterSpec(BaseModel):
    """Type of one function argument (or of the return value).

    Attributes:
        type: "number", "integer", "string" or "boolean".
        enum: Optional list of the only values allowed (strings).
    """

    type: str = "string"
    enum: Optional[List[str]] = None


class FunctionDefinition(BaseModel):
    """One function from functions_definition.json."""

    name: str = Field(min_length=1)
    description: str = ""
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: Optional[ParameterSpec] = None


class PromptItem(BaseModel):
    """One entry of function_calling_tests.json."""

    prompt: str


class FunctionCall(BaseModel):
    """One entry of the output file."""

    prompt: str
    name: str
    parameters: Dict[str, Any]
