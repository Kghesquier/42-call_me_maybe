"""Pydantic data models shared by the whole project."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

# Spellings accepted in functions_definition.json, mapped to the four
# JSON types the decoder knows how to generate.
TYPE_ALIASES: Dict[str, str] = {
    "number": "number",
    "float": "number",
    "double": "number",
    "integer": "integer",
    "int": "integer",
    "string": "string",
    "str": "string",
    "boolean": "boolean",
    "bool": "boolean",
}

SUPPORTED_TYPES = ("number", "integer", "string", "boolean")


class ParameterSpec(BaseModel):
    """Type description of one function argument (or return value)."""

    type: str = "string"
    description: str = ""

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: Any) -> str:
        """Lower-case the type name and resolve common aliases.

        Args:
            value: Raw ``type`` field read from the JSON file.

        Returns:
            The normalized type name.

        Raises:
            ValueError: If the type is not a string.
        """
        if not isinstance(value, str):
            raise ValueError("parameter type must be a string")
        cleaned = value.strip().lower()
        return TYPE_ALIASES.get(cleaned, cleaned)


class FunctionDefinition(BaseModel):
    """One callable function, as described in functions_definition.json."""

    name: str = Field(min_length=1)
    description: str = ""
    parameters: Dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: Optional[ParameterSpec] = None


class PromptItem(BaseModel):
    """One natural-language request from function_calling_tests.json."""

    prompt: str


class FunctionCall(BaseModel):
    """The structured function call produced for one prompt."""

    prompt: str
    name: str
    parameters: Dict[str, Any]
