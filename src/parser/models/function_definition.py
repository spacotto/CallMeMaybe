"""
Defines the top-level data model for a complete function schema.

This module contains the primary Pydantic model used by the parser to
ingest, validate, and structure the root elements of each function
definition provided by the user.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from src.parser.models.function_property import FunctionProperty


class FunctionDefinition(BaseModel):
    """
    The top-level model representing a complete callable function.

    This model acts as the root blueprint for an available tool. It ensures
    that every function passed to the LLM masking engine has a guaranteed
    name and safely defaulted parameter blocks.

    Attributes:
        name (str): The exact, required string identifier for the function.
        description (Optional[str]): The prompt-injected explanation of what
            the function does. Defaults to an empty string.
        parameters (Optional[Dict[str, FunctionProperty]]): The dictionary
            defining the schema of expected arguments. Defaults to empty.
        returns (Optional[Dict[str, Any]]): The expected return type of the
            function, if defined by the schema.
    """
    name: str
    description: Optional[str] = ""
    parameters: Optional[Dict[str, FunctionProperty]] = Field(
        default_factory=dict
    )
    returns: Optional[Dict[str, Any]] = None
