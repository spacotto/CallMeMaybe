"""
Defines the data model for the parameters block of a function.

This module contains the Pydantic model representing the overarching
'parameters' dictionary in a standard JSON schema, tracking required
fields and containing nested parameter definitions.
"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from src.parser.models.function_property import FunctionProperty


class FunctionParameters(BaseModel):
    """
    Represents the 'parameters' block of a function schema.

    This model validates the structural container that holds all
    individual arguments for a specific function, enforcing the list
    of required keys.

    Attributes:
        type (Optional[str]): The overarching type of the parameters block.
            Defaults to "object".
        properties (Optional[Dict[str, FunctionProperty]]): A dictionary
            mapping parameter names to their specific property definitions.
        required (Optional[List[str]]): A strictly defined list of parameter
            names that the LLM must generate for a valid function call.
    """
    type: Optional[str] = "object"
    properties: Optional[Dict[str, FunctionProperty]] = Field(
        default_factory=dict
    )
    required: Optional[List[str]] = Field(default_factory=list)
