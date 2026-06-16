"""
Defines the data model for individual function parameters.

This module contains the Pydantic model used to validate the attributes
of a single parameter within a JSON schema, including its type,
description, and any deeply nested properties.
"""

from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field


class FunctionProperty(BaseModel):
    """
    Represents a single parameter's properties within a function.

    This model strictly types the expected schema fields for an individual
    argument, ensuring that nested objects and enums are safely parsed
    without raising unexpected runtime exceptions.

    Attributes:
        type (Optional[str]): The data type of the parameter (e.g., "string",
            "integer", "object"). Defaults to "string".
        description (Optional[str]): A natural language explanation of the
            parameter's purpose to guide the LLM.
        enum (Optional[List[Any]]): A restricted list of valid values for
            this parameter, if applicable.
        properties (Optional[Dict[str, Any]]): A nested dictionary of further
            properties if the parameter type is designated as an "object".
    """
    type: Optional[str] = "string"
    description: Optional[str] = None
    enum: Optional[List[Any]] = None

    properties: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("Nested properties dictionary "
                     "if the type is designated as an object.")
    )
