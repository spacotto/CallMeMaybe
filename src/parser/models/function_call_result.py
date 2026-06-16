"""
Defines the output schema for the constrained decoding pipeline.

This module contains the Pydantic model used by the PostProcessor to
validate and enforce the final structure of the generated function calls
before they are serialized to disk.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field


class FunctionCallResult(BaseModel):
    """
    Validates and enforces the mandatory output structure for the project.

    This model acts as the final structural guardrail in the pipeline.
    Every generated result must exactly match this schema to ensure
    downstream compatibility and absolute JSON compliance.

    Attributes:
        prompt (str): The original natural-language request from the user.
        name (str): The name of the targeted function to execute.
        parameters (Dict[str, Any]): The extracted arguments, correctly
            typed and mapped according to the function's definition.
    """
    prompt: str = Field(
        ...,
        description="The original natural-language request from the user."
    )
    name: str = Field(
        ...,
        description="The name of the targeted function to execute."
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description="All required args casted to their correct types."
    )
