from typing import Dict, Any
from pydantic import BaseModel, Field


class FunctionCallResult(BaseModel):
    """
    Validates and enforces the mandatory output structure for the project.
    Every recorded item must exactly match this schema.
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
