from typing import Optional
from pydantic import BaseModel, Field
from src.parser.models.function_parameters import FunctionParameters


class FunctionDefinition(BaseModel):
    """
    The top-level model representing a complete callable function.
    """
    name: str
    description: Optional[str] = ""
    parameters: Optional[FunctionParameters] = Field(default_factory=FunctionParameters)
