from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from src.parser.models.function_property import FunctionProperty


class FunctionParameters(BaseModel):
    """
    Represents the 'parameters' block of a function schema.
    """
    type: Optional[str] = "object"
    properties: Optional[Dict[str, FunctionProperty]] = Field(
        default_factory=dict
    )
    required: Optional[List[str]] = Field(default_factory=list)
