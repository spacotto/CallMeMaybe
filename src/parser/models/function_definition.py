from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from src.parser.models.function_property import FunctionProperty

class FunctionDefinition(BaseModel):
    """
    The top-level model representing a complete callable function.
    """
    name: str
    description: Optional[str] = ""
    parameters: Optional[Dict[str, FunctionProperty]] = Field(default_factory=dict)
    returns: Optional[Dict[str, Any]] = None
