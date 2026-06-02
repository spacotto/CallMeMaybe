from typing import Optional, List, Any
from pydantic import BaseModel


class FunctionProperty(BaseModel):
    """
    Represents a single parameter's properties within a function.
    """
    type: Optional[str] = "string"
    description: Optional[str] = None
    enum: Optional[List[Any]] = None
