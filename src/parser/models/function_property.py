from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field


class FunctionProperty(BaseModel):
    """
    Represents a single parameter's properties within a function.
    """
    type: Optional[str] = "string"
    description: Optional[str] = None
    enum: Optional[List[Any]] = None

    properties: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("Nested properties dictionary "
                     "if the type is designated as an object.")
    )
