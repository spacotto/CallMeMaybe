import json
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from src.utils import error, warning


# ----------------------------------------------------------------------------
# Pydantic Schemas for JSON Validation
# ----------------------------------------------------------------------------

class FunctionProperty(BaseModel):
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None


class FunctionParameters(BaseModel):
    type: str
    properties: Dict[str, FunctionProperty]
    required: List[str] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = ""
    parameters: Optional[FunctionParameters] = None


# ----------------------------------------------------------------------------
# The Pydantic-Validated Parser Class
# ----------------------------------------------------------------------------

class SchemaParser(BaseModel):
    """
    Inheriting from BaseModel ensures that whenever SchemaParser is instantiated,
    the file_path is strictly validated as a string.
    """
    file_path: str = Field(..., description="Path to the functions definition JSON file")

    def load_functions(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            error(f"Schema file not found at: {self.file_path}")
            return []

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            if not isinstance(raw_data, list):
                error("Invalid schema format: Root element must be a list.")
                return []

            valid_functions: List[Dict[str, Any]] = []

            # Pydantic physically enforces the schema on every dictionary
            for idx, func_dict in enumerate(raw_data):

                try:
                    # If this succeeds, the dictionary perfectly matches the required structure
                    validated_func = FunctionDefinition(**func_dict)

                    # We dump it back to a dictionary to pass to the engine
                    # exclude_none=True removes empty fields to keep the prompt clean
                    valid_functions.append(validated_func.model_dump(exclude_none=True))

                except ValidationError as ve:
                    warning(f"Skipping malformed function at index {idx}: {ve.errors()[0]['msg']}")
                    continue

            return valid_functions

        except json.JSONDecodeError as e:
            error(f"Failed to parse JSON schema: Syntax error at {e}")
            return []

       except Exception as e:
            error(f"Unexpected I/O error reading schema: {e}")
            return []
