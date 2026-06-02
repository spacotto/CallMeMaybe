import json
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

from src.parser.models import FunctionDefinition
from src.utils import error, warning


class SchemaParser(BaseModel):
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

            for idx, func_dict in enumerate(raw_data):
                try:
                    validated_func = FunctionDefinition(**func_dict)
                    valid_functions.append(validated_func.model_dump(exclude_none=True))

                except ValidationError as ve:
                    missing_field = ve.errors()[0].get('loc', ['Unknown'])[0]
                    error_msg = ve.errors()[0].get('msg', 'Validation error')
                    warning(f"Skipping function at index {idx}: Field '{missing_field}' -> {error_msg}")
                    continue

            return valid_functions

        except json.JSONDecodeError as e:
            error(f"Failed to parse JSON schema: Syntax error at {e}")
            return []
        except Exception as e:
            error(f"Unexpected I/O error reading schema: {e}")
            return []
