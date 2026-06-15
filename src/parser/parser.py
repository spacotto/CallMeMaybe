import json
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

from src.utils import error
from src.parser.models.function_definition import FunctionDefinition


class SchemaParser(BaseModel):
    file_path: str = Field(
        ...,
        description="Path to the functions definition JSON file"
    )

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

            # Enforce strict Pydantic validation on the input schema
            validated_schema = []
            for item in raw_data:
                try:
                    # Validate the structure using Pydantic
                    validated_item = FunctionDefinition(**item)
                    # Dump back to dict format for the rest of the pipeline
                    validated_schema.append(
                            validated_item.model_dump(exclude_none=True)
                        )
                except ValidationError as ve:
                    func_name = item.get("name", "Unknown Function")
                    error("Pydantic Validation Error in schema "
                          f"'{func_name}': {ve}")
                    # Skip malformed functions to prevent downstream crashes
                    continue

            return validated_schema

        except json.JSONDecodeError as e:
            error(f"Failed to parse JSON schema: Syntax error at {e}")
            return []
        except Exception as e:
            error(f"Unexpected I/O error reading schema: {e}")
            return []

    @staticmethod
    def is_nested(
        target_name: str, functions_schema: List[Dict[str, Any]]
    ) -> bool:
        for f_schema in functions_schema:
            if f_schema.get("name") == target_name:
                params = f_schema.get("parameters", {})
                for key, attr in params.items():
                    if isinstance(attr, dict) and attr.get("type") == "object":
                        return True
        return False
