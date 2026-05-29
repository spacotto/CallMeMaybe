import json
import os
from typing import List, Dict, Any
from src.utils import error, warning

class SchemaParser:
    def __init__(self, file_path: str = "data/input/functions_definition.json") -> None:
        """
        Initializes the parser with the target path to the JSON schema file.
        """
        self.file_path = file_path

    def load_functions(self) -> List[Dict[str, Any]]:
        """
        Safely ingests and validates the local JSON schema file.
        Returns an empty list if the ingestion fails, preventing pipeline crashes.
        """
        if not os.path.exists(self.file_path):
            error(f"Schema file not found at: {self.file_path}")
            return []

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Strict structural validation
            if not isinstance(data, list):
                error("Invalid schema format: The root JSON element must be a list of function dictionaries.")
                return []

            # Filter out malformed nodes
            valid_functions = []
            for func in data:
                if not isinstance(func, dict) or "name" not in func:
                    warning(f"Skipping malformed function definition missing a 'name' key.")
                    continue
                valid_functions.append(func)

            return valid_functions

        except json.JSONDecodeError as e:
            error(f"Failed to parse JSON schema: Syntax error at {e}")
            return []
        except Exception as e:
            error(f"Unexpected I/O error reading schema: {e}")
            return []
