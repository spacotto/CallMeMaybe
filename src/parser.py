# ----------------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------------

import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
#  Pydantic Schemas
# ----------------------------------------------------------------------------

class PromptItem(BaseModel):
    prompt: str

class FunctionParameter(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, str]

# ----------------------------------------------------------------------------
#  Parsing Functions
# ----------------------------------------------------------------------------

def load_function_definitions(filepath: str) -> List[FunctionDefinition]:
    """Safely loads and validates the available functions schema."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Functions definition file not found at '{filepath}'")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Functions definition root must be a JSON array.")

        return [FunctionDefinition(**item) for item in data]

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON syntax in functions file: {e}")


def load_input_prompts(filepath: str) -> List[PromptItem]:
    """Safely loads and validates the target user evaluation prompts."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input prompts file not found at '{filepath}'")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Input prompts root must be a JSON array.")

        return [PromptItem(**item) for item in data]

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON syntax in prompts file: {e}")
