import json
import os
from enum import Enum
from typing import List, Dict, Any


class ModelFormat(Enum):
    CHATML = "chatml"
    INSTRUCT = "instruct"


class Formatter:
    def __init__(self, format_type: ModelFormat = ModelFormat.CHATML) -> None:
        self.format_type = format_type
        self.examples_str = self._load_and_build_examples()

    def _load_and_build_examples(self) -> str:
        """Dynamically loads the few-shot examples from the adjacent JSON asset."""
        # Get the absolute path to the directory containing THIS file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "few_shot.json")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                examples_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load few_shot.json from {json_path}: {e}")
            return "--- EXAMPLES ---\n----------------\n\n"

        # Compile the JSON dicts into the exact string formatting required by the LLM
        formatted = "--- EXAMPLES ---\n"
        for ex in examples_data:
            formatted += f"User: {ex.get('user', '')}\n"
            out_str = json.dumps(ex.get('output', {}))
            formatted += f"Output: {out_str}\n\n"
        formatted += "----------------\n\n"
        return formatted

    def build_function_prompt(self, user_prompt: str, functions: List[Dict[str, Any]]) -> str:
        schema_str = json.dumps(functions, indent=2)

        system_instruction = (
            "You are a logical data extraction engine.\n"
            "You MUST respond with a single, valid JSON object containing ONLY the keys 'name' and 'parameters'.\n"
            "You MUST Map the user's request to the correct function and extract the parameters EXACTLY as defined in the schema.\n\n"
            f"AVAILABLE FUNCTIONS:\n{schema_str}\n\n"
            f"{self.examples_str}"
            "Do NOT include any conversational text.\n"
            "Do NOT wrap the output in markdown blocks. Generate ONLY the raw JSON object."
        )

        if self.format_type == ModelFormat.CHATML:
            return self._build_chatml(system_instruction, user_prompt)
        elif self.format_type == ModelFormat.INSTRUCT:
            return self._build_instruct(system_instruction, user_prompt)
        else:
            raise ValueError(f"Unsupported template format: {self.format_type}")

    def _build_chatml(self, system_instruction: str, user_prompt: str) -> str:
        return (
            f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _build_instruct(self, system_instruction: str, user_prompt: str) -> str:
        return (
            f"[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )
