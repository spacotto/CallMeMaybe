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
        # Dynamically map the path to the 'few_shot' folder located in the same directory as this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.examples_dir = os.path.join(current_dir, "few_shot")

    def load_examples(self, func_name: str) -> List[Dict[str, Any]]:
        """Dynamically loads few-shot examples for a specific function."""
        filepath = os.path.join(self.examples_dir, f"{func_name}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse examples for {func_name}: {e}")
        return []

    def _format_examples_string(self, target_name: str, examples: List[Dict[str, Any]]) -> str:
        """Compiles targeted JSON examples into a strict string layout."""
        if not examples:
            return ""

        formatted = "--- EXAMPLES ---\n"
        for ex in examples:
            formatted += f"User: {ex.get('prompt', '')}\n"

            # Construct the expected full JSON output so the LLM sees the exact structure
            expected_output = {
                "name": target_name,
                "parameters": ex.get('parameters', {})
            }
            out_str = json.dumps(expected_output, indent=2)

            formatted += f"Output:\n{out_str}\n\n"
        formatted += "----------------\n\n"
        return formatted

    def build_classification_prompt(self, user_prompt: str, functions: List[Dict[str, Any]]) -> str:
        """PASS 1: Zero-Shot prompt to force the LLM to identify the function name."""
        # For classification, we only need the names and descriptions to save context window space
        schema_summary = [{"name": f["name"], "description": f.get("description", "")} for f in functions]
        schema_str = json.dumps(schema_summary, indent=2)

        system_instruction = (
            "You are a routing engine. Your ONLY job is to classify the user's request.\n"
            "You MUST respond with a single JSON object containing ONLY the key 'name'.\n"
            "Map the user's request to the correct function name from the list below.\n\n"
            f"AVAILABLE FUNCTIONS:\n{schema_str}\n\n"
            "Do NOT include any conversational text. Generate ONLY the raw JSON object."
        )

        return self._route_template(system_instruction, user_prompt)

    def build_extraction_prompt(self, user_prompt: str, target_name: str, functions: List[Dict[str, Any]], examples: List[Dict[str, Any]]) -> str:
        """PASS 2: Few-Shot prompt to force the LLM to extract complex nested parameters."""
        # Find the specific schema for the target function
        target_schema = next((f for f in functions if f["name"] == target_name), {})
        schema_str = json.dumps(target_schema, indent=2)

        # Build the exact few-shot strings based on the loaded JSON file
        examples_str = self._format_examples_string(target_name, examples)

        system_instruction = (
            "You are a precise data extraction engine.\n"
            f"The user's request maps to the function: '{target_name}'.\n"
            "You MUST extract the parameters EXACTLY as defined in the schema below.\n"
            "You MUST respond with a single JSON object containing the keys 'name' and 'parameters'.\n\n"
            "CRITICAL RULES FOR REGEX REPLACEMENT:\n"
            r"- If replacing numbers, the 'regex' parameter MUST be '\\d+'." + "\n"
            r"- If replacing vowels, the 'regex' parameter MUST be '[aeiouAEIOU]'." + "\n"
            r"- If replacing exact words (like 'cat'), the 'regex' parameter MUST be '\\bcat\\b'." + "\n\n"
            f"FUNCTION SCHEMA:\n{schema_str}\n\n"
            f"{examples_str}"
            "Do NOT include any conversational text. Generate ONLY the raw JSON object."
        )

        return self._route_template(system_instruction, user_prompt)

    def _route_template(self, system_instruction: str, user_prompt: str) -> str:
        """Applies the correct model template wrapper."""
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
