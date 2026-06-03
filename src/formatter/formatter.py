import json
from enum import Enum
from typing import List, Dict, Any


class ModelFormat(Enum):
    """
    chatml      Used by Qwen, SmolLM, etc.
    instruct    Used by Mistral, Llama 2, etc.
    """
    CHATML = "chatml"
    INSTRUCT = "instruct"


class Formatter:
    def __init__(self, format_type: ModelFormat = ModelFormat.CHATML) -> None:
        """
        Initializes the Formatter to decouple prompt engineering
        from the decoding engine.
        """
        self.format_type = format_type

    def build_function_prompt(self, user_prompt: str, functions: List[Dict[str, Any]]) -> str:
        """
        Wraps the user's prompt and the target JSON schema in the model's
        specific chat template, utilizing Few-Shot Prompting to enforce context.
        """
        # Stringify the target JSON schema
        schema_str = json.dumps(functions, indent=2)

        # Few-Shot System Instruction - ONLY ask for name and parameters!
        system_instruction = (
            "You are a logical data extraction engine.\n"
            "You MUST respond with a single, valid JSON object containing ONLY the keys 'name' and 'parameters'.\n"
            "You MUST Map the user's request to the correct function and extract the parameters EXACTLY as defined in the schema.\n\n"
            f"AVAILABLE FUNCTIONS:\n{schema_str}\n\n"
            "--- EXAMPLES ---\n"
            "User: What is the sum of 10 and 20?\n"
            'Output: {"name": "fn_add_numbers", "parameters": {"a": 10.0, "b": 20.0}}\n\n'
            "User: Replace all numbers in 'test 123' with NUM.\n"
            'Output: {"name": "fn_substitute_string_with_regex", "parameters": {"source_string": "test 123", "regex": "[0-9]+", "replacement": "NUM"}}\n\n'
            "User: Greet shrek\n"
            'Output: {"name": "fn_greet", "parameters": {"name": "shrek"}}\n'
            "----------------\n\n"
            "Do NOT include any conversational text.\n"
            "Do NOT wrap the output in markdown blocks. Generate ONLY the raw JSON object."
        )

        # Apply the specific model's template
        if self.format_type == ModelFormat.CHATML:
            return self._build_chatml(system_instruction, user_prompt)
        elif self.format_type == ModelFormat.INSTRUCT:
            return self._build_instruct(system_instruction, user_prompt)
        else:
            raise ValueError(f"Unsupported template format: {self.format_type}")

    def _build_chatml(self, system_instruction: str, user_prompt: str) -> str:
        """
        Constructs the Qwen/SmolLM ChatML format.
        """
        return (
            f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _build_instruct(self, system_instruction: str, user_prompt: str) -> str:
        """
        Constructs the standard Instruct format.
        """
        return (
            f"[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )
