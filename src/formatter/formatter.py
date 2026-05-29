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
        specific chat template, priming it to output nothing but JSON.
        """
        # Stringify the target JSON schema
        schema_str = json.dumps(functions, indent=2)

        # Build the strict system instruction
        system_instruction = (
            "You are a strict logical engine.\n"
            "You MUST respond with a single, "
            "valid JSON object representing a function call.\n"
            "Do NOT include any conversational text.\n"
            "Do NOT wrap the output in markdown blocks.\n"
            "You have access to the following functions.\n"
            f"Functions:\n{schema_str}"
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
        # Leave the string hanging immediately after the assistant header
        # This forces the next generated token to be the start of the JSON
        return (
            f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _build_instruct(self, system_instruction: str, user_prompt: str) -> str:
        """
        Constructs the standard Instruct format as an architectural proof of
        concept.
        """
        return (
            f"[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )
