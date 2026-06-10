import json
import os
from enum import Enum
from typing import List, Dict, Any, cast


class ModelFormat(Enum):
    CHATML = "chatml"
    INSTRUCT = "instruct"


class Formatter:
    def __init__(
        self, format_type: ModelFormat = ModelFormat.CHATML
    ) -> None:
        self.format_type = format_type
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.examples_dir = os.path.join(current_dir, "few_shot")

    def load_examples(self, func_name: str) -> List[Dict[str, Any]]:
        """Dynamically loads few-shot examples for a specific function."""
        filepath = os.path.join(self.examples_dir, f"{func_name}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return cast(List[Dict[str, Any]], json.load(f))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse examples for "
                      f"{func_name}: {e}")
        return []

    def build_classification_prompt(
        self, user_prompt: str, functions: List[Dict[str, Any]]
    ) -> str:
        """PASS 1: Zero-Shot prompt to force the LLM to identify the
        function name.
        """
        schema_summary = [
            {
                "name": f["name"],
                "description": f.get("description", "")
            }
            for f in functions
        ]
        schema_str = json.dumps(schema_summary, indent=2)

        system_instruction = (
            "You are a routing engine. Your ONLY job is to classify "
            "the user's request.\n"
            "You MUST respond with a single JSON object containing "
            "ONLY the key 'name'.\n"
            "Map the user's request to the correct function name "
            "from the list below.\n\n"
            f"AVAILABLE FUNCTIONS:\n{schema_str}\n\n"
            "Do NOT include any conversational text. "
            "Generate ONLY the raw JSON object."
        )

        # Classification keeps a simple system/user structure
        if self.format_type == ModelFormat.CHATML:
            return (
                f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            return (
                f"[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n"
                f"{user_prompt} [/INST]"
            )

    def build_extraction_prompt(
        self,
        user_prompt: str,
        target_name: str,
        functions: List[Dict[str, Any]],
        examples: List[Dict[str, Any]]
    ) -> str:
        """PASS 2: Few-Shot prompt utilizing native message boundaries
        to ground the model and safely extract parameters.
        """
        target_schema = next(
            (f for f in functions if f["name"] == target_name), {}
        )
        schema_str = json.dumps(target_schema, indent=2)

        # 🛠️ FIX 1: Hardcoded text heuristics stripped out completely.
        system_instruction = (
            "You are a precise data extraction engine.\n"
            f"The user's request maps to the function: '{target_name}'.\n"
            "You MUST extract the parameters EXACTLY as defined in "
            "the schema below.\n"
            "You MUST respond with a single JSON object containing "
            "the keys 'name' and 'parameters'.\n\n"
            f"FUNCTION SCHEMA:\n{schema_str}\n\n"
            "Do NOT include any conversational text. "
            "Generate ONLY the raw JSON object."
        )

        # 🛠️ FIX 2: Structure examples as real native message cycles
        if self.format_type == ModelFormat.CHATML:
            return self._build_chatml_extraction(
                system_instruction, user_prompt, target_name, examples
            )
        elif self.format_type == ModelFormat.INSTRUCT:
            return self._build_instruct_extraction(
                system_instruction, user_prompt, target_name, examples
            )
        else:
            raise ValueError(
                f"Unsupported template format: {self.format_type}"
            )

    def _build_chatml_extraction(
        self, system_instruction: str, user_prompt: str,
        target_name: str, examples: List[Dict[str, Any]]
    ) -> str:
        """Assembles a ChatML prompt stream with explicit dialogue turns."""
        prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n"

        for ex in examples:
            prompt += f"<|im_start|>user\n{ex.get('prompt', '')}<|im_end|>\n"
            expected_output = {
                "name": target_name,
                "parameters": ex.get('parameters', {})
            }
            out_str = json.dumps(expected_output)
            prompt += f"<|im_start|>assistant\n{out_str}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        return prompt

    def _build_instruct_extraction(
        self, system_instruction: str, user_prompt: str,
        target_name: str, examples: List[Dict[str, Any]]
    ) -> str:
        """Assembles an Instruct prompt stream with explicit turn switches."""
        prompt = f"[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n"

        if examples:
            # First example closes the initial system context tag block
            ex = examples[0]
            prompt += f"{ex.get('prompt', '')} [/INST] "
            expected_output = {
                "name": target_name,
                "parameters": ex.get('parameters', {})
            }
            prompt += f"{json.dumps(expected_output)} </s>"

            # Chain subsequent sample inputs cleanly
            for ex in examples[1:]:
                prompt += f" <s>[INST] {ex.get('prompt', '')} [/INST] "
                expected_output = {
                    "name": target_name,
                    "parameters": ex.get('parameters', {})
                }
                prompt += f"{json.dumps(expected_output)} </s>"

            prompt += f" <s>[INST] {user_prompt} [/INST]"
        else:
            prompt += f"{user_prompt} [/INST]"

        return prompt
