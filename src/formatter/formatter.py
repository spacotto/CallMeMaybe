"""
Formats and structures dynamic prompts for the LLM engine.

This module constructs the specific prompt strings required for both
Phase 1 (Classification) and Phase 2 (Extraction) of the constrained
decoding pipeline. It handles template switching (ChatML vs Instruct)
and implements O(1) memory caching for few-shot examples.
"""

import json
import os
from enum import Enum
from typing import List, Dict, Any, cast


class ModelFormat(Enum):
    """
    Enumeration of supported LLM prompt template formats.
    """
    CHATML = "chatml"
    INSTRUCT = "instruct"


class Formatter:
    """
    Constructs phase-specific LLM prompts using optimal memory techniques.

    This class abstracts the complex string formatting required to
    communicate with different LLM architectures. It relies on list
    joining for large strings rather than standard concatenation to
    prevent memory fragmentation during large batch processing.

    Attributes:
        format_type (ModelFormat): The prompt syntax to enforce.
        examples_dir (str): Absolute path to the few-shot JSON folder.
    """
    def __init__(
        self, format_type: ModelFormat = ModelFormat.CHATML
    ) -> None:
        """
        Initializes the formatter and its I/O cache.

        Args:
            format_type (ModelFormat): Template syntax. Defaults to CHATML.
        """
        self.format_type = format_type
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.examples_dir = os.path.join(current_dir, "few_shot")

        # In-memory cache to eliminate redundant Disk I/O
        self._examples_cache: Dict[str, List[Dict[str, Any]]] = {}

    def load_examples(self, func_name: str) -> List[Dict[str, Any]]:
        """
        Dynamically loads few-shot examples with O(1) memory caching.

        If the examples for a specific function have been read from disk
        previously, they are served immediately from the cache to prevent
        redundant I/O bottlenecks during batch runs.

        Args:
            func_name (str): The target function name to load.

        Returns:
            List[Dict[str, Any]]: A list of few-shot example objects.
        """
        if func_name in self._examples_cache:
            return self._examples_cache[func_name]

        filepath = os.path.join(self.examples_dir, f"{func_name}.json")
        examples = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    examples = cast(List[Dict[str, Any]], json.load(f))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse examples for "
                      f"{func_name}: {e}")

        # Cache the result (even if empty, to prevent repeated failed reads)
        self._examples_cache[func_name] = examples
        return examples

    def build_classification_prompt(
        self, user_prompt: str, functions: List[Dict[str, Any]]
    ) -> str:
        """
        Assembles the Phase 1 zero-shot routing prompt.

        This prompt forces the model to evaluate a compressed summary
        of available schemas and output the single correct function name.

        Args:
            user_prompt (str): The raw input from the user.
            functions (List[Dict[str, Any]]): All available schemas.

        Returns:
            str: The fully formatted ChatML or Instruct prompt.
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
        """
        Assembles the Phase 2 few-shot extraction prompt.

        Injects the full JSON schema and available few-shot examples
        into the model's native message boundaries to ground the
        extraction process before logit masking begins.

        Args:
            user_prompt (str): The raw input from the user.
            target_name (str): The function name chosen in Phase 1.
            functions (List[Dict[str, Any]]): All available schemas.
            examples (List[Dict[str, Any]]): Loaded few-shot examples.

        Returns:
            str: The fully formatted ChatML or Instruct prompt.
        """
        target_schema = next(
            (f for f in functions if f["name"] == target_name), {}
        )
        schema_str = json.dumps(target_schema, indent=2)

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
        """
        Assembles a ChatML prompt using highly optimized list joining.

        Args:
            system_instruction (str): The core engine directives.
            user_prompt (str): The natural language query.
            target_name (str): The targeted function name.
            examples (List[Dict[str, Any]]): Injected examples.

        Returns:
            str: A continuous ChatML prompt string.
        """
        parts = [f"<|im_start|>system\n{system_instruction}<|im_end|>\n"]

        for ex in examples:
            parts.append("<|im_start|>user\n"
                         f"{ex.get('prompt', '')}<|im_end|>\n")
            expected_output = {
                "name": target_name,
                "parameters": ex.get('parameters', {})
            }
            out_str = json.dumps(expected_output)
            parts.append(f"<|im_start|>assistant\n{out_str}<|im_end|>\n")

        parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")

        return "".join(parts)

    def _build_instruct_extraction(
        self, system_instruction: str, user_prompt: str,
        target_name: str, examples: List[Dict[str, Any]]
    ) -> str:
        """
        Assembles an Instruct prompt using optimized list joining.

        Args:
            system_instruction (str): The core engine directives.
            user_prompt (str): The natural language query.
            target_name (str): The targeted function name.
            examples (List[Dict[str, Any]]): Injected examples.

        Returns:
            str: A continuous Llama-style Instruct prompt string.
        """
        parts = [f"[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n"]

        if examples:
            ex = examples[0]
            parts.append(f"{ex.get('prompt', '')} [/INST] ")
            expected_output = {
                "name": target_name,
                "parameters": ex.get('parameters', {})
            }
            parts.append(f"{json.dumps(expected_output)} </s>")

            for ex in examples[1:]:
                parts.append(f" <s>[INST] {ex.get('prompt', '')} [/INST] ")
                expected_output = {
                    "name": target_name,
                    "parameters": ex.get('parameters', {})
                }
                parts.append(f"{json.dumps(expected_output)} </s>")

            parts.append(f" <s>[INST] {user_prompt} [/INST]")
        else:
            parts.append(f"{user_prompt} [/INST]")

        return "".join(parts)
