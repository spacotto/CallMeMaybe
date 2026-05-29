import math
from typing import List, Dict, Any, Set

from src.tokenizer import Tokenizer
from src.formatter import Formatter, ModelFormat
from llm_sdk import Small_LLM_Model

class ConstrainedDecoder:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """
        Initializes the execution engine by loading the local LLM SDK,
        the custom tokenizer, and the decoupled presentation formatter.
        """
        self.sdk = Small_LLM_Model(model_name=model_name)
        self.tokenizer = Tokenizer()

        # Determine format type automatically based on the model target
        if "qwen" in model_name.lower() or "smol" in model_name.lower():
            self.formatter = Formatter(format_type=ModelFormat.CHATML)
        else:
            self.formatter = Formatter(format_type=ModelFormat.INSTRUCT)

    def generate_function_call(
        self,
        user_prompt: str,
        functions: List[Dict[str, Any]],
        max_new_tokens: int = 120
    ) -> str:
        """
        Executes the autoregressive generation loop, forcing the model
        to emit a valid, schema-compliant JSON structure.
        """
        primed_prompt = self.formatter.build_function_prompt(user_prompt, functions)
        input_ids = self.tokenizer.encode(primed_prompt)

        generated_ids: List[int] = []

        for _ in range(max_new_tokens):
            current_sequence = input_ids + generated_ids
            raw_logits = self.sdk.get_logits_from_input_ids(current_sequence)

            current_prefix = self.tokenizer.decode(generated_ids)
            allowed_chars = self._get_allowed_next_characters(current_prefix)

            masked_logits = self._apply_constraint_mask(raw_logits, allowed_chars)

            # Pure Python argmax equivalent
            next_token_id = max(range(len(masked_logits)), key=masked_logits.__getitem__)

            # Break early if the engine selects an End-of-Sequence marker or structural termination
            if next_token_id == self.tokenizer.token_to_id.get("<|im_end|>", -1) or current_prefix.endswith("}"):
                break

            generated_ids.append(next_token_id)

        return self.tokenizer.decode(generated_ids)

    def _get_allowed_next_characters(self, current_prefix: str) -> Set[str]:
        """
        Evaluates the existing generated string prefix to determine the exact set
        of grammatically legal characters that can follow.
        """
        stripped = current_prefix.strip()

        if not stripped:
            return {"{"}

        if stripped == "{":
            return {'"'}

        if stripped.startswith('{"') and not stripped.endswith('"'):
            return set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_\"")

        if stripped.endswith('"') and ":" not in stripped:
            return {":"}

        if stripped.endswith(":"):
            return {'"', "[", "{", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

        return set('{}[]:,"-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ ')

    def _apply_constraint_mask(self, logits: List[float], allowed_chars: Set[str]) -> List[float]:
        """
        Scans every token in the tokenizer vocabulary. If a token's character
        representation breaks syntax progression, its logit is crushed to -inf.
        """
        masked_logits = list(logits)

        for token_id, token_str in self.id_to_token.items():
            actual_str = token_str.replace("Ġ", " ")

            if actual_str and not any(char in allowed_chars for char in actual_str):
                # Using standard math.inf instead of torch
                masked_logits[token_id] = -math.inf

        return masked_logits

    @property
    def id_to_token(self) -> Dict[int, str]:
        return self.tokenizer.id_to_token
