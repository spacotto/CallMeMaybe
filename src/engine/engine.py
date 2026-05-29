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

    def generate_function_call(self, user_prompt: str,
                               functions: List[Dict[str, Any]],
                               max_new_tokens: int = 120) -> str:
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

            next_token_id = max(range(len(masked_logits)), key=masked_logits.__getitem__)
            generated_ids.append(next_token_id)

            new_prefix = self.tokenizer.decode(generated_ids)

            if self._is_complete_json(new_prefix) or next_token_id == self.tokenizer.token_to_id.get("<|im_end|>", -1):
                break

        return self.tokenizer.decode(generated_ids)

    def _is_complete_json(self, text: str) -> bool:
        """
        Tracks bracket depth to definitively trigger loop termination
        the millisecond the root JSON object closes.
        """
        if not text or "{" not in text:
            return False

        json_str = text[text.find("{"):]
        depth = 0
        in_string = False
        escape = False

        for char in json_str:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return True
        return False

    def _get_allowed_next_characters(self, current_prefix: str) -> Set[str]:
        """
        Evaluates the generated prefix using a strict JSON state machine.
        Toggles allowed character sets based on whether the generation cursor
        is currently inside or outside a string literal.
        """
        if not current_prefix.strip():
            return {"{"}

        in_string = False
        escape = False
        for char in current_prefix:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string

        if in_string:
            return set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789 -.,!?@"\'/\\')
        else:
            return set('{}[]:,"0123456789 \n\r\ttruefalsenull')

    def _apply_constraint_mask(self, logits: List[float], allowed_chars: Set[str]) -> List[float]:
        """
        Scans every token in the tokenizer vocabulary. If a token's character
        representation breaks syntax progression, its logit is crushed to -inf.
        """
        masked_logits = list(logits)

        for token_id, token_str in self.id_to_token.items():
            actual_str = token_str.replace("Ġ", " ")

            if actual_str and not all(char in allowed_chars for char in actual_str):
                masked_logits[token_id] = -math.inf

        return masked_logits

    @property
    def id_to_token(self) -> Dict[int, str]:
        return self.tokenizer.id_to_token
