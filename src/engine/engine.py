import numpy as np
from typing import List, Dict, Any, Set

from src.tokenizer import Tokenizer
from src.formatter import Formatter, ModelFormat
from src.utils import Formatter as clr
from llm_sdk import Small_LLM_Model


class ConstrainedDecoder:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """
        Initializes the execution engine by loading the local LLM SDK,
        the custom tokenizer, and the decoupled presentation formatter.
        """
        self.sdk = Small_LLM_Model(model_name=model_name)
        self.tokenizer = Tokenizer()

        if "qwen" in model_name.lower() or "smol" in model_name.lower():
            self.formatter = Formatter(format_type=ModelFormat.CHATML)
        else:
            self.formatter = Formatter(format_type=ModelFormat.INSTRUCT)

    def generate_function_call(self, user_prompt: str,
                               functions: List[Dict[str, Any]],
                               max_new_tokens: int = 120,
                               verbose: bool = False) -> str:
        """
        Executes the autoregressive generation loop.
        If verbose is True, prints a real-time visualization of the state machine.
        """
        primed_prompt = self.formatter.build_function_prompt(user_prompt, functions)
        input_ids = self.tokenizer.encode(primed_prompt)

        generated_ids: List[int] = []

        if verbose:
            print(clr.apply(None, 'gray', "-" * 70))

        for step in range(max_new_tokens):
            current_sequence = input_ids + generated_ids
            raw_logits = self.sdk.get_logits_from_input_ids(current_sequence)

            current_prefix = self.tokenizer.decode(generated_ids)

            # --- 1. State Tracking ---
            in_string = self._is_in_string(current_prefix) # Assuming you abstracted this check
            allowed_chars = self._get_allowed_next_characters(current_prefix)

            # --- 2. Mask & Argmax ---
            masked_logits = self._apply_constraint_mask(raw_logits, allowed_chars)
            next_token_id = int(np.argmax(masked_logits))
            generated_ids.append(next_token_id)

            # --- 3. The Mindful Print (Bonus Visualization) ---
            if verbose:
                state_str = "Inside String" if in_string else "Structural JSON"
                # Safely escape newlines so they don't break the terminal layout
                token_str = self.tokenizer.decode([next_token_id]).replace('\n', '\\n').replace('\r', '\\r')

                # Build the formatted string
                line = (
                    clr.apply('bold', 'blue', f"[{step+1:03d}] ") +
                    clr.apply(None, 'cyan', f"State: ") + clr.apply('bold', 'white', f"{state_str:<16}") +
                    clr.apply(None, 'gray', " | ") +
                    clr.apply(None, 'yellow', f"Mask: ") + clr.apply('bold', 'yellow', f"{len(allowed_chars):02d} allowed chars") +
                    clr.apply(None, 'gray', " | ") +
                    clr.apply(None, 'lime', f"Token Generated: ") + clr.apply('bold', 'lime', f"'{token_str}'")
                )

                # \r brings the cursor to the start of the line.
                # \033[K clears everything from the cursor to the end of the line.
                # end="" prevents Python from automatically moving to the next line.
                print(f"\r\033[K{line}", end="", flush=True)

            new_prefix = self.tokenizer.decode(generated_ids)

            if self._is_complete_json(new_prefix) or next_token_id == self.tokenizer.token_to_id.get("<|im_end|>", -1):
                if verbose:
                    print(clr.apply(None, 'gray', "\n" + "-" * 70))
                break

        return self.tokenizer.decode(generated_ids)

    # (Optional) Extract the in_string logic into a tiny helper for cleaner code
    def _is_in_string(self, text: str) -> bool:
        in_string = False
        escape = False
        for char in text:
            if escape:
                escape = False; continue
            if char == "\\":
                escape = True; continue
            if char == '"':
                in_string = not in_string
        return in_string

    def _is_complete_json(self, text: str) -> bool:
        """Tracks bracket depth to definitively trigger loop termination."""
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
        """Toggles allowed character sets based on string literal state."""
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

    def _apply_constraint_mask(self, logits: List[float], allowed_chars: Set[str]) -> np.ndarray:
        """
        Converts logits to a NumPy array and applies -np.inf to illegal tokens.
        """
        # Convert to a fast NumPy array
        logits_array = np.array(logits, dtype=np.float32)

        # Collect indices of tokens that violate the allowed characters
        invalid_indices = []
        for token_id, token_str in self.id_to_token.items():
            actual_str = token_str.replace("Ġ", " ")

            if actual_str and not all(char in allowed_chars for char in actual_str):
                invalid_indices.append(token_id)

        # Vectorized masking: Crush all invalid token probabilities instantly
        if invalid_indices:
            logits_array[invalid_indices] = -np.inf

        return logits_array

    @property
    def id_to_token(self) -> Dict[int, str]:
        return self.tokenizer.id_to_token
