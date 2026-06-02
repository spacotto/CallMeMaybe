import numpy as np
import re
from typing import List, Dict, Any, Set, Tuple

from src.tokenizer import Tokenizer
from src.formatter import Formatter, ModelFormat
from src.utils import Formatter as clr
from llm_sdk import Small_LLM_Model

from src.engine.trie import SchemaTrie

class ConstrainedDecoder:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
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

        # 1. Setup
        primed_prompt = self.formatter.build_function_prompt(user_prompt, functions)
        input_ids = self.tokenizer.encode(primed_prompt)
        generated_ids: List[int] = []

        # Build a single Trie strictly for the function names
        valid_names = [f["name"] for f in functions]
        name_trie = SchemaTrie(valid_names)

        if verbose: print(clr.apply(None, 'gray', "-" * 70))

        # 2. Execution Loop
        for step in range(max_new_tokens):
            current_prefix = self.tokenizer.decode(generated_ids)

            # Determine the mathematical mask
            invalid_ids, allowed_chars = self._get_mask(current_prefix, name_trie)

            # Predict and mask
            current_sequence = input_ids + generated_ids
            raw_logits = self.sdk.get_logits_from_input_ids(current_sequence)

            logits_array = np.array(raw_logits, dtype=np.float32)
            if invalid_ids:
                logits_array[invalid_ids] = -np.inf

            next_token_id = int(np.argmax(logits_array))
            generated_ids.append(next_token_id)

            # Visualization
            if verbose:
                state_name = "Inside String" if self._is_in_string(current_prefix) else "Structural JSON"
                self._print_step(step, next_token_id, allowed_chars, state_name)

            # Termination
            new_prefix = self.tokenizer.decode(generated_ids)
            if self._is_complete(new_prefix, next_token_id):
                if verbose: print(clr.apply(None, 'gray', "\n" + "-" * 70))
                break

        return self.tokenizer.decode(generated_ids)

    # -----------------------------------------------------------------------
    # Core Masking Logic
    # -----------------------------------------------------------------------

    def _get_mask(self, current_prefix: str, name_trie: SchemaTrie) -> Tuple[List[int], Set[str]]:
        """A simplified state machine that targets the root function name and sandboxes syntax."""
        invalid_ids = []
        allowed_chars_viz = set()

        in_string = self._is_in_string(current_prefix)

        if not in_string:
            allowed_chars_viz = set('{}[]:,"0123456789 \n\r\ttruefalsenull')

            # --- FIX 1: Anti-Nesting Trap ---
            # Prevent the model from opening nested objects or arrays inside arguments.
            if '"arguments"' in current_prefix:
                allowed_chars_viz.discard('{')
                allowed_chars_viz.discard('[')
                # Exception: Allow the very first '{' to open the arguments dictionary itself.
                if re.search(r'"arguments"\s*:\s*$', current_prefix):
                    allowed_chars_viz.add('{')

            for t_id, t_str in self.tokenizer.id_to_token.items():
                s = t_str.replace("Ġ", " ")

                # Sandboxing: Block empty tokens and illegal chars
                if not s or not all(c in allowed_chars_viz for c in s):
                    invalid_ids.append(t_id)
                    continue

                # Sandboxing: Force clean quote transitions
                if '"' in s and (s.count('"') > 1 or not s.endswith('"')):
                    invalid_ids.append(t_id)
                    continue

                # Sandboxing: Ban the alphabet exploit
                letters = re.findall(r'[a-z]+', s.lower())
                banned_word = False
                for word in letters:
                    if not any(valid.startswith(word) or word in valid for valid in ['true', 'false', 'null']):
                        banned_word = True
                        break
                if banned_word:
                    invalid_ids.append(t_id)

        else:
            parts = current_prefix.rsplit('"', 1)

            # --- FIX 2: Root-Only Trie Routing ---
            # Apply the Trie ONLY if we are typing the "name" key BEFORE arguments are opened.
            is_root_name = len(parts) > 1 and re.search(r'"name"\s*:\s*$', parts[0]) and '"arguments"' not in parts[0]

            if is_root_name:
                current_name_prefix = parts[1]
                allowed_chars_viz = name_trie.get_allowed_next_chars(current_name_prefix)
                for t_id, t_str in self.tokenizer.id_to_token.items():
                    s = t_str.replace("Ġ", " ")
                    if not s or not name_trie.is_valid_path(current_name_prefix + s):
                        invalid_ids.append(t_id)
            else:
                # Standard string fallback for all arguments
                allowed_chars_viz = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789 -.,!?@"\'/\\')
                for t_id, t_str in self.tokenizer.id_to_token.items():
                    s = t_str.replace("Ġ", " ")
                    if not s or not all(c in allowed_chars_viz for c in s):
                        invalid_ids.append(t_id)
                        continue
                    if '"' in s and (s.count('"') > 1 or not s.endswith('"')):
                        invalid_ids.append(t_id)

        return invalid_ids, allowed_chars_viz

    def _is_in_string(self, text: str) -> bool:
        """Robustly tracks quote depth."""
        in_string, escape = False, False
        for char in text:
            if escape: escape = False; continue
            if char == "\\": escape = True; continue
            if char == '"': in_string = not in_string
        return in_string

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _print_step(self, step: int, next_token_id: int, allowed_chars: Set[str], state_name: str) -> None:
        token_str = self.tokenizer.decode([next_token_id]).replace('\n', '\\n').replace('\r', '\\r')
        line = (
            clr.apply('bold', 'blue', f"[{step+1:03d}] ") +
            clr.apply(None, 'cyan', f"State: ") +
            clr.apply('bold', 'white', f"{state_name:<16}") +
            clr.apply(None, 'gray', " | ") +
            clr.apply(None, 'yellow', f"Mask: ") +
            clr.apply('bold', 'yellow', f"{len(allowed_chars):02d} allowed chars") +
            clr.apply(None, 'gray', " | ") +
            clr.apply(None, 'lime', f"Token Generated: ") +
            clr.apply('bold', 'lime', f"'{token_str}'")
        )
        print(f"\r\033[K{line}", end="", flush=True)

    def _is_complete(self, text: str, last_token_id: int) -> bool:
        if last_token_id == self.tokenizer.token_to_id.get("<|im_end|>", -1):
            return True

        if not text or "{" not in text:
            return False

        json_str = text[text.find("{"):]
        depth, in_string, escape = 0, False, False

        for char in json_str:
            if escape: escape = False; continue
            if char == "\\": escape = True; continue
            if char == '"': in_string = not in_string; continue

            if not in_string:
                if char == "{": depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0: return True
        return False
