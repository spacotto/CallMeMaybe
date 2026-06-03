import numpy as np
import re
import json
from typing import List, Dict, Any, Set, Tuple

from src.tokenizer import Tokenizer
from src.formatter import Formatter, ModelFormat
from llm_sdk import Small_LLM_Model
from src.engine.trie import SchemaTrie
from src.visualizer import Visualizer

class ConstrainedDecoder:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.sdk = Small_LLM_Model(model_name=model_name)
        self.tokenizer = Tokenizer()

        if "qwen" in model_name.lower() or "smol" in model_name.lower():
            self.formatter = Formatter(format_type=ModelFormat.CHATML)
        else:
            self.formatter = Formatter(format_type=ModelFormat.INSTRUCT)

        # 1. Base Vocabulary Setup
        max_id = max(self.tokenizer.id_to_token.keys()) if self.tokenizer.id_to_token else 151643
        self.vocab_size = max_id + 1
        self.clean_vocab = [""] * self.vocab_size

        # 2. PURE VECTORIZATION: Precompute structural and safety masks ONCE
        self.struct_mask = np.zeros(self.vocab_size, dtype=bool)
        self.wildcard_string_mask = np.zeros(self.vocab_size, dtype=bool)
        self.safe_quote_mask = np.ones(self.vocab_size, dtype=bool)
        self.banned_keyword_mask = np.ones(self.vocab_size, dtype=bool)

        struct_chars = set('{}[]:,"0123456789 \n\r\ttruefalsenull')
        wildcard_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789 -.,!?@"\'/\\()[]{}*+^$|')
        word_pattern = re.compile(r'\b[a-z]+\b')

        for t_id, t_str in self.tokenizer.id_to_token.items():
            s = t_str.replace("Ġ", " ")
            self.clean_vocab[t_id] = s

            if not s:
                continue

            t_chars = set(s)

            # Precompute subsets
            if t_chars.issubset(struct_chars):
                self.struct_mask[t_id] = True
            if t_chars.issubset(wildcard_chars):
                self.wildcard_string_mask[t_id] = True

            # Precompute quote safety
            if '"' in s and (s.count('"') > 1 or not s.endswith('"')):
                self.safe_quote_mask[t_id] = False

            # Precompute boolean/null overlap safety
            letters = word_pattern.findall(s.lower())
            for word in letters:
                if not any(valid.startswith(word) or word in valid for valid in ['true', 'false', 'null']):
                    self.banned_keyword_mask[t_id] = False
                    break

        self.param_pattern = re.compile(r'"([^"]+)"\s*:\s*"([^"]*)$')

    def generate_function_call(self, user_prompt: str,
                               functions: List[Dict[str, Any]],
                               max_new_tokens: int = 120,
                               verbose: bool = False) -> str:

        primed_prompt = self.formatter.build_function_prompt(user_prompt, functions)

        escaped_prompt = json.dumps(user_prompt)[1:-1]
        boilerplate_start = f'{{"prompt": "{escaped_prompt}", "name": "'

        input_ids = self.tokenizer.encode(primed_prompt)
        generated_ids: List[int] = self.tokenizer.encode(boilerplate_start)

        valid_names = [f["name"] for f in functions]
        name_trie = SchemaTrie(valid_names)

        prompt_lower = user_prompt.lower()
        prompt_words = set(re.findall(r'\b\w+\b', prompt_lower))
        quoted_phrases = set()
        for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', prompt_lower):
            phrase = match.group(1) if match.group(1) else match.group(2)
            if phrase:
                quoted_phrases.add(phrase)

        for step in range(max_new_tokens):
            # SAFE DECODING: Revert to proper BPE tokenizer decode to preserve semantic whitespace
            current_prefix = self.tokenizer.decode(generated_ids)

            mask, allowed_chars = self._get_mask(
                current_prefix, name_trie, prompt_lower, prompt_words, quoted_phrases
            )

            current_sequence = input_ids + generated_ids
            raw_logits = self.sdk.get_logits_from_input_ids(current_sequence)
            logits_array = np.array(raw_logits, dtype=np.float32)

            if mask.shape[0] < logits_array.shape[0]:
                padded_mask = np.zeros(logits_array.shape[0], dtype=bool)
                padded_mask[:mask.shape[0]] = mask
                mask = padded_mask
            elif mask.shape[0] > logits_array.shape[0]:
                mask = mask[:logits_array.shape[0]]

            logits_array[~mask] = -np.inf

            next_token_id = int(np.argmax(logits_array))
            generated_ids.append(next_token_id)

            if verbose:
                new_token_str = self.clean_vocab[next_token_id]
                state_name = "Inside String" if self._is_in_string(current_prefix) else "Structural JSON"
                Visualizer.print_step(step, new_token_str, allowed_chars, state_name)

            new_prefix = self.tokenizer.decode(generated_ids)
            if new_prefix.endswith('"'):
                parts = new_prefix.rsplit('"', 2)
                if len(parts) >= 3 and re.search(r'"name"\s*:\s*$', parts[-3]) and '"parameters"' not in parts[-3]:
                    ff_tokens = self.tokenizer.encode(', "parameters": {')
                    generated_ids.extend(ff_tokens)
                    new_prefix = self.tokenizer.decode(generated_ids)

            if self._is_complete(new_prefix, next_token_id):
                if verbose:
                    Visualizer.print_step_complete()
                break

        return self.tokenizer.decode(generated_ids)

    def _get_mask(self, current_prefix: str, name_trie: SchemaTrie,
                  prompt_lower: str, prompt_words: Set[str], quoted_phrases: Set[str]) -> Tuple[np.ndarray, Set[str]]:

        mask = np.ones(self.vocab_size, dtype=bool)
        allowed_chars_viz = set()

        in_string = self._is_in_string(current_prefix)

        if not in_string:
            # INSTANT BITWISE FILTERING - No Python loops!
            mask &= self.struct_mask
            mask &= self.safe_quote_mask

            if current_prefix.strip():
                mask &= self.banned_keyword_mask

            allowed_chars_viz = set('{}[]:,"0123456789 \n\r\ttruefalsenull')

        else:
            parts = current_prefix.rsplit('"', 1)
            is_root_name = len(parts) > 1 and re.search(r'"name"\s*:\s*$', parts[0]) and '"parameters"' not in parts[0]

            if is_root_name:
                current_name_prefix = parts[1]
                allowed_chars_viz = name_trie.get_allowed_next_chars(current_name_prefix)
                start_node = name_trie.get_node(current_name_prefix)

                if start_node is None:
                    mask[:] = False
                else:
                    # Trie traversal is the only necessary Python loop left
                    for t_id in range(self.vocab_size):
                        s = self.clean_vocab[t_id]
                        if not s or not name_trie.is_valid_suffix(start_node, s):
                            mask[t_id] = False
            else:
                # INSTANT BITWISE FILTERING for standard string characters
                mask &= self.wildcard_string_mask
                mask &= self.safe_quote_mask
                allowed_chars_viz = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789 -.,!?@"\'/\\()[]{}*+^$|')

                param_match = self.param_pattern.search(current_prefix)
                active_key = param_match.group(1) if param_match else None
                current_val = param_match.group(2) if param_match else ""

                if active_key in ["name", "s", "source_string"]:
                    # Extractive rules require localized looping, but over a much smaller pre-filtered subset
                    valid_indices = np.where(mask)[0]
                    for t_id in valid_indices:
                        s = self.clean_vocab[t_id]
                        s_val = s[:-1] if s.endswith('"') else s

                        if s_val:
                            proposed_val = current_val + s_val
                            if proposed_val.lower() not in prompt_lower:
                                mask[t_id] = False
                                continue

                        if s.endswith('"'):
                            final_val = (current_val + s_val).lower()
                            is_valid_closure = False

                            if final_val:
                                clean_val = final_val.strip("'\"")
                                if active_key in ["name", "s"] and (clean_val in prompt_words or clean_val in quoted_phrases):
                                    is_valid_closure = True
                                elif active_key == "source_string" and (clean_val in quoted_phrases or clean_val == prompt_lower):
                                    is_valid_closure = True

                            if not is_valid_closure:
                                mask[t_id] = False

        return mask, allowed_chars_viz

    def _is_in_string(self, text: str) -> bool:
        in_string, escape = False, False
        for char in text:
            if escape: escape = False; continue
            if char == "\\": escape = True; continue
            if char == '"': in_string = not in_string
        return in_string

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
