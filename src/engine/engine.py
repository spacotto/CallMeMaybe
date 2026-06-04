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

        # Base Vocabulary Setup
        max_id = max(self.tokenizer.id_to_token.keys()) if self.tokenizer.id_to_token else 151643
        self.vocab_size = max_id + 1
        self.clean_vocab = [""] * self.vocab_size

        # PURE VECTORIZATION: Precompute structural and safety masks ONCE
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

            if t_chars.issubset(struct_chars):
                self.struct_mask[t_id] = True
            if t_chars.issubset(wildcard_chars):
                self.wildcard_string_mask[t_id] = True

            if '"' in s and (s.count('"') > 1 or not s.endswith('"')):
                self.safe_quote_mask[t_id] = False

            letters = word_pattern.findall(s.lower())
            for word in letters:
                if not any(valid.startswith(word) or word in valid for valid in ['true', 'false', 'null']):
                    self.banned_keyword_mask[t_id] = False
                    break

        self.param_pattern = re.compile(r'"([^"]+)"\s*:\s*"([^"]*)$')

    def generate_batch(self, prompts: List[str],
                       functions: List[Dict[str, Any]],
                       max_new_tokens: int = 120,
                       verbose: bool = False) -> List[str]:
        """
        Processes multiple prompts simultaneously using Active-State Batching.
        Safely bypasses SDK 2D limitations while retaining Numpy matrix acceleration.
        """
        batch_size = len(prompts)
        valid_names = [f["name"] for f in functions]
        name_trie = SchemaTrie(valid_names)

        # 1. Precompute prompt-specific regex dependencies
        prompt_data = []
        for prompt in prompts:
            prompt_lower = prompt.lower()
            prompt_words = set(re.findall(r'\b\w+\b', prompt_lower))
            quoted_phrases = set()
            for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', prompt_lower):
                phrase = match.group(1) if match.group(1) else match.group(2)
                if phrase:
                    quoted_phrases.add(phrase)
            prompt_data.append((prompt_lower, prompt_words, quoted_phrases))

        # 2. Setup Initial Sequences & Prefilling
        input_sequences = []
        generated_sequences = []

        for prompt in prompts:
            primed_prompt = self.formatter.build_function_prompt(prompt, functions)
            input_sequences.append(self.tokenizer.encode(primed_prompt))

            escaped_prompt = json.dumps(prompt)[1:-1]
            boilerplate_start = f'{{"prompt": "{escaped_prompt}", "name": "'
            generated_sequences.append(self.tokenizer.encode(boilerplate_start))

        is_finished = [False] * batch_size

        # Execution Loop
        for step in range(max_new_tokens):
            if all(is_finished):
                break

            # 3. Isolate active prompts to save forward-pass compute
            active_indices = [i for i, finished in enumerate(is_finished) if not finished]

            # 4. Neural Network Forward Pass (Safe 1D Execution)
            batch_logits = []
            for i in active_indices:
                seq = input_sequences[i] + generated_sequences[i]
                logits = self.sdk.get_logits_from_input_ids(seq)

                # Standardize shape to 1D (vocab_size) if SDK returns full history
                logits_np = np.array(logits, dtype=np.float32)
                if len(logits_np.shape) > 1:
                    logits_np = logits_np[-1]
                batch_logits.append(logits_np)

            # Stack into a 2D matrix [active_batch_size, vocab_size]
            logits_matrix = np.stack(batch_logits)

            # 5. Build 2D Boolean Mask Matrix for active prompts
            mask_matrix = np.ones((len(active_indices), self.vocab_size), dtype=bool)

            for idx, orig_i in enumerate(active_indices):
                current_prefix = self.tokenizer.decode(generated_sequences[orig_i])
                p_lower, p_words, q_phrases = prompt_data[orig_i]

                indiv_mask, allowed_chars = self._get_mask(
                    current_prefix, name_trie, p_lower, p_words, q_phrases
                )

                # Align mask shape if tokenizer size differs slightly
                if indiv_mask.shape[0] < self.vocab_size:
                    padded_indiv = np.zeros(self.vocab_size, dtype=bool)
                    padded_indiv[:indiv_mask.shape[0]] = indiv_mask
                    indiv_mask = padded_indiv
                elif indiv_mask.shape[0] > self.vocab_size:
                    indiv_mask = indiv_mask[:self.vocab_size]

                mask_matrix[idx] = indiv_mask

                if verbose:
                    new_token_str = self.clean_vocab[generated_sequences[orig_i][-1]] if step > 0 else ""
                    state_name = "Inside String" if self._is_in_string(current_prefix) else "Structural JSON"
                    print(f"[Prompt {orig_i + 1}] ", end="")
                    Visualizer.print_status(step, new_token_str, allowed_chars, state_name)

            # Align the 2D mask matrix with padded hardware model logits
            if mask_matrix.shape[1] < logits_matrix.shape[1]:
                padded_batch_mask = np.zeros((len(active_indices), logits_matrix.shape[1]), dtype=bool)
                padded_batch_mask[:, :mask_matrix.shape[1]] = mask_matrix
                mask_matrix = padded_batch_mask
            elif mask_matrix.shape[1] > logits_matrix.shape[1]:
                mask_matrix = mask_matrix[:, :logits_matrix.shape[1]]

            # 6. Apply 2D Masking and Argmax extraction simultaneously
            logits_matrix[~mask_matrix] = -np.inf
            next_token_ids = np.argmax(logits_matrix, axis=1)

            # 7. State Update and Deterministic Fast-Forwarding
            for idx, orig_i in enumerate(active_indices):
                next_id = int(next_token_ids[idx])
                generated_sequences[orig_i].append(next_id)

                new_prefix = self.tokenizer.decode(generated_sequences[orig_i])

                if new_prefix.endswith('"'):
                    parts = new_prefix.rsplit('"', 2)
                    if len(parts) >= 3 and re.search(r'"name"\s*:\s*$', parts[-3]) and '"parameters"' not in parts[-3]:
                        ff_tokens = self.tokenizer.encode(', "parameters": {')
                        generated_sequences[orig_i].extend(ff_tokens)
                        new_prefix = self.tokenizer.decode(generated_sequences[orig_i])

                if self._is_complete(new_prefix, next_id):
                    is_finished[orig_i] = True

        if verbose:
            Visualizer.print_div()

        return [self.tokenizer.decode(seq) for seq in generated_sequences]

    def _get_mask(self, current_prefix: str, name_trie: SchemaTrie,
                  prompt_lower: str, prompt_words: Set[str], quoted_phrases: Set[str]) -> Tuple[np.ndarray, Set[str]]:

        mask = np.ones(self.vocab_size, dtype=bool)
        allowed_chars_viz = set()

        in_string = self._is_in_string(current_prefix)

        if not in_string:
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
                    for t_id in range(self.vocab_size):
                        s = self.clean_vocab[t_id]
                        if not s or not name_trie.is_valid_suffix(start_node, s):
                            mask[t_id] = False
            else:
                mask &= self.wildcard_string_mask
                mask &= self.safe_quote_mask
                allowed_chars_viz = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789 -.,!?@"\'/\\()[]{}*+^$|')

                param_match = self.param_pattern.search(current_prefix)
                active_key = param_match.group(1) if param_match else None
                current_val = param_match.group(2) if param_match else ""

                if active_key in ["name", "s", "source_string"]:
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
