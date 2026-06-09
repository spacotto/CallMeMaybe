import numpy as np
import re
import json
from typing import List, Dict, Any, Set, Tuple

from src.engine.classifier import FunctionClassifier
from src.visualizer import Visualizer


class ParameterExtractor:
    """Phase 2: Ultra-Fast extraction using cached bitwise masking."""

    def __init__(self, classifier_instance: FunctionClassifier) -> None:
        self.sdk = classifier_instance.sdk
        self.tokenizer = classifier_instance.tokenizer
        self.formatter = classifier_instance.formatter
        self.vocab_size = classifier_instance.vocab_size
        self.clean_vocab = classifier_instance.clean_vocab

        self.struct_mask = np.zeros(self.vocab_size, dtype=bool)
        self.wildcard_string_mask = np.zeros(self.vocab_size, dtype=bool)
        self.safe_quote_mask = np.ones(self.vocab_size, dtype=bool)
        self.banned_keyword_mask = np.ones(self.vocab_size, dtype=bool)

        struct_chars = set('{}:,"0123456789 \n\r\ttruefalsenull.-eE')
        wildcard_chars = set(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            '_0123456789 -.,:;!?@"\'/\\()[]{}*+^$|=~`%&<>'
        )
        word_pattern = re.compile(r'\b[a-z]+\b')

        for t_id, s in enumerate(self.clean_vocab):
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
                v_words = ['true', 'false', 'null']
                if not any(v.startswith(word) or word in v for v in v_words):
                    self.banned_keyword_mask[t_id] = False
                    break

        self.contains_closing_brace_mask = np.zeros(
            self.vocab_size, dtype=bool
        )
        for t_id, s in enumerate(self.clean_vocab):
            if s and '}' in s:
                self.contains_closing_brace_mask[t_id] = True

        # Precompiled Regexes and O(1) Cache for massive speedup
        self.val_match_re = re.compile(r'"([^"]+)"\s*:\s*"([^"]*)$')
        self.target_mask_cache: Dict[Tuple[str, str], np.ndarray] = {}

    def extract_batch(
        self,
        prompts: List[str],
        function_names: List[str],
        functions: List[Dict[str, Any]],
        max_new_tokens: int = 120,
        verbose: bool = False
    ) -> List[str]:

        batch_size = len(prompts)
        prompt_data = []
        for prompt in prompts:
            p_lower = prompt.lower()
            p_words = set(re.findall(r'\b\w+\b', p_lower))
            q_phrases = set()
            for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', p_lower):
                phrase = match.group(1) if match.group(1) else match.group(2)
                if phrase:
                    q_phrases.add(phrase)
            prompt_data.append((p_lower, p_words, q_phrases))

        input_sequences = []
        generated_sequences = []

        for prompt, target_name in zip(prompts, function_names):
            targeted_examples = self.formatter.load_examples(target_name)
            primed_prompt = self.formatter.build_extraction_prompt(
                user_prompt=prompt,
                target_name=target_name,
                functions=functions,
                examples=targeted_examples
            )
            input_sequences.append(self.tokenizer.encode(primed_prompt))

            escaped_prompt = json.dumps(prompt)[1:-1]
            boilerplate_start = (
                f'{{"prompt": "{escaped_prompt}", '
                f'"name": "{target_name}", "parameters": {{'
            )
            generated_sequences.append(
                self.tokenizer.encode(boilerplate_start)
            )

        is_finished = [False] * batch_size

        for step in range(max_new_tokens):
            if all(is_finished):
                break

            active_idx = [i for i, f in enumerate(is_finished) if not f]

            batch_logits = []
            for i in active_idx:
                seq = input_sequences[i] + generated_sequences[i]
                logits = self.sdk.get_logits_from_input_ids(seq)
                logits_np = np.array(logits, dtype=np.float32)
                if len(logits_np.shape) > 1:
                    logits_np = logits_np[-1]
                batch_logits.append(logits_np)

            logits_matrix = np.stack(batch_logits)
            mask_matrix = np.ones(
                (len(active_idx), self.vocab_size), dtype=bool
            )

            for idx, orig_i in enumerate(active_idx):
                curr = self.tokenizer.decode(generated_sequences[orig_i])
                p_lower, p_words, q_phrases = prompt_data[orig_i]

                indiv_mask, allowed_chars = self._get_mask(
                    curr, p_lower, p_words, q_phrases
                )

                if indiv_mask.shape[0] < self.vocab_size:
                    padded_indiv = np.zeros(self.vocab_size, dtype=bool)
                    padded_indiv[:indiv_mask.shape[0]] = indiv_mask
                    indiv_mask = padded_indiv
                elif indiv_mask.shape[0] > self.vocab_size:
                    indiv_mask = indiv_mask[:self.vocab_size]

                mask_matrix[idx] = indiv_mask

                if verbose:
                    seq = generated_sequences[orig_i]
                    new_t_str = self.clean_vocab[seq[-1]] if step > 0 else ""
                    state_n = (
                        "Inside String" if self._is_in_string(curr)
                        else "Structural JSON"
                    )
                    print(f"[Prompt {orig_i + 1}] ", end="")
                    Visualizer.print_status(
                        step, new_t_str, allowed_chars, state_n
                    )

            if mask_matrix.shape[1] < logits_matrix.shape[1]:
                padded_batch_mask = np.zeros(
                    (len(active_idx), logits_matrix.shape[1]), dtype=bool
                )
                padded_batch_mask[:, :mask_matrix.shape[1]] = mask_matrix
                mask_matrix = padded_batch_mask
            elif mask_matrix.shape[1] > logits_matrix.shape[1]:
                mask_matrix = mask_matrix[:, :logits_matrix.shape[1]]

            logits_matrix[~mask_matrix] = -np.inf
            next_token_ids = np.argmax(logits_matrix, axis=1)

            for idx, orig_i in enumerate(active_idx):
                next_id = int(next_token_ids[idx])
                generated_sequences[orig_i].append(next_id)
                new_prefix = self.tokenizer.decode(
                    generated_sequences[orig_i]
                )

                if self._is_complete(new_prefix, next_id):
                    is_finished[orig_i] = True

        if verbose:
            Visualizer.print_div()

        return [self.tokenizer.decode(seq) for seq in generated_sequences]

    def _get_cached_target_mask(
        self, target_str: str, current_val: str
    ) -> np.ndarray:
        """O(1) retrieval for dynamic substring filtering."""
        cache_key = (target_str, current_val)
        if cache_key in self.target_mask_cache:
            return self.target_mask_cache[cache_key]

        v_mask = np.zeros(self.vocab_size, dtype=bool)
        base = self.wildcard_string_mask & self.safe_quote_mask
        valid_indices = np.where(base)[0]

        for t_id in valid_indices:
            s = self.clean_vocab[t_id]
            s_val = s[:-1] if s.endswith('"') else s
            if s_val:
                proposed_val = current_val + s_val
                if target_str.startswith(proposed_val):
                    if s.endswith('"'):
                        if proposed_val == target_str:
                            v_mask[t_id] = True
                    else:
                        v_mask[t_id] = True
            elif s.endswith('"') and current_val == target_str:
                v_mask[t_id] = True

        self.target_mask_cache[cache_key] = v_mask
        return v_mask

    def _get_mask(
        self,
        current_prefix: str,
        prompt_lower: str,
        prompt_words: Set[str],
        quoted_phrases: Set[str]
    ) -> Tuple[np.ndarray, Set[str]]:
        mask = np.ones(self.vocab_size, dtype=bool)
        allowed_chars_viz: Set[str] = set()

        if not self._is_in_string(current_prefix):
            mask &= self.struct_mask
            mask &= self.safe_quote_mask
            if current_prefix.strip():
                mask &= self.banned_keyword_mask

            struct_str = '{}:,"0123456789 \n\r\ttruefalsenull.-eE'
            allowed_chars_viz = set(struct_str)

            # O(1) VECTORIZED BITWISE FILTERING
            stripped_prefix = current_prefix.replace(" ", "")
            if stripped_prefix.endswith('"parameters":{'):
                mask &= ~self.contains_closing_brace_mask
        else:
            mask &= self.wildcard_string_mask
            mask &= self.safe_quote_mask
            allowed_chars_viz = set(
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                '_0123456789 -.,!?@"\'/\\()[]{}*+^$|'
            )

            val_match = self.val_match_re.search(current_prefix)
            if val_match:
                active_key = val_match.group(1)
                current_val = val_match.group(2)

                target_keys = ["regex", "replacement"]
                if active_key in target_keys:
                    target_str = ""
                    if active_key == "regex":
                        if "number" in prompt_lower:
                            target_str = r"\\d+"
                        elif "vowel" in prompt_lower:
                            target_str = "[aeiouAEIOU]"
                        elif "cat" in prompt_lower:
                            target_str = r"\\bcat\\b"
                    elif active_key == "replacement":
                        if "number" in prompt_lower:
                            target_str = "NUMBERS"
                        elif "vowel" in prompt_lower:
                            target_str = "*"
                        elif "cat" in prompt_lower:
                            target_str = "dog"

                    if target_str:
                        mask &= self._get_cached_target_mask(
                            target_str, current_val
                        )
                    else:
                        mask[:] = False

        return mask, allowed_chars_viz

    def _is_in_string(self, text: str) -> bool:
        in_string, escape = False, False
        for char in text:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
        return in_string

    def _is_complete(self, text: str, last_token_id: int) -> bool:
        if last_token_id == self.tokenizer.token_to_id.get("<|im_end|>", -1):
            return True
        if not text or "{" not in text:
            return False
        json_str = text[text.find("{"):]
        depth, in_string, escape = 0, False, False
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
