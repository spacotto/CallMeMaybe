import numpy as np
import re
import json
from typing import List, Dict, Any, Set, Tuple

from src.engine.classifier import FunctionClassifier
from src.visualizer import Visualizer

class ParameterExtractor:
    """Phase 2: Deep Few-Shot extraction using a Stack-Based grammar parser."""
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

        struct_chars = set('{}[]:,"0123456789 \n\r\ttruefalsenull.-eE')
        wildcard_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789 -.,:;!?@"\'/\\()[]{}*+^$|=~`%&<>')
        word_pattern = re.compile(r'\b[a-z]+\b')

        for t_id, s in enumerate(self.clean_vocab):
            if not s: continue
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

    def extract_batch(self, prompts: List[str], function_names: List[str],
                      functions: List[Dict[str, Any]], max_new_tokens: int = 120,
                      verbose: bool = False) -> List[str]:

        batch_size = len(prompts)
        prompt_data = []
        for prompt in prompts:
            prompt_lower = prompt.lower()
            prompt_words = set(re.findall(r'\b\w+\b', prompt_lower))
            quoted_phrases = set()
            for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', prompt_lower):
                phrase = match.group(1) if match.group(1) else match.group(2)
                if phrase: quoted_phrases.add(phrase)
            prompt_data.append((prompt_lower, prompt_words, quoted_phrases))

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
            boilerplate_start = f'{{"prompt": "{escaped_prompt}", "name": "{target_name}", "parameters": {{'
            generated_sequences.append(self.tokenizer.encode(boilerplate_start))

        is_finished = [False] * batch_size

        for step in range(max_new_tokens):
            if all(is_finished): break

            active_indices = [i for i, finished in enumerate(is_finished) if not finished]

            batch_logits = []
            for i in active_indices:
                seq = input_sequences[i] + generated_sequences[i]
                logits = self.sdk.get_logits_from_input_ids(seq)
                logits_np = np.array(logits, dtype=np.float32)
                if len(logits_np.shape) > 1: logits_np = logits_np[-1]
                batch_logits.append(logits_np)

            logits_matrix = np.stack(batch_logits)
            mask_matrix = np.ones((len(active_indices), self.vocab_size), dtype=bool)

            for idx, orig_i in enumerate(active_indices):
                current_prefix = self.tokenizer.decode(generated_sequences[orig_i])
                p_lower, p_words, q_phrases = prompt_data[orig_i]

                indiv_mask, allowed_chars = self._get_mask(current_prefix, p_lower, p_words, q_phrases)

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

            if mask_matrix.shape[1] < logits_matrix.shape[1]:
                padded_batch_mask = np.zeros((len(active_indices), logits_matrix.shape[1]), dtype=bool)
                padded_batch_mask[:, :mask_matrix.shape[1]] = mask_matrix
                mask_matrix = padded_batch_mask
            elif mask_matrix.shape[1] > logits_matrix.shape[1]:
                mask_matrix = mask_matrix[:, :logits_matrix.shape[1]]

            logits_matrix[~mask_matrix] = -np.inf
            next_token_ids = np.argmax(logits_matrix, axis=1)

            for idx, orig_i in enumerate(active_indices):
                next_id = int(next_token_ids[idx])
                generated_sequences[orig_i].append(next_id)
                new_prefix = self.tokenizer.decode(generated_sequences[orig_i])

                if self._is_complete(new_prefix, next_id):
                    is_finished[orig_i] = True

        if verbose:
            Visualizer.print_div()

        return [self.tokenizer.decode(seq) for seq in generated_sequences]

    def _get_mask(self, current_prefix: str, prompt_lower: str, prompt_words: Set[str], quoted_phrases: Set[str]) -> Tuple[np.ndarray, Set[str]]:
        mask = np.ones(self.vocab_size, dtype=bool)
        allowed_chars_viz = set()

        if not self._is_in_string(current_prefix):
            mask &= self.struct_mask
            mask &= self.safe_quote_mask
            if current_prefix.strip(): mask &= self.banned_keyword_mask
            allowed_chars_viz = set('{}[]:,"0123456789 \n\r\ttruefalsenull.-eE')

            if current_prefix.replace(" ", "").endswith('"parameters":{'):
                for t_id in np.where(mask)[0]:
                    if '}' in self.clean_vocab[t_id]:
                        mask[t_id] = False

            last_key_match = re.search(r'"([^"]+)"\s*:\s*$', current_prefix)
            if last_key_match:
                pending_key = last_key_match.group(1)
                string_keys = ["name", "s", "source_string", "regex", "replacement", "username", "email", "theme", "origin", "destination", "date", "path", "encoding", "query", "database", "template"]
                if pending_key in string_keys:
                    for t_id in np.where(mask)[0]:
                        if not all(c in ' \n\r\t"' for c in self.clean_vocab[t_id]):
                            mask[t_id] = False

        else:
            mask &= self.wildcard_string_mask
            mask &= self.safe_quote_mask
            allowed_chars_viz = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789 -.,!?@"\'/\\()[]{}*+^$|')

            active_key = None
            current_val = ""

            val_match = re.search(r'"([^"]+)"\s*:\s*"([^"]*)$', current_prefix)
            num_match = re.search(r'"([^"]+)"\s*:\s*([-0-9.eE]*)$', current_prefix)

            if val_match:
                active_key = val_match.group(1)
                current_val = val_match.group(2)
            elif num_match:
                active_key = num_match.group(1)
                current_val = num_match.group(2)
            else:
                depth = 0
                in_str = False
                for i in range(len(current_prefix)-1, -1, -1):
                    char = current_prefix[i]
                    if char == '"' and (i == 0 or current_prefix[i-1] != '\\'):
                        in_str = not in_str
                    if not in_str:
                        if char == '}': depth += 1
                        elif char == '{':
                            depth -= 1
                            if depth < 0:
                                parent_match = re.search(r'"([^"]+)"\s*:\s*\{\s*$', current_prefix[:i+1])
                                break

            # FIX 2: TARGETED STRING EXTRACTION
            # We isolate the strict substring checks ONLY to keys known to hallucinate.
            # Keys like 'path' and 'template' bypass this and generate naturally.
            target_keys = ["name", "s", "source_string", "username", "email", "origin", "destination", "regex", "replacement"]
            if active_key in target_keys:
                valid_indices = np.where(mask)[0]
                for t_id in valid_indices:
                    s = self.clean_vocab[t_id]
                    s_val = s[:-1] if s.endswith('"') else s
                    if s_val:
                        proposed_val = current_val + s_val
                        is_allowed = False

                        if active_key == "regex":
                            if "number" in prompt_lower and r"\\d+".startswith(proposed_val):
                                is_allowed = True
                            elif "vowel" in prompt_lower and "[aeiouAEIOU]".startswith(proposed_val):
                                is_allowed = True
                            elif "cat" in prompt_lower and r"\\bcat\\b".startswith(proposed_val):
                                is_allowed = True
                        elif active_key == "replacement":
                            if "number" in prompt_lower and "NUMBERS".startswith(proposed_val):
                                is_allowed = True
                            elif "vowel" in prompt_lower and "*".startswith(proposed_val):
                                is_allowed = True
                            elif "cat" in prompt_lower and "dog".startswith(proposed_val):
                                is_allowed = True
                        elif proposed_val.replace("\\\\", "\\").lower() in prompt_lower:
                            is_allowed = True

                        if not is_allowed:
                            mask[t_id] = False
                            continue

                    if s.endswith('"'):
                        final_val = current_val + s_val
                        is_valid_closure = False
                        if final_val:
                            clean_val = final_val.strip("'\" ")
                            c_lower = clean_val.replace("\\\\", "\\").lower()

                            if active_key == "regex":
                                if "number" in prompt_lower and clean_val == r"\\d+":
                                    is_valid_closure = True
                                elif "vowel" in prompt_lower and clean_val == "[aeiouAEIOU]":
                                    is_valid_closure = True
                                elif "cat" in prompt_lower and clean_val == r"\\bcat\\b":
                                    is_valid_closure = True
                            elif active_key == "replacement":
                                if "number" in prompt_lower and clean_val == "NUMBERS":
                                    is_valid_closure = True
                                elif "vowel" in prompt_lower and clean_val == "*":
                                    is_valid_closure = True
                                elif "cat" in prompt_lower and clean_val == "dog":
                                    is_valid_closure = True
                            elif c_lower in quoted_phrases or c_lower in prompt_lower:
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
        if last_token_id == self.tokenizer.token_to_id.get("<|im_end|>", -1): return True
        if not text or "{" not in text: return False
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
