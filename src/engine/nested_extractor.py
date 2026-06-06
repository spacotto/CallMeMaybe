import numpy as np
import re
import json
from typing import List, Dict, Any, Tuple

from src.engine.classifier import FunctionClassifier
from src.visualizer import Visualizer

class NestedExtractor:
    """Phase 2 (Slow Path): Context-Aware CFG parser with key-history tracking."""
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
            if t_chars.issubset(struct_chars): self.struct_mask[t_id] = True
            if t_chars.issubset(wildcard_chars): self.wildcard_string_mask[t_id] = True
            if '"' in s and (s.count('"') > 1 or not s.endswith('"')): self.safe_quote_mask[t_id] = False
            letters = word_pattern.findall(s.lower())
            for word in letters:
                if not any(valid.startswith(word) or word in valid for valid in ['true', 'false', 'null']):
                    self.banned_keyword_mask[t_id] = False
                    break

        self.type_masks = {
            "object": np.zeros(self.vocab_size, dtype=bool),
            "string": np.zeros(self.vocab_size, dtype=bool),
            "number": np.zeros(self.vocab_size, dtype=bool),
            "boolean": np.zeros(self.vocab_size, dtype=bool)
        }
        self.contains_closing_brace_mask = np.zeros(self.vocab_size, dtype=bool)
        for t_id, s in enumerate(self.clean_vocab):
            if not s: continue
            if all(c in ' \n\r\t{' for c in s): self.type_masks["object"][t_id] = True
            if all(c in ' \n\r\t"' for c in s): self.type_masks["string"][t_id] = True
            if all(c in ' \n\r\t0123456789-.' for c in s): self.type_masks["number"][t_id] = True
            if all(c in ' \n\r\ttrufalse' for c in s): self.type_masks["boolean"][t_id] = True
            if '}' in s: self.contains_closing_brace_mask[t_id] = True

        self.key_mask_cache = {}

    def _build_schema_maps(self, properties: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        key_map = {"parameters": list(properties.keys())}
        type_map = {}
        def traverse(k, v):
            t = v.get("type", "string")
            type_map[k] = t
            if t == "object" and "properties" in v:
                key_map[k] = list(v["properties"].keys())
                for sub_k, sub_v in v["properties"].items():
                    traverse(sub_k, sub_v)
        for k, v in properties.items():
            traverse(k, v)
        return key_map, type_map

    def _get_json_state(self, current_prefix: str) -> Tuple[str, List[Dict[str, Any]]]:
        match = re.search(r'"parameters"\s*:\s*\{', current_prefix)
        if not match: return "UNKNOWN", [{"key": "parameters", "completed": set()}]

        json_part = current_prefix[match.end():]
        state = "EXPECT_KEY"
        escape = False
        stack = [{"key": "parameters", "completed": set()}]
        current_key = ""

        for char in json_part:
            if escape: escape = False; continue
            if char == "\\": escape = True; continue

            if state == "EXPECT_KEY":
                if char == '"': state = "IN_KEY"; current_key = ""
                elif char in '}]':
                    state = "EXPECT_COMMA"
                    if char == '}' and len(stack) > 1:
                        completed_obj_key = stack.pop()["key"]
                        stack[-1]["completed"].add(completed_obj_key)
            elif state == "IN_KEY":
                if char == '"': state = "EXPECT_COLON"
                else: current_key += char
            elif state == "EXPECT_COLON":
                if char == ':': state = "EXPECT_VALUE"
            elif state == "EXPECT_VALUE":
                if char == '"': state = "IN_STRING_VALUE"
                elif char in '0123456789-.': state = "IN_NUMBER_VALUE"
                elif char == '{':
                    state = "EXPECT_KEY"
                    stack.append({"key": current_key, "completed": set()})
                elif char in 'tfn': state = "IN_LITERAL_VALUE"
            elif state in ["IN_STRING_VALUE", "IN_NUMBER_VALUE", "IN_LITERAL_VALUE"]:
                if char == ',':
                    stack[-1]["completed"].add(current_key)
                    state = "EXPECT_KEY"
                elif char in '}]':
                    stack[-1]["completed"].add(current_key)
                    state = "EXPECT_COMMA"
                    if char == '}' and len(stack) > 1:
                        completed_obj_key = stack.pop()["key"]
                        stack[-1]["completed"].add(completed_obj_key)
            elif state == "EXPECT_COMMA":
                if char == ',': state = "EXPECT_KEY"
                elif char in '}]':
                    if char == '}' and len(stack) > 1:
                        completed_obj_key = stack.pop()["key"]
                        stack[-1]["completed"].add(completed_obj_key)

        return state, stack

    def extract_batch(self, prompts: List[str], function_names: List[str],
                      functions: List[Dict[str, Any]], max_new_tokens: int = 180,
                      verbose: bool = False) -> List[str]:
        batch_size = len(prompts)
        input_sequences = []
        generated_sequences = []

        for prompt, target_name in zip(prompts, function_names):
            targeted_examples = self.formatter.load_examples(target_name)
            primed_prompt = self.formatter.build_extraction_prompt(
                user_prompt=prompt, target_name=target_name, functions=functions, examples=targeted_examples
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
                target_name = function_names[orig_i]
                target_schema = next((f for f in functions if f["name"] == target_name), {})
                key_map, type_map = self._build_schema_maps(target_schema.get("parameters", {}))

                indiv_mask = self._get_mask(current_prefix, key_map, type_map)

                if indiv_mask.shape[0] < self.vocab_size:
                    padded_indiv = np.zeros(self.vocab_size, dtype=bool)
                    padded_indiv[:indiv_mask.shape[0]] = indiv_mask
                    indiv_mask = padded_indiv
                elif indiv_mask.shape[0] > self.vocab_size:
                    indiv_mask = indiv_mask[:self.vocab_size]

                mask_matrix[idx] = indiv_mask

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

        return [self.tokenizer.decode(seq) for seq in generated_sequences]

    def _get_mask(self, current_prefix: str, key_map: Dict[str, List[str]], type_map: Dict[str, str]) -> np.ndarray:
        mask = np.ones(self.vocab_size, dtype=bool)
        state, stack = self._get_json_state(current_prefix)
        in_string = state in ["IN_KEY", "IN_STRING_VALUE"]

        if not in_string:
            mask &= self.struct_mask
            mask &= self.safe_quote_mask
            if current_prefix.strip(): mask &= self.banned_keyword_mask

            if state == "EXPECT_VALUE":
                key_match = re.findall(r'"([^"]+)"\s*:', current_prefix)
                if key_match:
                    pending_key = key_match[-1]
                    expected_type = type_map.get(pending_key, "string")
                    if expected_type == "object":
                        mask &= self.type_masks["object"]
                    elif expected_type in ["integer", "number"]:
                        mask &= self.type_masks["number"]
                    elif expected_type == "string":
                        mask &= self.type_masks["string"]
                    elif expected_type == "boolean":
                        mask &= self.type_masks["boolean"]

            elif state in ["EXPECT_KEY", "EXPECT_COMMA"]:
                parent_key = stack[-1]["key"]
                completed = stack[-1]["completed"]
                valid_keys = [k for k in key_map.get(parent_key, []) if k not in completed]

                if not valid_keys:
                    # Mathematical Funnel: If no keys are left, ban quotes and commas to force the `}` closure!
                    for t_id in np.where(mask)[0]:
                        if state == "EXPECT_KEY" and '"' in self.clean_vocab[t_id]:
                            mask[t_id] = False
                        if state == "EXPECT_COMMA" and ',' in self.clean_vocab[t_id]:
                            mask[t_id] = False

        else:
            mask &= self.wildcard_string_mask
            mask &= self.safe_quote_mask

            if state == "IN_KEY":
                parent_key = stack[-1]["key"]
                completed = stack[-1]["completed"]
                valid_keys = [k for k in key_map.get(parent_key, []) if k not in completed]

                key_match = re.search(r'"([^"]*)$', current_prefix)
                if key_match:
                    current_key_prefix = key_match.group(1)

                    cache_key = f"{','.join(sorted(valid_keys))}|{current_key_prefix}"
                    if cache_key not in self.key_mask_cache:
                        valid_mask = np.zeros(self.vocab_size, dtype=bool)
                        base_mask = self.wildcard_string_mask & self.safe_quote_mask
                        valid_indices = np.where(base_mask)[0]
                        for t_id in valid_indices:
                            s = self.clean_vocab[t_id]
                            if s:
                                s_val = s[:-1] if s.endswith('"') else s
                                proposed = current_key_prefix + s_val
                                is_allowed = any(vk.startswith(proposed) for vk in valid_keys)
                                is_closure = any(proposed == vk for vk in valid_keys) and s.endswith('"')
                                if is_allowed and (not s.endswith('"') or is_closure):
                                    valid_mask[t_id] = True
                        self.key_mask_cache[cache_key] = valid_mask
                    mask &= self.key_mask_cache[cache_key]

        return mask

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
