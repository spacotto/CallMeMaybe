import numpy as np
import re
import json
from typing import List, Dict, Any, Set, Tuple

from src.engine.classifier import FunctionClassifier


class NestedExtractor:
    """Phase 2 (Slow Path): Context-Aware CFG parser with key-tracking."""

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
        wildcard_chars = set(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789 '
            '-.,:;!?@"\'/\\()[]{}*+^$|=~`%&<>'
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
                valid_words = ['true', 'false', 'null']
                if not any(v.startswith(word)
                           or word in v for v in valid_words):
                    self.banned_keyword_mask[t_id] = False
                    break

        self.type_masks = {
            "object": np.zeros(self.vocab_size, dtype=bool),
            "string": np.zeros(self.vocab_size, dtype=bool),
            "number": np.zeros(self.vocab_size, dtype=bool),
            "boolean": np.zeros(self.vocab_size, dtype=bool)
        }
        self.contains_closing_brace_mask = np.zeros(
            self.vocab_size, dtype=bool
        )

        for t_id, s in enumerate(self.clean_vocab):
            if not s:
                continue
            if all(c in ' \n\r\t{' for c in s):
                self.type_masks["object"][t_id] = True
            if all(c in ' \n\r\t"' for c in s):
                self.type_masks["string"][t_id] = True
            if all(c in ' \n\r\t0123456789-.' for c in s):
                self.type_masks["number"][t_id] = True
            if all(c in ' \n\r\ttrufalse' for c in s):
                self.type_masks["boolean"][t_id] = True
            if '}' in s:
                self.contains_closing_brace_mask[t_id] = True

        self.key_mask_cache: Dict[str, np.ndarray] = {}

    def _build_schema_maps(
        self, properties: Dict[str, Any]
    ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        key_map: Dict[str, List[str]] = {"parameters": list(properties.keys())}
        type_map: Dict[str, str] = {}

        def traverse(k: str, v: Dict[str, Any]) -> None:
            t = v.get("type", "string")
            type_map[k] = t
            if t == "object" and "properties" in v:
                sub_props = v["properties"]
                key_map[k] = list(sub_props.keys())
                for sub_k, sub_v in sub_props.items():
                    traverse(sub_k, sub_v)

        for k, v in properties.items():
            traverse(k, v)
        return key_map, type_map

    def _get_json_state(
        self, current_prefix: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        match_obj = re.search(r'"parameters"\s*:\s*\{', current_prefix)
        stack: List[Dict[str, Any]] = [
            {"key": "parameters", "completed": set()}
        ]

        if not match_obj:
            return "UNKNOWN", stack

        json_part = current_prefix[match_obj.end():]
        state = "EXPECT_KEY"
        escape = False
        current_key = ""

        for char in json_part:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue

            if state == "EXPECT_KEY":
                if char == '"':
                    state = "IN_KEY"
                    current_key = ""
                elif char in '}]':
                    state = "EXPECT_COMMA"
                    if char == '}' and len(stack) > 1:
                        completed_key = stack.pop()["key"]
                        completed_set: Set[str] = stack[-1]["completed"]
                        completed_set.add(completed_key)
            elif state == "IN_KEY":
                if char == '"':
                    state = "EXPECT_COLON"
                else:
                    current_key += char
            elif state == "EXPECT_COLON":
                if char == ':':
                    state = "EXPECT_VALUE"
            elif state == "EXPECT_VALUE":
                if char == '"':
                    state = "IN_STRING_VALUE"
                elif char in '0123456789-.':
                    state = "IN_NUMBER_VALUE"
                elif char == '{':
                    state = "EXPECT_KEY"
                    stack.append({"key": current_key, "completed": set()})
                elif char in 'tfn':
                    state = "IN_LITERAL_VALUE"
            elif state in [
                "IN_STRING_VALUE", "IN_NUMBER_VALUE", "IN_LITERAL_VALUE"
            ]:
                if char == ',':
                    completed_set2: Set[str] = stack[-1]["completed"]
                    completed_set2.add(current_key)
                    state = "EXPECT_KEY"
                elif char in '}]':
                    completed_set3: Set[str] = stack[-1]["completed"]
                    completed_set3.add(current_key)
                    state = "EXPECT_COMMA"
                    if char == '}' and len(stack) > 1:
                        completed_key = stack.pop()["key"]
                        completed_set4: Set[str] = stack[-1]["completed"]
                        completed_set4.add(completed_key)
            elif state == "EXPECT_COMMA":
                if char == ',':
                    state = "EXPECT_KEY"
                elif char in '}]':
                    if char == '}' and len(stack) > 1:
                        completed_key = stack.pop()["key"]
                        completed_set5: Set[str] = stack[-1]["completed"]
                        completed_set5.add(completed_key)

        return state, stack

    def extract_batch(
        self,
        prompts: List[str],
        function_names: List[str],
        functions: List[Dict[str, Any]],
        max_new_tokens_list: List[int],
        verbose: bool = False
    ) -> List[str]:
        input_sequences = []
        generated_sequences = []

        for prompt, target_name in zip(prompts, function_names):
            examples = self.formatter.load_examples(target_name)
            primed = self.formatter.build_extraction_prompt(
                user_prompt=prompt,
                target_name=target_name,
                functions=functions,
                examples=examples
            )
            input_sequences.append(self.tokenizer.encode(primed))
            escaped = json.dumps(prompt)[1:-1]
            start_str = (
                f'{{"prompt": "{escaped}", "name": "{target_name}", '
                f'"parameters": {{'
            )
            generated_sequences.append(self.tokenizer.encode(start_str))

        is_finished = [False] * len(prompts)

        # 🔄 Calculate the absolute maximum iteration boundary from your custom list
        absolute_max_steps = max(max_new_tokens_list) if max_new_tokens_list else 180

        for step in range(absolute_max_steps):
            if all(is_finished):
                break

            # 🔄 Force-finish individual prompts if they cross their personal limit
            for i in range(len(prompts)):
                if not is_finished[i] and step >= max_new_tokens_list[i]:
                    is_finished[i] = True

            active_idx = [i for i, f in enumerate(is_finished) if not f]
            if not active_idx:
                break

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
                name = function_names[orig_i]
                schema = next((f for f in functions if f["name"] == name), {})
                params = schema.get("parameters", {})
                key_map, type_map = self._build_schema_maps(params)

                mask = self._get_mask(curr, key_map, type_map)
                if mask.shape[0] < self.vocab_size:
                    padded = np.zeros(self.vocab_size, dtype=bool)
                    padded[:mask.shape[0]] = mask
                    mask = padded
                mask_matrix[idx] = mask

            if mask_matrix.shape[1] < logits_matrix.shape[1]:
                padded_batch_mask = np.zeros(
                    (len(active_idx), logits_matrix.shape[1]), dtype=bool
                )
                padded_batch_mask[:, :mask_matrix.shape[1]] = mask_matrix
                mask_matrix = padded_batch_mask
            elif mask_matrix.shape[1] > logits_matrix.shape[1]:
                mask_matrix = mask_matrix[:, :logits_matrix.shape[1]]

            logits_matrix[~mask_matrix] = -np.inf
            next_ids = np.argmax(logits_matrix, axis=1)

            for idx, orig_i in enumerate(active_idx):
                next_id = int(next_ids[idx])
                generated_sequences[orig_i].append(next_id)
                new_seq = generated_sequences[orig_i]
                if self._is_complete(self.tokenizer.decode(new_seq), next_id):
                    is_finished[orig_i] = True

        return [self.tokenizer.decode(s) for s in generated_sequences]

    def _get_mask(
        self, current_prefix: str, key_map: Dict[str, List[str]],
        type_map: Dict[str, str]
    ) -> np.ndarray:
        mask = np.ones(self.vocab_size, dtype=bool)
        state, stack = self._get_json_state(current_prefix)
        in_string = state in ["IN_KEY", "IN_STRING_VALUE"]

        if not in_string:
            mask &= self.struct_mask
            mask &= self.safe_quote_mask
            if current_prefix.strip():
                mask &= self.banned_keyword_mask

            if state == "EXPECT_VALUE":
                matches = re.findall(r'"([^"]+)"\s*:', current_prefix)
                if matches:
                    pending_key = matches[-1]
                    e_type = type_map.get(pending_key, "string")
                    if e_type == "object":
                        mask &= self.type_masks["object"]
                    elif e_type in ["integer", "number"]:
                        mask &= self.type_masks["number"]
                    elif e_type == "string":
                        mask &= self.type_masks["string"]
                    elif e_type == "boolean":
                        mask &= self.type_masks["boolean"]

            elif state in ["EXPECT_KEY", "EXPECT_COMMA"]:
                parent_key = stack[-1]["key"]
                completed: Set[str] = stack[-1]["completed"]
                valid = [
                    k for k in key_map.get(parent_key, [])
                    if k not in completed
                ]
                if not valid:
                    for t_id in np.where(mask)[0]:
                        v_char = self.clean_vocab[t_id]
                        if state == "EXPECT_KEY" and '"' in v_char:
                            mask[t_id] = False
                        if state == "EXPECT_COMMA" and ',' in v_char:
                            mask[t_id] = False
        else:
            mask &= self.wildcard_string_mask
            mask &= self.safe_quote_mask

            if state == "IN_KEY":
                parent = stack[-1]["key"]
                completed_keys: Set[str] = stack[-1]["completed"]
                valid = [
                    k for k in key_map.get(parent, [])
                    if k not in completed_keys
                ]
                key_match = re.search(r'"([^"]*)$', current_prefix)
                if key_match:
                    prefix = key_match.group(1)
                    cache_k = f"{','.join(sorted(valid))}|{prefix}"
                    if cache_k not in self.key_mask_cache:
                        v_mask = np.zeros(self.vocab_size, dtype=bool)
                        base = (
                            self.wildcard_string_mask &
                            self.safe_quote_mask
                        )
                        for t_id in np.where(base)[0]:
                            s = self.clean_vocab[t_id]
                            if s:
                                s_val = s[:-1] if s.endswith('"') else s
                                prop = prefix + s_val
                                allowed = any(
                                    v.startswith(prop) for v in valid
                                )
                                closure = any(
                                    prop == v for v in valid
                                ) and s.endswith('"')
                                if allowed and (
                                    not s.endswith('"') or closure
                                ):
                                    v_mask[t_id] = True
                        self.key_mask_cache[cache_k] = v_mask
                    mask &= self.key_mask_cache[cache_k]

        return mask

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
