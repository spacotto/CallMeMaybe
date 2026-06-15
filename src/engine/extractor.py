import numpy as np
from typing import List, Dict, Any, Set, Tuple

from src.engine.classifier import FunctionClassifier
from src.visualizer import Visualizer


class JSONParserState:
    """Incremental O(1) State Machine to track JSON generation."""

    def __init__(self) -> None:
        self.state = "EXPECT_KEY"
        self.stack: List[Dict[str, Any]] = [
            {"key": "parameters", "completed": set()}
        ]
        self.escape = False
        self.current_key = ""
        self.pending_key = ""
        self.depth = 2  # Starts inside {"name": "...", "parameters": {

    def update(self, new_chars: str) -> None:
        for char in new_chars:
            if self.escape:
                self.escape = False
                continue
            if char == "\\":
                self.escape = True
                continue

            # Track bracket depth for O(1) completion checking
            if self.state not in ["IN_KEY", "IN_STRING_VALUE"]:
                if char == '{':
                    self.depth += 1
                elif char == '}':
                    self.depth -= 1

            if self.state == "EXPECT_KEY":
                if char == '"':
                    self.state = "IN_KEY"
                    self.current_key = ""
                elif char in '}]':
                    self.state = "EXPECT_COMMA"
                    if char == '}' and len(self.stack) > 1:
                        completed_key = self.stack.pop()["key"]
                        self.stack[-1]["completed"].add(completed_key)

            elif self.state == "IN_KEY":
                if char == '"':
                    self.state = "EXPECT_COLON"
                    self.pending_key = self.current_key
                else:
                    self.current_key += char

            elif self.state == "EXPECT_COLON":
                if char == ':':
                    self.state = "EXPECT_VALUE"

            elif self.state == "EXPECT_VALUE":
                if char == '"':
                    self.state = "IN_STRING_VALUE"
                elif char in '0123456789-.':
                    self.state = "IN_NUMBER_VALUE"
                elif char == '{':
                    self.state = "EXPECT_KEY"
                    self.stack.append({
                        "key": self.pending_key,
                        "completed": set()
                    })
                elif char in 'tfn':
                    self.state = "IN_LITERAL_VALUE"

            elif self.state == "IN_STRING_VALUE":
                if char == '"':
                    self.state = "EXPECT_COMMA"
                    self.stack[-1]["completed"].add(self.pending_key)

            elif self.state in ["IN_NUMBER_VALUE", "IN_LITERAL_VALUE"]:
                if char == ',':
                    self.stack[-1]["completed"].add(self.pending_key)
                    self.state = "EXPECT_KEY"
                elif char in '}]':
                    self.stack[-1]["completed"].add(self.pending_key)
                    self.state = "EXPECT_COMMA"
                    if char == '}' and len(self.stack) > 1:
                        completed_key = self.stack.pop()["key"]
                        self.stack[-1]["completed"].add(completed_key)

            elif self.state == "EXPECT_COMMA":
                if char == ',':
                    self.state = "EXPECT_KEY"
                elif char in '}]':
                    if char == '}' and len(self.stack) > 1:
                        completed_key = self.stack.pop()["key"]
                        self.stack[-1]["completed"].add(completed_key)


class SchemaExtractor:
    """Phase 2 (Fast Path): Context-Aware CFG parser with key-tracking."""

    def __init__(self, classifier_instance: FunctionClassifier) -> None:
        self.sdk = classifier_instance.sdk
        self.tokenizer = classifier_instance.tokenizer
        self.formatter = classifier_instance.formatter
        self.vocab_size = classifier_instance.vocab_size
        self.clean_vocab = classifier_instance.clean_vocab

        self.struct_mask = np.zeros(self.vocab_size, dtype=bool)
        self.wildcard_string_mask = np.zeros(self.vocab_size, dtype=bool)
        self.safe_quote_mask = np.ones(self.vocab_size, dtype=bool)

        struct_chars = set('{}[]:,"0123456789 \n\r\ttruefalsenull.-eE')
        wildcard_chars = set(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789 '
            '-.,:;!?@"\'/\\()[]{}*+^$|=~`%&<>'
        )

        self.stripped_vocab = [""] * self.vocab_size

        for t_id, s in enumerate(self.clean_vocab):
            if not s:
                continue

            self.stripped_vocab[t_id] = s.lstrip(' \n\r\t')

            t_chars = set(s)
            if t_chars.issubset(struct_chars):
                self.struct_mask[t_id] = True
            if t_chars.issubset(wildcard_chars):
                self.wildcard_string_mask[t_id] = True
            if '"' in s and (s.count('"') > 1 or not s.endswith('"')):
                self.safe_quote_mask[t_id] = False

        self.type_masks = {
            "object": np.zeros(self.vocab_size, dtype=bool),
            "string": np.zeros(self.vocab_size, dtype=bool),
            "number": np.zeros(self.vocab_size, dtype=bool),
            "boolean": np.zeros(self.vocab_size, dtype=bool)
        }

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

        self.struct_mask_indices = np.where(self.struct_mask)[0]
        base_wildcard = self.wildcard_string_mask & self.safe_quote_mask
        self.wildcard_safe_indices = np.where(base_wildcard)[0]

        self.key_mask_cache: Dict[str, np.ndarray] = {}

        self.global_schema_cache: Dict[
            str, Tuple[Dict[str, List[str]], Dict[str, str]]
        ] = {}

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
        current_prefixes = []

        parser_states = [JSONParserState() for _ in prompts]

        for prompt, target_name in zip(prompts, function_names):
            examples = self.formatter.load_examples(target_name)
            primed = self.formatter.build_extraction_prompt(
                user_prompt=prompt,
                target_name=target_name,
                functions=functions,
                examples=examples
            )
            input_sequences.append(self.tokenizer.encode(primed))
            start_str = f'{{"name": "{target_name}", "parameters": {{'
            generated_sequences.append(self.tokenizer.encode(start_str))
            current_prefixes.append(start_str)

        is_finished = [False] * len(prompts)
        absolute_max_steps = (
            max(max_new_tokens_list) if max_new_tokens_list else 180
        )

        for func in functions:
            name = func["name"]
            if name not in self.global_schema_cache:
                ps = func.get("parameters", {})
                self.global_schema_cache[name] = self._build_schema_maps(ps)

        for step in range(absolute_max_steps):
            if all(is_finished):
                break

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
                name = function_names[orig_i]

                key_map, type_map = self.global_schema_cache.get(
                    name, ({"parameters": []}, {})
                )

                mask = self._get_mask(
                    parser_states[orig_i], key_map, type_map
                )

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

                new_char = self.tokenizer.decode([next_id])
                current_prefixes[orig_i] += new_char

                parser_states[orig_i].update(new_char)

                if verbose:
                    allowed_count = int(np.sum(mask_matrix[idx]))
                    # Cast to Set[str] to satisfy Mypy
                    dummy_set: Set[str] = set(
                        str(x) for x in range(allowed_count)
                    )
                    Visualizer.print_status(
                        step,
                        new_char,
                        dummy_set,
                        parser_states[orig_i].state
                    )

                end_token = self.tokenizer.token_to_id.get("<|im_end|>", -1)
                if parser_states[orig_i].depth == 0 or next_id == end_token:
                    is_finished[orig_i] = True

        if verbose:
            Visualizer.print_div()
        return current_prefixes

    def _get_mask(
        self, parser_state: JSONParserState, key_map: Dict[str, List[str]],
        type_map: Dict[str, str]
    ) -> np.ndarray:
        mask = np.ones(self.vocab_size, dtype=bool)
        state = parser_state.state
        stack = parser_state.stack
        in_string = state in ["IN_KEY", "IN_STRING_VALUE"]

        if not in_string:
            struct_allowed = np.zeros(self.vocab_size, dtype=bool)

            parent_key = stack[-1]["key"] if stack else ""
            completed: Set[str] = stack[-1]["completed"] if stack else set()
            valid = [
                k for k in key_map.get(parent_key, []) if k not in completed
            ]

            for t_id in self.struct_mask_indices:
                s_strip = self.stripped_vocab[t_id]

                if state == "EXPECT_KEY":
                    if not s_strip:
                        struct_allowed[t_id] = True
                    elif s_strip.startswith('}') and not valid:
                        struct_allowed[t_id] = True
                    elif s_strip.startswith('"'):
                        s_content = s_strip[1:]
                        if not s_content:
                            struct_allowed[t_id] = True
                        else:
                            if s_content.endswith('"'):
                                s_val = s_content[:-1]
                            else:
                                s_val = s_content

                            allowed = any(v.startswith(s_val) for v in valid)
                            closure = (
                                any(s_val == v for v in valid)
                                and s_content.endswith('"')
                            )
                            if allowed and (
                                not s_content.endswith('"') or closure
                            ):
                                struct_allowed[t_id] = True

                elif state == "EXPECT_COLON":
                    if not s_strip or s_strip.startswith(':'):
                        struct_allowed[t_id] = True

                elif state == "EXPECT_COMMA":
                    if not s_strip:
                        struct_allowed[t_id] = True
                    elif s_strip.startswith(','):
                        if valid:
                            struct_allowed[t_id] = True
                    elif s_strip.startswith('}'):
                        if not valid:
                            struct_allowed[t_id] = True

                elif state == "EXPECT_VALUE":
                    if not s_strip or s_strip[0] in '"0123456789tfn{[':
                        struct_allowed[t_id] = True

            mask &= struct_allowed
            mask &= self.safe_quote_mask

            if state == "EXPECT_VALUE":
                e_type = type_map.get(parser_state.pending_key, "string")
                if e_type == "object":
                    mask &= self.type_masks["object"]
                elif e_type in ["integer", "number"]:
                    mask &= self.type_masks["number"]
                elif e_type == "string":
                    mask &= self.type_masks["string"]
                elif e_type == "boolean":
                    mask &= self.type_masks["boolean"]

        else:
            mask &= self.wildcard_string_mask
            mask &= self.safe_quote_mask

            if state == "IN_KEY":
                parent = stack[-1]["key"] if stack else ""
                completed_keys: Set[str] = (
                    stack[-1]["completed"] if stack else set()
                )
                valid = [
                    k for k in key_map.get(parent, [])
                    if k not in completed_keys
                ]

                prefix = parser_state.current_key
                cache_k = f"{','.join(sorted(valid))}|{prefix}"

                if cache_k not in self.key_mask_cache:
                    v_mask = np.zeros(self.vocab_size, dtype=bool)

                    for t_id in self.wildcard_safe_indices:
                        s = self.clean_vocab[t_id]
                        if s:
                            if s.endswith('"'):
                                s_val = s[:-1]
                            else:
                                s_val = s

                            prop = prefix + s_val
                            allowed = any(v.startswith(prop) for v in valid)
                            closure = (
                                any(prop == v for v in valid)
                                and s.endswith('"')
                            )
                            if allowed and (not s.endswith('"') or closure):
                                v_mask[t_id] = True
                    self.key_mask_cache[cache_k] = v_mask
                mask &= self.key_mask_cache[cache_k]

        if not np.any(mask):
            mask = self.wildcard_string_mask | self.safe_quote_mask

        return mask
