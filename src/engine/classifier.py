import numpy as np
import json
from typing import List, Dict, Any

from src.tokenizer import Tokenizer
from src.formatter import Formatter, ModelFormat
from llm_sdk import Small_LLM_Model  # type: ignore
from src.engine.trie import SchemaTrie


class FunctionClassifier:
    """Phase 1: Ultra-fast Zero-Shot classification with O(1) caching."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.sdk = Small_LLM_Model(model_name=model_name)
        self.tokenizer = Tokenizer()

        if "qwen" in model_name.lower() or "smol" in model_name.lower():
            self.formatter = Formatter(format_type=ModelFormat.CHATML)
        else:
            self.formatter = Formatter(format_type=ModelFormat.INSTRUCT)

        ids = self.tokenizer.id_to_token.keys()
        max_id = max(ids) if ids else 151643
        self.vocab_size = max_id + 1
        self.clean_vocab = [""] * self.vocab_size

        for t_id, t_str in self.tokenizer.id_to_token.items():
            self.clean_vocab[t_id] = t_str.replace("Ġ", " ")

        # O(1) Lazy Cache for bitwise masks
        self.prefix_mask_cache: Dict[str, np.ndarray] = {}

    def classify_batch(
        self,
        prompts: List[str],
        functions: List[Dict[str, Any]],
        max_new_tokens: int = 10
    ) -> List[str]:
        batch_size = len(prompts)
        valid_names = [f["name"] for f in functions]
        name_trie = SchemaTrie(valid_names)

        input_sequences = []
        generated_sequences = []
        current_prefixes = [""] * batch_size

        for prompt in prompts:
            primed = self.formatter.build_classification_prompt(
                prompt, functions
            )
            input_sequences.append(self.tokenizer.encode(primed))

            escaped_prompt = json.dumps(prompt)[1:-1]
            boilerplate_start = f'{{"prompt": "{escaped_prompt}", "name": "'
            generated_sequences.append(
                self.tokenizer.encode(boilerplate_start)
            )

        is_finished = [False] * batch_size
        extracted_names = [""] * batch_size

        for step in range(max_new_tokens):
            if all(is_finished):
                break

            active_indices = [
                i for i, finished in enumerate(is_finished) if not finished
            ]

            batch_logits = []
            for i in active_indices:
                seq = input_sequences[i] + generated_sequences[i]
                logits = self.sdk.get_logits_from_input_ids(seq)
                logits_np = np.array(logits, dtype=np.float32)
                if len(logits_np.shape) > 1:
                    logits_np = logits_np[-1]
                batch_logits.append(logits_np)

            logits_matrix = np.stack(batch_logits)
            mask_matrix = np.zeros(
                (len(active_indices), self.vocab_size), dtype=bool
            )

            for idx, orig_i in enumerate(active_indices):
                prefix = current_prefixes[orig_i]

                if prefix not in self.prefix_mask_cache:
                    v_mask = np.zeros(self.vocab_size, dtype=bool)
                    start_node = name_trie.get_node(prefix)
                    if start_node is not None:
                        for t_id in range(self.vocab_size):
                            s = self.clean_vocab[t_id]
                            if s and name_trie.is_valid_suffix(start_node, s):
                                v_mask[t_id] = True
                    self.prefix_mask_cache[prefix] = v_mask

                mask_matrix[idx, :] = self.prefix_mask_cache[prefix]

            # Align Mask Shape
            if mask_matrix.shape[1] < logits_matrix.shape[1]:
                padded = np.zeros(
                    (len(active_indices), logits_matrix.shape[1]), dtype=bool
                )
                padded[:, :mask_matrix.shape[1]] = mask_matrix
                mask_matrix = padded
            elif mask_matrix.shape[1] > logits_matrix.shape[1]:
                mask_matrix = mask_matrix[:, :logits_matrix.shape[1]]

            logits_matrix[~mask_matrix] = -np.inf
            next_token_ids = np.argmax(logits_matrix, axis=1)

            for idx, orig_i in enumerate(active_indices):
                next_id = int(next_token_ids[idx])
                generated_sequences[orig_i].append(next_id)

                new_char = self.tokenizer.decode([next_id])

                if '"' in new_char:
                    is_finished[orig_i] = True
                    final_part = new_char.split('"')[0]
                    extracted_names[orig_i] = current_prefixes[orig_i] + final_part
                else:
                    current_prefixes[orig_i] += new_char

        return extracted_names
