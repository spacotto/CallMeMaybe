import numpy as np
import json
from typing import List, Dict, Any

from src.tokenizer import Tokenizer
from src.formatter import Formatter, ModelFormat
from llm_sdk import Small_LLM_Model
from src.engine.trie import SchemaTrie

class FunctionClassifier:
    """Pass 1: Ultra-fast Zero-Shot classification to identify the target function."""
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.sdk = Small_LLM_Model(model_name=model_name)
        self.tokenizer = Tokenizer()

        if "qwen" in model_name.lower() or "smol" in model_name.lower():
            self.formatter = Formatter(format_type=ModelFormat.CHATML)
        else:
            self.formatter = Formatter(format_type=ModelFormat.INSTRUCT)

        # Precompute minimal vocabulary setup (only string literals needed for classification)
        max_id = max(self.tokenizer.id_to_token.keys()) if self.tokenizer.id_to_token else 151643
        self.vocab_size = max_id + 1
        self.clean_vocab = [""] * self.vocab_size

        for t_id, t_str in self.tokenizer.id_to_token.items():
            self.clean_vocab[t_id] = t_str.replace("Ġ", " ")

    def classify_batch(self, prompts: List[str], functions: List[Dict[str, Any]], max_new_tokens: int = 30) -> List[str]:
        batch_size = len(prompts)
        valid_names = [f["name"] for f in functions]
        name_trie = SchemaTrie(valid_names)

        input_sequences = []
        generated_sequences = []

        for prompt in prompts:
            primed_prompt = self.formatter.build_classification_prompt(prompt, functions)
            input_sequences.append(self.tokenizer.encode(primed_prompt))

            escaped_prompt = json.dumps(prompt)[1:-1]
            boilerplate_start = f'{{"prompt": "{escaped_prompt}", "name": "'
            generated_sequences.append(self.tokenizer.encode(boilerplate_start))

        is_finished = [False] * batch_size
        extracted_names = [""] * batch_size

        # Classification Loop
        for step in range(max_new_tokens):
            if all(is_finished):
                break

            active_indices = [i for i, finished in enumerate(is_finished) if not finished]

            batch_logits = []
            for i in active_indices:
                seq = input_sequences[i] + generated_sequences[i]
                logits = self.sdk.get_logits_from_input_ids(seq)
                logits_np = np.array(logits, dtype=np.float32)
                if len(logits_np.shape) > 1:
                    logits_np = logits_np[-1]
                batch_logits.append(logits_np)

            logits_matrix = np.stack(batch_logits)
            mask_matrix = np.zeros((len(active_indices), self.vocab_size), dtype=bool)

            for idx, orig_i in enumerate(active_indices):
                current_prefix = self.tokenizer.decode(generated_sequences[orig_i])
                parts = current_prefix.rsplit('"', 1)
                current_name_prefix = parts[1] if len(parts) > 1 else ""

                start_node = name_trie.get_node(current_name_prefix)
                if start_node is not None:
                    for t_id in range(self.vocab_size):
                        s = self.clean_vocab[t_id]
                        if s and name_trie.is_valid_suffix(start_node, s):
                            mask_matrix[idx, t_id] = True

            # Align Mask Shape
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

                # Stop the moment the closing quote is typed
                if new_prefix.endswith('"'):
                    is_finished[orig_i] = True
                    parts = new_prefix.rsplit('"', 2)
                    extracted_names[orig_i] = parts[-2]

        return extracted_names
