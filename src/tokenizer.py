"""
Implementation Note (SDK Discrepancy):
The project subject explicitly instructs the use of the method
`get_path_to_vocabulary_json()` to retrieve the token mappings.
However, inspection of the provided `Small_LLM_Model` SDK reveals
this method is actually implemented as `get_path_to_vocab_file()`.
This module uses the implemented method to ensure successful execution.
"""

import json
from typing import List, Dict
from llm_sdk import Small_LLM_Model

class Tokenizer:
    def __init__(self) -> None:
        """
        Initializes the tokenizer by loading the vocabulary from the SDK
        and building the byte-level translation dictionaries.
        """
        self.sdk = Small_LLM_Model()
        vocab_path = self.vocab_path = self.sdk.get_path_to_vocab_file()

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.token_to_id: Dict[str, int] = json.load(f)

        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

        self.byte_encoder: Dict[int, str] = self._get_byte_to_unicode_mapping()
        self.byte_decoder: Dict[str, int] = {v: k for k, v in self.byte_encoder.items()}

    @staticmethod
    def _get_byte_to_unicode_mapping() -> Dict[int, str]:
        """Generates the standard BPE byte-to-unicode mapping."""
        bs = (
            list(range(ord("!"), ord("~") + 1)) +
            list(range(ord("¡"), ord("¬") + 1)) +
            list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]

        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1

        return {byte: chr(unicode_val) for byte, unicode_val in zip(bs, cs)}

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of token IDs into a standard UTF-8 string.
        1. Translates the IDs back into their BPE string representations.
        2. Joinis them into a single continuous sequence.
        3. Maps every single character back to its raw byte integer (0-255).
        4. Shovels the integers into a bytearray and decode to standard text.
        The errors='replace' flag ensures that if the LLM hallucinated an invalid
        byte sequence, it inserts a '' instead of crashing your whole program.
        """
        if not token_ids:
            return ""

        bpe_tokens = [self.id_to_token.get(tok_id, "") for tok_id in token_ids]
        bpe_text = "".join(bpe_tokens)
        raw_bytes = [self.byte_decoder[char] for char in bpe_text]

        return bytearray(raw_bytes).decode('utf-8', errors='replace')

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        """
        pass
