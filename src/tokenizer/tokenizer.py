"""
Custom Byte-Level BPE (BBPE) Tokenizer implementation.

This module translates raw text into integer token IDs and vice versa.
It leverages a byte-to-unicode mapping to ensure that any arbitrary
string (including unknown characters) can be tokenized without
Out-Of-Vocabulary (OOV) errors.

Implementation Note (SDK Discrepancy):
The project subject explicitly instructs the use of the method
`get_path_to_vocabulary_json()` to retrieve the token mappings.
However, inspection of the provided `Small_LLM_Model` SDK reveals
this method is actually implemented as `get_path_to_vocab_file()`.
This module uses the implemented method to ensure successful execution.
"""

import json
from typing import List, Dict
from llm_sdk import Small_LLM_Model  # type: ignore


class Tokenizer:
    """
    Handles encoding and decoding of text using Byte-Level BPE.

    By mapping all 256 individual bytes to specific unicode characters
    before applying BPE tokenization, this class ensures complete
    coverage of the UTF-8 spectrum, preventing crashes when the LLM
    encounters unexpected characters or emojis.

    Attributes:
        sdk (Small_LLM_Model): The initialized language model SDK.
        vocab_path (str): The absolute path to the vocabulary JSON.
        token_to_id (Dict[str, int]): Maps BPE strings to integer IDs.
        id_to_token (Dict[int, str]): Maps integer IDs to BPE strings.
        byte_encoder (Dict[int, str]): Maps raw bytes to unicode chars.
        byte_decoder (Dict[str, int]): Maps unicode chars to raw bytes.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """
        Initializes the tokenizer and loads the vocabulary.

        Args:
            model_name (str): The identifier for the LLM weights and
                vocabulary to load. Defaults to Qwen3-0.6B.
        """
        # Pass the model_name to ensure the correct vocabulary is loaded
        self.sdk = Small_LLM_Model(model_name=model_name)
        vocab_path = self.vocab_path = self.sdk.get_path_to_vocab_file()

        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.token_to_id: Dict[str, int] = json.load(f)

        self.id_to_token: Dict[int, str] = {
                v: k for k, v in self.token_to_id.items()
        }

        self.byte_encoder: Dict[int, str] = self._get_byte_to_unicode_mapping()
        self.byte_decoder: Dict[str, int] = {
                v: k for k, v in self.byte_encoder.items()
        }

    @staticmethod
    def _get_byte_to_unicode_mapping() -> Dict[int, str]:
        """
        Generates the standard BPE byte-to-unicode mapping.

        This function shifts unprintable or control bytes into a
        printable unicode range. This is the foundational trick of
        BBPE that allows the model to treat raw bytes as standard
        characters during tokenization.

        Returns:
            Dict[int, str]: A dictionary mapping integer bytes (0-255)
            to their corresponding printable unicode characters.
        """
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
        Decodes a list of token IDs back into a UTF-8 string.

        This method reverses the BBPE process: it maps IDs to BPE
        tokens, joins them, translates the mapped characters back into
        their original raw bytes, and decodes the bytearray into text.

        Args:
            token_ids (List[int]): The sequence of model-generated IDs.

        Returns:
            str: The human-readable decoded string. Replaces malformed
            bytes with the standard unicode replacement character.
        """
        if not token_ids:
            return ""

        # Translate the IDs back into their BPE string representations
        # Example: [892, 318] -> ["What", "Ġis"]
        bpe_tokens = [self.id_to_token.get(tok_id, "") for tok_id in token_ids]

        # Join them into a single continuous sequence
        # Example: "WhatĠis"
        bpe_text = "".join(bpe_tokens)

        # Map every single character back to its raw byte integer (0-255)
        # Use self.byte_decoder to translate into raw numbers
        raw_bytes = [self.byte_decoder[char] for char in bpe_text]

        return bytearray(raw_bytes).decode('utf-8', errors='replace')

    def encode(self, text: str) -> List[int]:
        """
        Encodes a standard string into a list of token IDs.

        Uses a greedy longest-match approach. It first converts the
        string into mapped BBPE characters, then iteratively scans
        backwards from the end of the string to find the longest
        possible substring that exists in the vocabulary.

        Args:
            text (str): The raw input string to encode.

        Returns:
            List[int]: A list of integer IDs representing the text.
        """
        if not text:
            return []

        # Convert to bytes
        raw_bytes = text.encode("utf-8")

        # Map to BPE characters
        bpe_chars = [self.byte_encoder[b] for b in raw_bytes]
        bpe_string = "".join(bpe_chars)

        token_ids: List[int] = []
        i = 0

        # Greedy Longest-Match Tokenization
        # The outer loop moves through the string
        while i < len(bpe_string):
            match_found = False

            # The inner loop looks at the remaining str and shrinks backwards
            # until it finds a valid token in your vocabulary dictionary
            for j in range(len(bpe_string), i, -1):

                # Start by checking the entire rest of the string
                chunk = bpe_string[i:j]

                if chunk in self.token_to_id:
                    token_ids.append(self.token_to_id[chunk])
                    i = j  # Jump the index forward past the matched chunk
                    match_found = True
                    break

            # --- Fallback (Safety Net)
            # Because BBPE maps every single byte (0-255) to a token
            # single characters should always exist in the vocabulary
            # If a bug occurs, skip ahead
            if not match_found:
                i += 1

        return token_ids
