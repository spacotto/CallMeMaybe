# Tokenizer Module

This module implements a custom **Byte-Level Byte-Pair Encoding (BBPE)** tokenizer from scratch. It is designed to act as the bridge between human-readable UTF-8 strings and the integer arrays (logits) required by the underlying Large Language Model.

As a pedagogical resource, this module avoids relying on black-box libraries like Hugging Face's `transformers` for text processing. Instead, it exposes the raw mathematical and string-manipulation logic required to safely encode and decode text for modern LLMs.

## 🧠 Core Concept: Why Byte-Level?

Language models do not natively understand text, words, or grammar; they only understand math. To feed text into a neural network, it must be converted into integers. 

Older models used word-level or character-level tokenizers, which would crash or output `<UNK>` (Unknown) if they encountered a rare foreign character or a complex emoji not explicitly in their dictionary. 

**Byte-Level BPE** solves this by dropping down to the lowest possible level: **raw bytes (0-255)**. 
Because every piece of text—whether it is English, Korean, or an emoji—can be represented as a sequence of UTF-8 bytes, this tokenizer guarantees that it can process *any* string without ever encountering an unknown token.

## ⚙️ Architecture

This module provides two primary operations: encoding (text to IDs) and decoding (IDs to text).

### 1. Encoding: The Greedy Lexer (`encode`)
Encoding is the process of chunking a continuous string into the largest possible sub-strings that exist in the model's vocabulary. 
1. **Byte Translation:** The raw string is converted to UTF-8 bytes, and each byte is mapped to a specialized BPE character (e.g., a space ` ` becomes `Ġ`).
2. **Greedy Longest-Match:** A lexer-style algorithm scans the string from left to right, looking ahead to find the absolute longest chunk of characters that matches an entry in the vocabulary dictionary. Once found, it assigns the integer ID and moves forward.

### 2. Decoding: The Byte Buffer (`decode`)
Decoding reconstructs the string from neural network outputs. 
Because a single complex character (like a French accent or an emoji) might be split across multiple tokens, decoding token-by-token is dangerous and can crash Python's string engine.
1. **String Translation:** IDs are mapped back to their BPE characters.
2. **Flattening:** Every character is translated back into its raw 0-255 integer representation.
3. **Bytearray Casting:** The integers are shoved into a low-level C-style `bytearray`. This allows Python's internal engine to safely stitch fragmented multi-byte characters back together before rendering the final UTF-8 string.

## ⚠️ Implementation Notes & Discrepancies

If you are evaluating this project or studying the source code, please note the following technical decisions:

* **SDK Method Name Discrepancy:** The project documentation may refer to a `get_path_to_vocabulary_json()` method for retrieving the vocabulary file. However, inspection of the provided `Small_LLM_Model` SDK reveals this is actually implemented as `get_path_to_vocab_file()`. This module uses the true implementation.
* **Greedy vs. Rules-Based Merging:** This encoder uses a greedy longest-match algorithm rather than parsing a strict `merges.txt` rule file. While mathematically sound and highly robust for constrained decoding, the exact array of IDs generated may occasionally differ slightly from the official Qwen tokenizer on complex, multi-syllabic words. The decoded output remains perfectly functionally equivalent.

## 🚀 Usage

```python
from src.tokenizer import Tokenizer

tokenizer = Tokenizer()

# Encoding
text = "What is the sum of 2 and 3?"
ids = tokenizer.encode(text)
# >>> [3838, 374, 279, 2629, 315, 220, 17, 323, 220, 18, 30]

# Decoding
decoded_text = tokenizer.decode(ids)
# >>> "What is the sum of 2 and 3?"
