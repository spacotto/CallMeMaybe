# Tokenizer Module: Byte-Level Encoding & Decoding

This module implements a custom **Byte-Level Byte-Pair Encoding (BBPE)** tokenizer from scratch. It acts as the critical bridge between human-readable UTF-8 strings and the integer arrays (logits) required by the underlying Large Language Model. 

As a pedagogical resource, this module deliberately avoids relying on black-box libraries like Hugging Face's `transformers` for text processing. Instead, it exposes the raw mathematical and string-manipulation logic required to safely encode and decode text for modern LLMs.

## Theoretical Concepts

### Byte-Level BPE

Language Models (LMs) do not natively understand text, words, or grammar; they only understand math. To feed text into a neural network, it must be converted into integers. 

**Byte-Level BPE** drops down to the lowest possible level: **raw bytes (0-255)**. Because every piece of text (whether it is English, Korean, or an emoji) can be represented as a sequence of UTF-8 bytes, this tokenizer guarantees that it can process *any* string without ever encountering an unknown token.

### Encoding: The Greedy Lexer (`encode`)

Encoding is the process of chunking a continuous string into the largest possible sub-strings that exist in the model's vocabulary.

```text
=========================================================================
STAGE 1: THE RAW INPUT TEXT
=========================================================================
Text: "What is the"
                               |
                               v

=========================================================================
STAGE 2: UTF-8 BYTE EXTRACTION (text.encode('utf-8'))
Python's core string engine breaks the text down into raw binary memory,
producing an array of low-level bytes (0-255).
=========================================================================
Buffer:     [ 87, 104, 97, 116, 32, 105, 115, 32, 116, 104, 101 ]
               |               |               |
               v               v               v

=========================================================================
STAGE 3: BYTE-TO-UNICODE TRANSLATION (self.byte_encoder)
Every single raw byte integer is mapped to its specialized, printable 
BPE character counterpart. The space byte (32) becomes 'Ġ'.
=========================================================================
Raw Bytes:  [ [87,104,97,116], [32, 105, 115],  [32, 116, 104, 101] ]
               |                |                |
               v                v                v

=========================================================================
STAGE 4: GREEDY LONGEST-MATCH LEXING
Scans the BPE string left-to-right, looking ahead to slice out the 
largest possible chunks that exist in the vocabulary.
=========================================================================
BPE Tokens: [ "What",          "Ġis",           "Ġthe" ]
               |                |                |
               v                v                v

=========================================================================
STAGE 5: VOCABULARY MAPPING (self.token_to_id)
Queries the vocabulary dictionary to replace the string tokens with the 
final integer token IDs required by the neural network.
=========================================================================
Array:      [ 3838,            374,             279 ]
```

### Decoding: The Byte Buffer (`decode`)

Decoding reconstructs the string from neural network outputs.

```text
=========================================================================
STAGE 1: THE RAW LOGITS / IDs
=========================================================================
Array:      [ 3838,            374,             279 ]
              |                |                |
              v                v                v

=========================================================================
STAGE 2: BPE VOCABULARY LOOKUP (self.id_to_token)
Looks up each integer in the vocab dictionary. The 'Ġ' artifact represents 
a space so it doesn't crash on unprintable whitespace characters.
=========================================================================
BPE Tokens: [ "What",          "Ġis",           "Ġthe" ]
              |                |                |
              v                v                v

=========================================================================
STAGE 3: BYTE-LEVEL TRANSLATION (self.byte_decoder)
Maps every character (including 'Ġ') back to its raw ASCII/Unicode byte 
integer (0-255). 'Ġ' translates to 32.
=========================================================================
Raw Bytes:  [ [87,104,97,116], [32, 105, 115],  [32, 116, 104, 101] ]
              |                |                |
              v                v                v

=========================================================================
STAGE 4: THE BYTEARRAY BUFFER (bytearray(raw_bytes))
Flattens all individual lists into one continuous, low-level memory 
buffer of raw numbers.
=========================================================================
Buffer:     [ 87, 104, 97, 116, 32, 105, 115, 32, 116, 104, 101 ]
                               |
                               v

=========================================================================
STAGE 5: UTF-8 DECODING (.decode('utf-8'))
Python's C-engine reads the raw memory buffer and safely casts it back 
into human-readable text.
=========================================================================
Final Text: "What is the"
```

## Design Decisions

### Pedagogical Transparency

Built from scratch using standard Python libraries to expose the mechanics of text tokenisation, avoiding dependency on heavy external AI libraries.

### Greedy vs. Rules-Based Merging

The encoder utilises a greedy longest-match algorithm rather than parsing a strict `merges.txt` rule file. While mathematically sound and highly robust for constrained decoding, **the exact array of IDs generated may occasionally differ slightly from the official model tokenisers on complex words**, though the decoded output remains perfectly functionally equivalent.

### Low-Level Byte Buffering

Instead of decoding token-by-token, the decoder pushes all integers into a C-style bytearray. This allows Python's internal engine to safely stitch fragmented multi-byte characters together before rendering, avoiding string manipulation crashes.

## Challenges Solved
