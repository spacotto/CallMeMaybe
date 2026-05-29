_This project has been created as part of the 42 curriculum by spacotto._

## Description

This project introduces function calling in Large Language Models (LLMs) by building a self-contained system that translates natural language prompts into structured function calls with explicitly typed arguments. While standard applications rely on prompt engineering (which is inherently non-deterministic and prone to syntax hallucinations), this architecture implements constrained decoding at the engine level by intercepting raw token probabilities (logits) during the autoregressive generation loop. By dynamically applying a mathematical mask to eliminate any token that violates the target schema, the engine guarantees structurally valid JSON output with near-perfect reliability. This approach achieves deterministic compliance even when deploying a lightweight 0.5B parameter model on resource-constrained environments, successfully bridging the gap between fluid human language and rigid, computer-executable operations.

## Instructions

## Algorithm explanation

## Design decisions

### [Tokenizer](https://github.com/spacotto/CallMeMaybe/blob/main/src/tokenizer/README.md): From-Scratch BBPE Engine

Instead of relying on black-box abstractions like Hugging Face's `transformers` library, this project implements a custom **Byte-Level Byte-Pair Encoding (BBPE)** tokenizer from scratch. 
* **Greedy Lexing Algorithm:** For translating text to neural network integers (encoding), the system uses a longest-match greedy lexer. This avoids the heavy computational overhead of parsing a strict `merges.txt` ruleset while maintaining near-perfect structural alignment for constrained decoding tasks.
* **Low-Level Byte Buffering:** For decoding, the system bypasses Python's high-level string handlers. It maps BPE characters directly to their raw 0-255 integer values and loads them into a contiguous `bytearray`. This allows the underlying C-engine to safely reconstruct UTF-8 strings before rendering them to the console.

## Performance analysis: Accuracy, Speed, and Reliability

## Challenges faced

### The Multi-Byte Fragmentation Crash

>[!WARNING]
>**The Problem:** Language models do not respect character boundaries when outputting tokens. A complex multi-byte character (like a French accent `é` or a system emoji) is often split across two or three separate neural network outputs. Attempting to decode these tokens one by one causes fatal runtime crashes, as half a UTF-8 character is technically invalid memory.

>[!TIP]
>**The Solution:** By dropping the outputs into the intermediate `bytearray` buffer mentioned in the design decisions, the engine safely collects the fragmented binary data over multiple autoregressive loops, only casting to a readable string once the memory block is complete and valid.

## Testing strategy

## Example usage

## Resources
