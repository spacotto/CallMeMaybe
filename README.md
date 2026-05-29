_This project has been created as part of the 42 curriculum by spacotto._

## Description

This project introduces function calling in Large Language Models (LLMs) by building a self-contained system that translates natural language prompts into structured function calls with explicitly typed arguments. While standard applications rely on prompt engineering (which is inherently non-deterministic and prone to syntax hallucinations), this architecture implements constrained decoding at the engine level by intercepting raw token probabilities (logits) during the autoregressive generation loop. By dynamically applying a mathematical mask to eliminate any token that violates the target schema, the engine guarantees structurally valid JSON output with near-perfect reliability. This approach achieves deterministic compliance even when deploying a lightweight 0.5B parameter model on resource-constrained environments, successfully bridging the gap between fluid human language and rigid, computer-executable operations.

## Instructions

## Algorithm explanation

## Design decisions

### Multi-Model Support & Resource Efficiency (SmolLM2-360M-Instruct)

To fulfil the multi-model capability criteria and prove that the core system architecture is entirely decoupled from specific text abstractions, **SmolLM2-360M-Instruct** was chosen as the secondary evaluation model.
* **Tokenizer Architecture Synergy:** SmolLM2 utilises the **same Byte-Level BPE tokenizer architecture as the primary Qwen model**, relying on identical GPT-2 style byte-to-unicode character maps (such as the `Ġ` space artefact). This architectural alignment validates the custom, low-level `bytearray` buffering mechanism across distinct models without duplicating the binary processing layer.
* **Decoupled Prompt Verification:** While sharing a tokenizer footprint, SmolLM2 uses a uniquely tuned instruction configuration. Implementing this model serves as a direct proof of concept that the `Formatter` class successfully **abstracts away chat template strings**, presenting a clean, unified prefix stream to the core execution engine.
* **Extreme Hardware Optimisation:** Operating at an ultra-lightweight 360 million parameters, this model **minimises local RAM and CPU usage**, aligning perfectly with the constraint requirements for **fast, reproducible evaluation loops** on restricted campus environments.

---

### [Tokenizer](https://github.com/spacotto/CallMeMaybe/blob/main/src/tokenizer/README.md): From-Scratch BBPE Engine

Instead of relying on black-box abstractions like Hugging Face's `transformers` library, this project implements a custom **Byte-Level Byte-Pair Encoding (BBPE)** tokenizer from scratch. 
* **Greedy Lexing Algorithm:** For translating text to neural network integers (encoding), the system uses a longest-match greedy lexer. This avoids the heavy computational overhead of parsing a strict `merges.txt` ruleset while maintaining near-perfect structural alignment for constrained decoding tasks.
* **Low-Level Byte Buffering:** For decoding, the system bypasses Python's high-level string handlers. It maps BPE characters directly to their raw 0-255 integer values and loads them into a contiguous `bytearray`. This allows the underlying C-engine to safely reconstruct UTF-8 strings before rendering them to the console.

---

### Formatter: Decoupled Multi-Model Context Preparation

To **prevent the core decoding engine from becoming tightly coupled to specific language model sub-dialects**, the presentation layer is isolated entirely inside the `Formatter` class.
* **The Context Prefilling Strategy:** Causal language models are inherently trained to generate natural conversational preambles (e.g., *"Sure! Here is the structured data you requested:"*). This breaks deterministic text parsing. The formatter solves this by **hard-injecting the exact terminal sequence of the assistant token** (e.g., `<|im_start|>assistant\n`) **at the end of the prompt**. This pins the model's generation cursor precisely where the JSON syntax must begin, **eliminating conversational noise** at zero computational cost.
* **Agnostic Token Boundary Encapsulation:** By parameterising template engines via explicit formatting enums (`ModelFormat.CHATML` and `ModelFormat.INSTRUCT`), the system ensures that adding or testing an entirely different target model family requires **zero structural alterations to the execution loop or the tokenizer logic**.

---

### Engine: Zero-Dependency Argmax Evaluation
Per the strict pedagogical constraints of the subject, external tensor libraries (such as `torch` or `numpy`) are forbidden. Rather than offloading the logit sorting to a C++ backend, the constrained decoding loop calculates the `argmax` over the ~150,000-token vocabulary using Python standard library routines (`max(range(), key=...)`). This guarantees the project remains entirely self-contained while still executing the masking loop efficiently on standard CPU hardware.

## Performance analysis: Accuracy, Speed, and Reliability

## Challenges faced

### The Multi-Byte Fragmentation Crash

>[!WARNING]
>**The Problem:** Language models do not respect character boundaries when outputting tokens. A complex multi-byte character (like a French accent `é` or a system emoji) is often split across two or three separate neural network outputs. Attempting to decode these tokens one by one causes fatal runtime crashes, as half a UTF-8 character is technically invalid memory.

>[!TIP]
>**The Solution:** By dropping the outputs into the intermediate `bytearray` buffer mentioned in the design decisions, the engine safely collects the fragmented binary data over multiple autoregressive loops, only casting to a readable string once the memory block is complete and valid.

---

### Conversational Padding and Syntax Deviations

>[!WARNING]
>**The Problem:** Even with clear system instructions demanding raw JSON output, models frequently **loop into markdown blocks** (````json ... ````) or **append polite introductory text**. In production or automated environments, this leads to immediate downstream runtime crashes during `json.loads()` verification phases.

>[!TIP]
>**The Solution:** By combining the **prompt-prefilling strategy** with the **strict logit tracking of the constrained decoding loop**, the model is trapped inside a **structural state machine**. Because the prompt **ends exactly at the point of assistant execution** and the engine actively **blocks non-JSON tokens from being predicted**, the model is physically **prevented from outputting markdown wrappers or introductory prose**.

## Testing strategy

## Example usage

## Resources
