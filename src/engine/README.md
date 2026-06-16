# Engine Module: Constrained Decoding Mechanics

The `engine` module is the core of the function-calling pipeline. It is responsible for intercepting the Large Language Model's generation process, evaluating the mathematical probabilities of the next potential tokens, and ruthlessly eliminating any outputs that violate the required JSON schema. 

Rather than relying on the LLM to "understand" JSON through prompt engineering, this engine forces structural compliance at the lowest level. It operates in three phases:
1. **Zero-Shot Routing (`classifier.py`):** Forces the model to select the correct function name.
2. **Context-Aware Extraction (`extractor.py`):** Drives the complex, stateful generation of JSON arguments.
3. **Validation & Fallback (`postprocessor.py`):** Enforces strict Pydantic rules and handles type coercion.

## Theoretical Concepts

### Autoregressive Generation & Logits

LLMs generate text one token at a time. For every step, the model outputs a massive array of **raw, unnormalised prediction scores** called **logits** (one score for every token in its vocabulary). Normally, these logits pass through a Softmax function to become probabilities, and the highest probability wins.

### Logit Masking (Constrained Decoding)

This engine **intercepts the logits** *before* they are converted into probabilities. If a token violates our JSON schema (e.g., the model tries to output a `}` when the schema requires a `"name"`), the engine applies a **Masking Matrix**. It **overwrites the invalid token's logit with `$-\infty$`**. When Softmax is applied, `$e^{-\infty}$` becomes exactly `0`. The LLM is mathematically blocked from generating illegal syntax.

### CFG State Machines

To know exactly which characters are legal at any given millisecond, the `extractor` uses an incremental **Context-Free Grammar (CFG)** state machine. It updates its internal context (e.g., `EXPECT_KEY`, `IN_STRING_VALUE`) character-by-character, mapping the current JSON depth to the permitted token masks.

## Design Decisions

### Matrix Parallelisation (Batching)

Instead of looping through prompts individually, the engine stacks the logits for an entire batch into a 2D NumPy matrix. This allows the CPU to apply boolean token masks to 32 parallel prompts simultaneously, massively increasing throughput.

### O(1) Inverted Index (`string_to_ids`)

Scanning a 150,000+ token vocabulary array every generation step to find matching IDs would cripple performance. The classifier pre-computes an inverted index upon initialisation, mapping raw string fragments directly to lists of token IDs for instant retrieval.

### Lazy Mask Caching

Computing a boolean mask for a specific string prefix is computationally expensive. The engine memoises these results (`prefix_mask_cache`, `key_mask_cache`). If multiple prompts (or subsequent generation steps) reach the same prefix, the mask is retrieved instantly in O(1) time.

### Structural Pre-computation

The extractor evaluates the vocabulary once upon startup, creating foundational boolean arrays for structural characters (`struct_mask`), safe strings (`wildcard_string_mask`), and data types (`type_masks`). During generation, it applies these via fast bitwise operations (`&`, `|`) rather than string parsing.

### The SchemaTrie

A custom prefix tree (`trie.py`) maps out all valid JSON keys and function names. It pre-computes all valid suffix and prefix paths during initialisation, reducing multi-token validation to a simple Python `set()` lookup.

## Challenges Solved

### Runaway Hallucinations

Small models (like 0.5B parameters) are highly prone to endless generation loops. By enforcing strict dynamic token limits and masking out unclosed quotes or brackets, the engine aggressively cuts off the LLM exactly when the schema is complete.

### CPU Inference Bottlenecks

Naive constrained decoding algorithms evaluate regex rules against every vocabulary token for every single generation step, slowing generation to a crawl. The combination of inverted indexing, bitwise masking arrays, and O(1) Trie caching ensures the engine runs exceptionally fast, even on standard laptop CPUs.

### Out-Of-Memory (OOM) Errors

The batching architecture combined with dynamic chunking limits the VRAM and RAM footprint to a safe, constant size, preventing the hardware crashes common when processing massive datasets.

### Catastrophic Syntax Failures

Even with constraints, models occasionally attempt invalid type coercions. The 3-Phase architecture solves this by delegating the final safety net to the `postprocessor`, which catches `JSONDecodeError`s and forces compliant fallback objects, ensuring a single bad generation never crashes a 40-minute batch job.

## Glossary

### Logits

The raw, unnormalised prediction scores output by the LLM for every token in its vocabulary prior to Softmax application.

### Masking Matrix

A NumPy boolean array applied to the logits to suppress invalid tokens by setting their probabilities to zero.

### Softmax

A mathematical function that converts a vector of raw logits into a normalised probability distribution (values between 0 and 1 that sum to 1).

### Autoregressive

A generation process where the model predicts the next element in a sequence based on all previously generated elements.

### Prefix

The exact string of text the model has generated up to the current millisecond.

### O(1) Time Complexity

An operation that takes a constant amount of time to execute, regardless of the size of the dataset (e.g., a direct dictionary lookup).
