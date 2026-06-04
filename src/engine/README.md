# Constrained Decoding Engine

This module contains the core generation loop and the mathematical state machine that enforces the Large Language Model (LLM) to output strictly valid JSON schemas.

## Architecture Overview

Instead of relying on probabilistic text generation, this engine **intercepts the neural network's raw `logits` output at every step of the autoregressive loop**. By applying **dynamic, vectorised boolean masks in NumPy**, the engine mathematically eliminates tokens that would violate the target JSON structure.

## Performance Optimisations (Caching & Batching)

To achieve high throughput on resource-constrained hardware, the engine implements **caching across multiple layers of the architecture to eliminate redundant computations**.

### 1. Vectorised State Caching (Precomputed Masks)

Instead of **iterating through the 150,000+ token vocabulary on every generation step** to check character validity, the engine **caches the entire validation state during initialisation**.
* **Implementation:** `struct_mask`, `wildcard_string_mask`, `safe_quote_mask`, and `banned_keyword_mask` are computed exactly once as NumPy arrays.
* **Impact:** Replaces millions of native Python string comparisons with instant, C-level bitwise operations (`mask &= self.struct_mask`).

### 2. Algorithmic State Caching (Trie Prefix Hoisting)
The engine caches the traversal path of the prefix tree when validating specific function names.
* **Implementation:** `name_trie.get_node()` evaluates the prefix string before the vocabulary loop begins.
* **Impact:** Prevents the Trie from re-traversing the root sequence from scratch for every single token in the vocabulary, caching its exact location in the memory graph.

### 3. I/O and Memory Caching
The engine intercepts **expensive string manipulations** and **tokeniser decoding operations**.
* **Implementation:** The `clean_vocab` list caches the results of byte-pair string replacements (`replace("Ġ", " ")`). The `self.fast_decode` method utilises Python's `@lru_cache`.
* **Impact:** Prevents the system from executing continuous string formatting and complex dictionary lookups during the hot execution loop.

### 4. Compilation Caching (Regex State Machines)
* **Implementation:** `re.compile()` is used at the class level for all structural extraction patterns.
* **Impact:** Bypasses Python's internal dictionary lookups, preventing the engine from rebuilding the regex C-level state machine on every prompt validation.

## Deterministic Fast-Forwarding

To further reduce GPU matrix multiplication overhead, the engine employs a fast-forwarding technique. When a **deterministic structural** boundary is reached (e.g., closing the quote on a function name), the engine artificially injects the next required JSON tokens (e.g., `, "parameters": {`) directly into the sequence buffer. This entirely skips the forward-pass execution for boilerplate syntax, significantly boosting generation speed.
