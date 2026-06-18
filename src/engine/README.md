# Engine Module: Constrained Decoding Mechanics

The `engine` module is the core of the function-calling pipeline. It is responsible for intercepting the Large Language Model's generation process, evaluating the mathematical probabilities of the next potential tokens, and ruthlessly eliminating any outputs that violate the required JSON schema. 

Rather than relying on the LLM to "understand" JSON through prompt engineering, this engine forces structural compliance at the lowest level. It operates in three phases:
1. **Zero-Shot Routing (`classifier.py`):** Forces the model to select the correct function name.
2. **Context-Aware Extraction (`extractor.py`):** Drives the complex, stateful generation of JSON arguments.
3. **Validation & Fallback (`postprocessor.py`):** Enforces strict Pydantic rules and handles type coercion.

## Execution Flow: Step-by-Step Algorithm

The constrained decoding algorithm operates as a rigorous pipeline that intercepts the LLM's text generation process at the mathematical level. Here is the exact lifecycle of a prompt passing through the engine.

### Phase 0: Pre-computation (The Setup)

Before any text is generated, the engine pre-computes the rules of the schema to avoid heavy calculations during the live generation loop.
1. **Trie Construction:** The allowed function names are inserted into a custom `SchemaTrie`. This Prefix Tree caches all valid string continuations in memory, reducing complex string validation to an O(1) set lookup.
2. **Vocabulary Masking:** The `SchemaExtractor` loops through the LLM's entire vocabulary (over 150,000 tokens) exactly once upon initialisation. It creates static boolean arrays (masks) for structural characters (`struct_mask`), safe wildcard strings (`wildcard_string_mask`), and specific data types (`type_masks`).

### Phase 1: Zero-Shot Routing (`classifier.py`)

The system must first determine *which* function the user wants to call.
1. **Formatting:** The user's prompt and a compressed summary of the available schemas are assembled into the model's native template (e.g., ChatML).
2. **Tokenization:** The text is encoded into integer IDs using the custom Byte-Level BPE tokeniser.
3. **The Masking Loop:** The model predicts the next token by outputting an array of unnormalized probabilities (logits). The engine checks the `SchemaTrie` to see which characters are legally allowed to follow the current prefix. 
4. **Suppression:** The engine applies a boolean mask to the logits. Any token ID that does not form a valid function name is overwritten with negative infinity (`-np.inf`). The model selects the most probable *allowed* token, and this loop repeats until a closing quote is generated.

### Phase 2: Argument Extraction (`extractor.py`)

Once the target function is known, the engine extracts the specific arguments in a highly controlled, batched loop.
1. **State Tracking:** The `JSONParserState` machine acts as the engine's eyes. Every time a new character is generated, this state machine updates its context (e.g., shifting from `EXPECT_KEY` to `EXPECT_COLON` or `IN_STRING_VALUE`).
2. **Dynamic Mask Assembly:** Based on the current state and the expected schema type, the engine combines the pre-computed vocabulary masks using fast bitwise operations (e.g., `mask &= self.type_masks["number"]`). 
3. **Batched Logit Masking:** The raw logits for all prompts in the batch are collected into a 2D NumPy matrix (`logits_matrix = np.stack(batch_logits)`). The dynamic masks are applied simultaneously to the entire batch, setting all illegal token probabilities to `-np.inf`.
4. **Greedy Selection & Decoding:** The engine uses `np.argmax` to select the mathematically highest allowed token ID for each prompt. The `Tokenizer` decodes these IDs back into text, the string is appended, and the cycle repeats until the JSON structure is fully closed.

### Phase 3: Post-Processing & Degradation (`postprocessor.py`)

Because constrained decoding guarantees structural syntax but can occasionally miss edge-case type coercions (like generating `"3"` instead of `3`), the final phase acts as a safety net.
1. **Parsing:** The generated raw string is parsed into a Python dictionary, catching any impossible `JSONDecodeError`s and substituting a safe fallback.
2. **Type Coercion:** The engine cross-references the dictionary against the original schema and forces variables into their correct types (e.g., casting floats to integers).
3. **Pydantic Enforcement:** The dictionary is pushed through the `FunctionCallResult` data model. If it passes, the perfect JSON is saved; if it fails, a graceful fallback is injected to ensure the pipeline never crashes.

## Theoretical Concepts

### Autoregressive Generation & Logits

LLMs generate text one token at a time. For every step, the model outputs a massive array of **raw, unnormalised prediction scores** called **logits** (one score for every token in its vocabulary). Normally, these logits pass through a Softmax function to become probabilities, and the highest probability wins.

### Logit Masking (Constrained Decoding)

This engine **intercepts the logits** *before* they are converted into probabilities. If a token violates our JSON schema (e.g., the model tries to output a `}` when the schema requires a `"name"`), the engine applies a **Masking Matrix**. It **overwrites the invalid token's logit with `$-\infty$`**. When Softmax is applied, `$e^{-\infty}$` becomes exactly `0`. The LLM is mathematically blocked from generating illegal syntax.

### CFG State Machines

To know exactly which characters are legal at any given millisecond, the `extractor` uses an incremental **Context-Free Grammar (CFG)** state machine. It updates its internal context (e.g., `EXPECT_KEY`, `IN_STRING_VALUE`) character-by-character, mapping the current JSON depth to the permitted token masks.

### The Constrained Generation Cycle (Tokenisation Integration)

Constrained decoding is an endless translation loop between human text and machine mathematics. The engine relies on the Tokenizer to bridge this gap at every single step of generation:

1. **State Tracking (Text):** The engine tracks the current JSON prefix (e.g., `{"name": "`).
2. **Rule Evaluation (Text):** The CFG state machine or Trie determines that the next valid characters can only be alphabetical or a closing quote `"`.
3. **Encoding Translation (Math):** The engine queries the Tokenizer's vocabulary: *"Which integer Token IDs contain only these allowed characters?"*
4. **Logit Masking (Math):** The engine takes the LLM's raw output array and sets the probabilities of all unapproved Token IDs to $-\infty$.
5. **Prediction (Math):** The model selects the most probable allowed integer ID (e.g., `452`).
6. **Decoding Translation (Text):** The Tokenizer decodes `452` back into a text string (e.g., `fn_`).
7. **Loop:** The new text is appended to the prefix, and the cycle repeats until the JSON is closed.

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


### Batching (Micro)

Stacking the raw logits of multiple parallel prompts into a single 2D NumPy array. This allows the CPU to apply token masks to the entire batch simultaneously using highly optimized, vectorised C-backend math.

### Caching (Memoisation)

Storing the results of heavy algorithmic computations (like evaluating 150,000 tokens against a JSON prefix). If the engine encounters the same string prefix again, it instantly retrieves the pre-computed boolean mask in O(1) time rather than recalculating it.
