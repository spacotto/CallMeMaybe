# System Entry & Architecture

## Theoretical Concepts

## Design Decisions

### Dynamic Model Initialisation

Model weights are loaded via the `LLM_MODEL_NAME` environment variable with a default fallback, rather than being hardcoded into the pipeline.

### Generator-Based Batching

The `chunk_data` function uses a Python generator (`yield`) to slice the dataset into batches of 32 (or 1 in verbose mode).

### Dynamic Token Limits (`calculate_prompt_limit`)

The engine inspects the target JSON schema (counting parameters, checking for string types, and identifying nesting) to calculate the strict `minimum max_new_tokens` required for that specific prompt.

### 3-Phase Execution Architecture

The loop strictly isolates generation into **Classification** (forcing the function name), **Extraction** (generating the arguments), and **Post-Processing** (validating the JSON).

### Fail-Safe Injection

If Phase 3 catches a critical parsing error for a single prompt, it injects a compliant fallback object (`{"parameters": {}}`) rather than aborting.

## Challenges Solved

### Out-Of-Memory (OOM) GPU Crashes

By using the `chunk_data` generator, the engine prevents memory overload when processing thousands of prompts on resource-constrained hardware.

### Runaway Hallucinations

Small models (like 0.5B parameters) are prone to endless generation loops. `calculate_prompt_limit` solves this by aggressively cutting off the LLM exactly when the schema should be complete, saving inference time and preventing garbage text.

### Pipeline Fragility

The fail-safe injection in the post-processing loop prevents a single LLM hallucination or `JSONDecodeError` from crashing a 40-minute batch job, ensuring the final output file is always safely written.

### Hardware Lock-in

The environment variable design allows the Makefile to swap in lighter models (like SmolLM) on the fly without altering the source code, bypassing hardware limitations.

## Glossary
