# System Entry & Architecture

The `src/` root module serves as the command centre and orchestrator for the entire constrained decoding pipeline. While the submodules (`parser`, `engine`, `formatter`) handle the specific low-level mechanics of parsing schemas or masking logits, this top-level module is responsible for the macro-execution. It acts as the pipeline manager: ingesting CLI arguments, managing hardware constraints via dynamic data chunking, executing the three-phase generation loop, and providing real-time pedagogical feedback via the terminal visualizer.

## Theoretical Concepts

### Pipeline Isolation (Divide and Conquer)

Small language models (sub-1B parameters) struggle to perform multiple complex reasoning tasks simultaneously. If asked to route a prompt *and* generate its complex arguments in a single step, they frequently hallucinate. By forcing the architecture into strictly isolated phases (Phase 1: Identify the target -> Phase 2: Extract the data -> Phase 3: Validate the format), the cognitive load on the LLM is drastically reduced, ensuring high deterministic accuracy on lightweight hardware.

### Batch Processing vs. Sequential Processing

Running prompts one-by-one leaves modern CPUs and GPUs idle, while attempting to process 10,000 prompts simultaneously will immediately exceed available VRAM. Batch processing strikes the mathematical balance: it groups data into manageable chunks, allowing the hardware to utilise highly optimised, parallelised matrix operations while strictly bounding maximum memory footprint.

### Dynamic Resource Allocation

In text generation, predicting tokens is computationally expensive. Not all prompts require the same resources; generating a single boolean value requires far fewer tokens than generating deeply nested arrays of strings. By dynamically calculating a strict `max_new_tokens` limit based on the specific shape of the targeted schema *before* generation begins, the orchestrator prevents computational waste and physically blocks the model from entering infinite hallucination loops.

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

### Batching (Macro)

The practice of grouping multiple prompts into a single payload before sending them to the LLM. In this pipeline, it is managed by a Python generator to protect system RAM and GPU VRAM from Out-Of-Memory (OOM) crashes.

### Caching (Macro)

Storing the results of slow, repetitive operations (like reading few-shot JSON files from the physical hard drive) in fast, temporary RAM (memory) so they only need to be executed once per batch run.

### Batch Size

The number of prompts processed simultaneously by the engine's matrix operations. 

### OOM (Out-Of-Memory)

A fatal hardware exception that occurs when an application attempts to allocate more VRAM/RAM than is physically available on the machine.

### Generator (`yield`)

A Python construct that returns a lazy iterator. Instead of loading an entire dataset into memory at once, it evaluates and returns one chunk at a time, keeping memory usage perfectly flat.

### Graceful Degradation

An architectural design principle ensuring that a localised failure (e.g., a single malformed JSON string) triggers a safe fallback rather than crashing the entire global process.

### Hallucination

In the context of constrained decoding, this refers to the LLM attempting to continuously generate infinite, repeating, or nonsensical tokens instead of cleanly closing the JSON structure.
