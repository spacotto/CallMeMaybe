_This project has been created as part of the 42 curriculum by spacotto._

## Description

This project introduces **function calling** in **Large Language Models (LLMs)** by building a **self-contained system** that translates natural language prompts into structured function calls with explicitly typed arguments. While standard applications rely on prompt engineering (which is inherently non-deterministic and prone to syntax hallucinations), this architecture implements **constrained decoding** at the engine level by intercepting raw token probabilities (**logits**) during the autoregressive generation loop. By dynamically applying a mathematical mask to eliminate any token that violates the target schema, the engine guarantees **structurally valid JSON output** with **near-perfect reliability**. This approach achieves **deterministic compliance** even when deploying a **lightweight 0.5B parameter model** in resource-constrained environments.

## Instructions

The project uses uv for fast dependency management and a Makefile for execution.

Standard initialisation: 

```bash
make install
```

Run Pipeline: 

```bash
make run
```

Visualise the engine (runs batch size 1 with live state-machine visualisation):

```bash
make visual
```

Run the pipeline with the alternative model:

```bash
make run-alt ALT="HuggingFaceTB/SmolLM2-360M-Instruct"
```

Display the list of the available commands:

```bash
make help
```

## Algorithm explanation

The constrained decoding pipeline operates in three phases to **guarantee deterministic outputs**:
- **Phase 1, Zero-Shot Classification.** The engine uses an O(1) masking cache to force the model to evaluate the prompt and output a valid function name from the provided JSON schemas.
- **Phase 2, Argument Extraction.** The engine dynamically bounds the generation length based on the target schema's shape. It traverses a SchemaTrie to mask out illegal structural characters (like misplaced brackets or quotes) while allowing the model to generate valid parameter values freely.
- **Phase 3, Post-Processing & Validation.** The raw generated strings are parsed into Python dictionaries. This phase serves as a final safety net, catching any JSONDecodeErrors, filtering out structural anomalies, and enforcing absolute schema compliance by injecting safe fallback objects when an edge case exceeds the extraction limits.

>[!IMPORTANT]
>For more detailed documentation, see the [**Engine Documentation**](https://github.com/spacotto/CallMeMaybe/blob/main/src/engine/README.md).

## Design decisions

### System Entry & Architecture (`src/__main__.py`)

* **Dynamic Model Initialisation:** Model weights are loaded via the `LLM_MODEL_NAME` environment variable with a default fallback, rather than being hardcoded into the pipeline.
* **Generator-Based Batching:** The `chunk_data` function uses a Python generator (`yield`) to slice the dataset into batches of 32 (or 1 in verbose mode).
* **Dynamic Token Limits (`calculate_prompt_limit`):** The engine inspects the target JSON schema (counting parameters, checking for string types, and identifying nesting) to calculate the strict `minimum max_new_tokens` required for that specific prompt.
* **3-Phase Execution Architecture:** The loop strictly isolates generation into **Classification** (forcing the function name), **Extraction** (generating the arguments), and **Post-Processing** (validating the JSON).
* **Fail-Safe Injection:** If Phase 3 catches a critical parsing error for a single prompt, it injects a compliant fallback object (`{"parameters": {}}`) rather than aborting.

### Data Ingestion (`src/parser/`)

* **Strict Type Validation:** The module leverages Pydantic (`BaseModel`, `Field`) to map and strictly validate the input JSON schemas against hierarchical data models, including `FunctionDefinition`, `FunctionParameters`, and `FunctionProperty`.  
* **Graceful Degradation**: During schema loading, if an individual function definition fails Pydantic validation, the parser logs the `ValidationError` and explicitly skips the malformed item rather than crashing the entire pipeline.
* **Downstream Compatibility:** After ensuring structural integrity, the validated Pydantic objects are converted back into standard Python dictionaries using `model_dump(exclude_none=True)` to remain compatible with the core masking engine.
* **Static Nesting Detection:** The `SchemaParser` includes a static `is_nested` method that iterates through function parameters to explicitly detect if any parameter is categorised as an "object".

### Context Preparation (`src/formatter/`)

* **Template Agnosticism:** The module utilises an `Enum` (`ModelFormat`) to dynamically construct prompts using either standard `CHATML` or `INSTRUCT` (Llama) syntax, decoupling the pipeline from a single model's architecture.
* **Phase-Specific Prompting:** The logic is split between Phase 1 (`build_classification_prompt`) and Phase 2 (`build_extraction_prompt`). Phase 1 provides a compressed, **zero-shot schema** summary, while Phase 2 injects the full schema along with **few-shot** examples.
* **O(1) I/O Caching:** The `load_examples` method implements an in-memory dictionary cache (`_examples_cache`). Once a function's few-shot JSON file is read from the disk, it is cached to serve all future batch requests instantly.
* **Optimised Assembly:** Large multi-turn prompt strings are assembled using list appending and `"".join(parts)` rather than standard `+=` string concatenation, optimising memory allocation during batch runs.
* **Dynamic Few-Shot Grounding:** The module automatically scans the `/few_shot/` directory for JSON files matching the target function's name (e.g., `fn_substitute_string_with_regex.json`) and weaves them natively into the model's message boundaries.

### Low-Level Mechanics (`src/tokenizer/`)

### Constrained Decoding, Finalisation & Output (`src/engine/`)

### Visualisation (`src/visualizer/`)

## Performance analysis: Accuracy, Speed, and Reliability

By replacing traditional "generate-and-pray" post-processing with pre-generation logit manipulation, the pipeline achieves:
- 100% JSON Structural Compliance: Guaranteed by the masking cache.
- High Throughput: O(1) lookups using an inverted index (`string_to_ids`) prevent the engine from slowing down during matrix generation.
- Memory Safety: Dynamic data chunking (`BATCH_SIZE`) prevents Out-Of-Memory exceptions during large processing runs.

## Challenges faced

### System Entry & Architecture (`src/__main__.py`)

* **Out-Of-Memory (OOM) GPU Crashes:** By using the `chunk_data` generator, the engine prevents memory overload when processing thousands of prompts on resource-constrained hardware.
* **Runaway Hallucinations:** Small models (like 0.5B parameters) are prone to endless generation loops. `calculate_prompt_limit` solves this by aggressively cutting off the LLM exactly when the schema should be complete, saving inference time and preventing garbage text.
* **Pipeline Fragility:** The fail-safe injection in the post-processing loop prevents a single LLM hallucination or `JSONDecodeError` from crashing a 40-minute batch job, ensuring the final output file is always safely written.
* **Hardware Lock-in:** The environment variable design allows the Makefile to swap in lighter models (like SmolLM) on the fly without altering the source code, bypassing hardware limitations.

### Data Ingestion (`src/parser/`)

* **Malformed Schema Crashes:** By enforcing structural rules upfront and catching both `json.JSONDecodeError` and Pydantic exceptions, the module prevents the engine from halting due to syntax errors, missing root lists, or invalid structures in the user-provided definitions.
* **Missing Data Handling:** The use of `Optional` fields and default dictionary/list factories in the data models ensures the downstream pipeline always receives safe, predictable structures, bypassing `NoneType` errors.
* **Output Consistency:** The module introduces the `FunctionCallResult` model, which acts as a structural guardrail to guarantee that every generated item matches the project's mandatory `prompt`, `name`, and `parameters` output format.

### Context Preparation (`src/formatter/`)

* **Syntax Hallucinations:** By strictly enforcing the correct **control tokens** (like `<|im_start|>` vs `[INST]`) based on the model type, it prevents the LLM from becoming confused and generating out-of-distribution garbage text.
* **Disk I/O Bottlenecks:** The memory caching mechanism prevents the engine from repeatedly opening and reading the same few-shot JSON files thousands of times during a large batch process, drastically reducing execution time.
* **Context Window Bloat:** Passing the full parameter schema for every available function during Phase 1 classification would overwhelm a small model's context window. The formatter solves this by only feeding names and descriptions initially, reserving the dense token payload for Phase 2.
* **Zero-Shot Weakness in Small Models:** Sub-1B models struggle heavily with complex logical extractions (like mapping natural language to regex strings). Dynamically injecting function-specific examples acts as a critical anchor, dramatically improving the model's accuracy before the logit masking even begins.

### Low-Level Mechanics (`src/tokenizer/`)

### Constrained Decoding, Finalisation & Output (`src/engine/`)

### Visualisation (`src/visualizer/`)

## Testing strategy

The architecture is verified through a comprehensive, custom-built test suite (`make test`) that validates the project across multiple vectors:
- **Input Validation:** Ensures graceful failure on missing files or malformed definitions.
- **Output Compliance:** Strictly checks the generated JSON against the Pydantic-style schema models.
- **Nested Parameters:** Tests the engine's ability to retain state tracking deep within nested object dictionaries.
- **Edge Cases:** Throws adversarial inputs to verify the resilience of the PostProcessor and fallback mechanics.

## Example usage

Run the engine against a custom schema and input file:

```bash
uv run python -m src \
  --functions_definition data/input/my_schema.json \
  --input data/input/my_prompts.json \
  --output data/output/results.json \
  --verbose
```

## Resources
- [Fast, High-Fidelity LLM Decoding with Regex Constraints](https://huggingface.co/blog/vivien/llm-decoding-with-regex-constraints) 
- [Function Calling](https://huggingface.co/docs/hugs/en/guides/function-calling)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Large Language Models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model)
- [Logit](https://en.wikipedia.org/wiki/Logit)
- [Understanding Byte-Pair Encoding (BPE)](https://huggingface.co/learn/llm-course/chapter6/5)
- [Understand tokens](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens)
- [uv Package Manager](https://docs.astral.sh/uv/)
- [What are Logits in LLMs?](https://docs.lm-kit.com/lm-kit-net/guides/glossary/logits.html)

### AI Usage

* **Subject Research:** Exploring and understanding the underlying mechanics, such as autoregressive generation, Byte-Level BPE tokenisation, logits manipulation, etc.
* **Debugging:** Assisting in diagnosing low-level errors, such as CUDA device-side assertions and byte-level BPE fragmentation issues.
* **Code Quality & Strict Typing:** Identifying and resolving formatting constraints and type-hinting errors (Flake8, Mypy) to ensure the codebase met required standards.
* **Documentation:** Drafting and structuring the README files and Python docstrings.
