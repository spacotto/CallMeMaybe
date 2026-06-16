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

* [System Entry & Architecture](https://github.com/spacotto/CallMeMaybe/blob/main/src/README.md#design-decisions) (`src/__main__.py`)
* [Data Ingestion](https://github.com/spacotto/CallMeMaybe/blob/main/src/parser/README.md#design-decisions) (`src/parser/`)
* [Context Preparation](https://github.com/spacotto/CallMeMaybe/blob/main/src/formatter/README.md#design-decisions) (`src/formatter/`)
* [Low-Level Mechanics](https://github.com/spacotto/CallMeMaybe/blob/main/src/tokenizer/README.md#design-decisions) (`src/tokenizer/`)
* [Constrained Decoding, Finalisation & Output](https://github.com/spacotto/CallMeMaybe/blob/main/src/engine/README.md#design-decisions) (`src/engine/`)
* [Visualisation](https://github.com/spacotto/CallMeMaybe/blob/main/src/visualizer/README.md#design-decisions) (`src/visualizer/`)

## Performance analysis: Accuracy, Speed, and Reliability

By replacing traditional "generate-and-pray" post-processing with pre-generation logit manipulation, the pipeline achieves:
- 100% JSON Structural Compliance: Guaranteed by the masking cache.
- High Throughput: O(1) lookups using an inverted index (`string_to_ids`) prevent the engine from slowing down during matrix generation.
- Memory Safety: Dynamic data chunking (`BATCH_SIZE`) prevents Out-Of-Memory exceptions during large processing runs.

## Challenges faced

* [System Entry & Architecture](https://github.com/spacotto/CallMeMaybe/blob/main/src/README.md#challenges-solved) (`src/__main__.py`)
* [Data Ingestion](https://github.com/spacotto/CallMeMaybe/blob/main/src/parser/README.md#challenges-solved) (`src/parser/`)
* [Context Preparation](https://github.com/spacotto/CallMeMaybe/blob/main/src/formatter/README.md#challenges-solved) (`src/formatter/`)
* [Low-Level Mechanics](https://github.com/spacotto/CallMeMaybe/blob/main/src/tokenizer/README.md#challenges-solved) (`src/tokenizer/`)
* [Constrained Decoding, Finalisation & Output](https://github.com/spacotto/CallMeMaybe/blob/main/src/engine/README.md#design-decisions) (`src/engine/`)
* [Visualisation](https://github.com/spacotto/CallMeMaybe/blob/main/src/visualizer/README.md#challenges-solved) (`src/visualizer/`)

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
