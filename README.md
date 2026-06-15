_This project has been created as part of the 42 curriculum by spacotto._

## Description

This project introduces **function calling** in **Large Language Models (LLMs)** by building a **self-contained system** that translates natural language prompts into structured function calls with explicitly typed arguments. While standard applications rely on prompt engineering (which is inherently non-deterministic and prone to syntax hallucinations), this architecture implements **constrained decoding** at the engine level by intercepting raw token probabilities (**logits**) during the autoregressive generation loop. By dynamically applying a mathematical mask to eliminate any token that violates the target schema, the engine guarantees **structurally valid JSON output** with **near-perfect reliability**. This approach achieves **deterministic compliance** even when deploying a **lightweight 0.5B parameter model** on resource-constrained environments.

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
- **Phase 2, Argument Extraction.** The engine dynamically bounds the generation length based on the target schema's shape. It traverses a SchemaTrie to mask out illegal structural characters (like misplaced brackets or quotes) while allowing the model to freely generate valid parameter values.
- **Phase 3, Post-Processing & Validation.** The raw generated strings are parsed into Python dictionaries. This phase acts as a final safety net, catching any JSONDecodeErrors, filtering out structural anomalies, and forcing absolute schema compliance by injecting safe fallback objects if an edge case bypasses the extraction limits.

>[!IMPORTANT]
>For more detailed documentation about the O(1) masking matrix, prefix caching, and the SchemaTrie implementation, see the [**Engine Documentation**](https://github.com/spacotto/CallMeMaybe/blob/main/src/engine/README.md).

## Design decisions

### Multi-Model Support & Resource Efficiency (SmolLM2-360M-Instruct)

## Performance analysis: Accuracy, Speed, and Reliability

## Challenges faced

### The Multi-Byte Fragmentation Crash

## Testing strategy

## Example usage

## Resources
