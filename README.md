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

## Algorithm explanation

## Design decisions

### Multi-Model Support & Resource Efficiency (SmolLM2-360M-Instruct)

## Performance analysis: Accuracy, Speed, and Reliability

## Challenges faced

### The Multi-Byte Fragmentation Crash

## Testing strategy

## Example usage

## Resources
