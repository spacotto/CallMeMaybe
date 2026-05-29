_This project has been created as part of the 42 curriculum by spacotto._

## Description

This project introduces function calling in Large Language Models (LLMs) by building a self-contained system that translates natural language prompts into structured function calls with explicitly typed arguments. While standard applications rely on prompt engineering (which is inherently non-deterministic and prone to syntax hallucinations), this architecture implements constrained decoding at the engine level by intercepting raw token probabilities (logits) during the autoregressive generation loop. By dynamically applying a mathematical mask to eliminate any token that violates the target schema, the engine guarantees structurally valid JSON output with near-perfect reliability. This approach achieves deterministic compliance even when deploying a lightweight 0.5B parameter model on resource-constrained environments, successfully bridging the gap between fluid human language and rigid, computer-executable operations.

## Instructions

## Algorithm explanation

## Design decisions

## Performance analysis: Accuracy, Speed, and Reliability

## Challenges faced

## Testing strategy

## Example usage

## Resources
