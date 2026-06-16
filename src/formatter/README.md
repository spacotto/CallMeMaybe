# Context Preparation: The Formatter

## Theoretical Concepts

## Glossary

## Design Decisions

### Template Agnosticism

The module utilises an `Enum` (`ModelFormat`) to dynamically construct prompts using either standard `CHATML` or `INSTRUCT` (Llama) syntax, decoupling the pipeline from a single model's architecture.

### Phase-Specific Prompting

The logic is split between Phase 1 (`build_classification_prompt`) and Phase 2 (`build_extraction_prompt`). Phase 1 provides a compressed, **zero-shot schema** summary, while Phase 2 injects the full schema along with **few-shot** examples.

### O(1) I/O Caching

The `load_examples` method implements an in-memory dictionary cache (`_examples_cache`). Once a function's few-shot JSON file is read from the disk, it is cached to serve all future batch requests instantly.

### Optimised Assembly

Large multi-turn prompt strings are assembled using list appending and `"".join(parts)` rather than standard `+=` string concatenation, optimising memory allocation during batch runs.

### Dynamic Few-Shot Grounding

The module automatically scans the `/few_shot/` directory for JSON files matching the target function's name (e.g., `fn_substitute_string_with_regex.json`) and weaves them natively into the model's message boundaries.

## Challenges Solved

### Syntax Hallucinations

By strictly enforcing the correct **control tokens** (like `<|im_start|>` vs `[INST]`) based on the model type, it prevents the LLM from becoming confused and generating out-of-distribution garbage text.

### Disk I/O Bottlenecks

The memory caching mechanism prevents the engine from repeatedly opening and reading the same few-shot JSON files thousands of times during a large batch process, drastically reducing execution time.

### Context Window Bloat

Passing the full parameter schema for every available function during Phase 1 classification would overwhelm a small model's context window. The formatter solves this by only feeding names and descriptions initially, reserving the dense token payload for Phase 2.

### Zero-Shot Weakness in Small Models

Sub-1B models struggle heavily with complex logical extractions (like mapping natural language to regex strings). Dynamically injecting function-specific examples acts as a critical anchor, dramatically improving the model's accuracy before the logit masking even begins.
