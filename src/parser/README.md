# Parser Module: Schema Validation and Ingestion

The `parser` module acts as the **gatekeeper for the constrained decoding pipeline**. Before the LLM engine even initialises, this module reads, structures, and validates the raw JSON function definitions provided by the user. By utilising **Pydantic to enforce data models**, the parser guarantees that the downstream token-masking engine never encounters unexpected `NoneType` errors or malformed schemas.

## Theoretical Concepts

### Data Validation vs. Data Parsing

Parsing simply means translating a string of text (like a JSON file) into a programmable object (like a Python dictionary). Validation goes a step further: it ensures that **the parsed data conforms to a strict set of structural and logical rules**. This module does both, refusing to pass data to the engine unless it fits the predefined mould.

### Static Typing in a Dynamic Environment

Python and JSON are inherently dynamic, meaning a dictionary key can hold a string, a list, or simply vanish without warning. The masking engine, however, relies on **predictability** to build its **O(1) mathematical matrices**. By using Pydantic, we introduce "C-like" strictness to Python dictionaries, enforcing required fields and default fallbacks.

### Graceful Degradation

In a pipeline, a single missing comma in one function definition should not crash the entire batch processing job. Graceful degradation is the practice of **catching localised errors, logging them, skipping the corrupted data, and allowing the rest of the healthy system to continue executing**.

## Design Decisions

### Strict Type Validation

The module leverages Pydantic (`BaseModel`, `Field`) to map and strictly validate the input JSON schemas against hierarchical data models, including `FunctionDefinition`, `FunctionParameters`, and `FunctionProperty`.  

### Graceful Degradation

During schema loading, if an individual function definition fails Pydantic validation, the parser logs the `ValidationError` and explicitly skips the malformed item rather than crashing the entire pipeline.

### Downstream Compatibility

After ensuring structural integrity, the validated Pydantic objects are converted back into standard Python dictionaries using `model_dump(exclude_none=True)` to remain compatible with the core masking engine.

### Static Nesting Detection

The `SchemaParser` includes a static `is_nested` method that iterates through function parameters to explicitly detect if any parameter is categorised as an "object".

## Challenges Solved

### Malformed Schema Crashes

By enforcing structural rules upfront and catching both `json.JSONDecodeError` and Pydantic exceptions, the module prevents the engine from halting due to syntax errors, missing root lists, or invalid structures in the user-provided definitions.

### Missing Data Handling

The use of `Optional` fields and default dictionary/list factories in the data models ensures the downstream pipeline always receives safe, predictable structures, bypassing `NoneType` errors.

## Glossary

### Model (Data Model)

In the context of the parser, a "model" refers to a Pydantic class that acts as a strict structural blueprint. It defines the exact shape, required fields, and data types a Python object must possess to be considered valid. This should not be confused with the Large Language Model (LLM) itself.

### Nested Object

A parameter within a function that is itself a dictionary containing its own parameters (e.g., a `location` object containing `city` and `country` properties).

### Pydantic

A Python data validation library that enforces type hints at runtime, throwing errors if incoming data doesn't match the expected model.

### Schema

The blueprint or structural definition of a function, dictating what parameters it requires and what data types they must be.

### Serialization/Deserialization

The process of converting complex objects (like Pydantic models) into flat formats (like JSON/dictionaries), and vice versa.

* **Output Consistency:** The module introduces the `FunctionCallResult` model, which acts as a structural guardrail to guarantee that every generated item matches the project's mandatory `prompt`, `name`, and `parameters` output format.
