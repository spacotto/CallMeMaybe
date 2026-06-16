# Parser Module: Schema Validation and Ingestion

The `parser` module acts as the **gatekeeper for the constrained decoding pipeline**. Before the LLM engine even initialises, this module reads, structures, and validates the raw JSON function definitions provided by the user. By utilising **Pydantic to enforce data models**, the parser guarantees that the downstream token-masking engine never encounters unexpected `NoneType` errors or malformed schemas.
