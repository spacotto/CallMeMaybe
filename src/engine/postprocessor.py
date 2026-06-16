"""
Phase 3 engine module: Post-Processing and Validation.

This module acts as the final safety net in the constrained decoding
pipeline. It parses the raw generated strings, gracefully catches
any structural anomalies, coerces data types to match the schema,
and enforces absolute compliance using Pydantic models.
"""

import json
from typing import List, Dict, Any
from pydantic import ValidationError
from src.parser.models.function_call_result import FunctionCallResult


class PostProcessor:
    """
    Phase 3: Validates JSON, coerces types, and enforces schemas.

    This class contains static methods designed to take the raw output
    from the generation loop and transform it into a strictly typed,
    guaranteed-compliant dictionary ready for serialization.
    """

    @staticmethod
    def process_result(
        prompt: str,
        target_name: str,
        json_result_str: str,
        functions_schema: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parses, casts, and validates a single generated result.

        Args:
            prompt (str): The original user request.
            target_name (str): The expected function name.
            json_result_str (str): The raw JSON string from the LLM.
            functions_schema (List[Dict[str, Any]]): Available schemas.

        Returns:
            Dict[str, Any]: A structurally perfect dictionary matching
            the `FunctionCallResult` schema.
        """
        # ------------------------------------------------------------------
        # [ERROR CATCHING]: JSON Parsing Fail-Safe
        # If the LLM somehow bypasses the masking constraints and generates
        # invalid JSON, this block catches the JSONDecodeError and injects
        # a safe, empty structural baseline to prevent pipeline crashes.
        # ------------------------------------------------------------------
        try:
            call_data = json.loads(json_result_str)
        except json.JSONDecodeError:
            call_data = {"name": target_name, "parameters": {}}

        func_name = call_data.get("name", target_name)
        raw_params = call_data.get("parameters", {})

        # Fast Schema Lookup using a generator expression to avoid O(N) loops
        target_schema = next(
            (f for f in functions_schema if f["name"] == func_name), {}
        )
        params_schema = target_schema.get("parameters", {})

        # ------------------------------------------------------------------
        # [ERROR CATCHING]: Type Coercion & Recovery
        # Constrained decoding guarantees correct keys, but type edge cases
        # (e.g., generating "3.0" for an integer) can still occur. This
        # block catches TypeErrors and ValueErrors, coercing the data back
        # into the expected schema format safely.
        # ------------------------------------------------------------------
        aligned_params: Dict[str, Any] = {}
        for key, prop in params_schema.items():
            expected_type = prop.get("type", "string")
            val = raw_params.get(key)

            if val is not None:
                if expected_type == "number" or expected_type == "integer":
                    try:
                        val = (float(val) if expected_type == "number"
                               else int(float(val)))
                    except (ValueError, TypeError):
                        val = 0.0 if expected_type == "number" else 0
                elif expected_type == "string":
                    val = str(val).strip()
                elif expected_type == "boolean":
                    val = bool(val)

            aligned_params[key] = val

        # ------------------------------------------------------------------
        # [ERROR CATCHING]: Pydantic Validation Fallback
        # As the absolute final check, the data is passed through the strict
        # Pydantic model. If it fails (e.g., missing mandatory fields), we
        # catch the ValidationError and force a compliant fallback dictionary
        # to guarantee downstream safety.
        # ------------------------------------------------------------------
        try:
            validated_output = FunctionCallResult(
                prompt=prompt,
                name=func_name,
                parameters=aligned_params
            )
            return validated_output.model_dump()
        except ValidationError:
            return {
                "prompt": prompt,
                "name": func_name,
                "parameters": aligned_params
            }
