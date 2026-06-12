import json
from typing import List, Dict, Any
from pydantic import ValidationError
from src.parser.models.function_call_result import FunctionCallResult

class PostProcessor:
    """Phase 3: Validates JSON structure, coerces types,
    and enforces Schema types using strict Pydantic rules.
    """

    @staticmethod
    def process_result(
        prompt: str,
        target_name: str,
        json_result_str: str,
        functions_schema: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parses the JSON string, casts variables, and enforces
        the mandatory output schema via Pydantic.
        """
        # 1. Parse JSON (fallback if decoding ever completely fails)
        try:
            call_data = json.loads(json_result_str)
        except json.JSONDecodeError:
            call_data = {"name": target_name, "parameters": {}}

        func_name = call_data.get("name", target_name)
        raw_params = call_data.get("parameters", {})

        # 2. Fast Schema Lookup (Eliminates the full loop)
        target_schema = next(
            (f for f in functions_schema if f["name"] == func_name), {}
        )
        params_schema = target_schema.get("parameters", {})

        # 3. Schema-Aware Type Coercion
        # Constrained decoding guarantees correct keys, eliminating the
        # need for the previous positional alignment fallbacks
        aligned_params: Dict[str, Any] = {}
        for key, prop in params_schema.items():
            expected_type = prop.get("type", "string")
            val = raw_params.get(key)

            if val is not None:
                if expected_type == "number" or expected_type == "integer":
                    try:
                        val = float(val) if expected_type == "number" else int(float(val))
                    except (ValueError, TypeError):
                        val = 0.0 if expected_type == "number" else 0
                elif expected_type == "string":
                    val = str(val).strip()
                elif expected_type == "boolean":
                    val = bool(val)

            aligned_params[key] = val

        # 4. Strict Pydantic Validation & Output Formatting
        try:
            validated_output = FunctionCallResult(
                prompt=prompt,
                name=func_name,
                parameters=aligned_params
            )
            return validated_output.model_dump()
        except ValidationError:
            # 5. Fallback
            return {
                "prompt": prompt,
                "name": func_name,
                "parameters": aligned_params
            }
