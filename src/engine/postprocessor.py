import json
from typing import List, Dict, Any
from pydantic import ValidationError
from src.parser.models.function_call_result import FunctionCallResult

class PostProcessor:
    """Phase 3: Validates JSON structure, perfectly aligns parameters,
    and enforces Schema types using strict Pydantic rules.
    """

    @staticmethod
    def process_result(
        prompt: str,
        target_name: str,
        json_result_str: str,
        functions_schema: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parses the raw JSON string, casts variables, and enforces
        the mandatory output schema via Pydantic.
        """
        try:
            call_data = json.loads(json_result_str)

            func_name = call_data.get("name", target_name)
            raw_params = call_data.get("parameters", {})

            # Structural validation against the schema
            expected_keys: List[str] = []
            param_types: Dict[str, Any] = {}
            for f_schema in functions_schema:
                if f_schema["name"] == func_name:
                    params_schema = f_schema.get("parameters", {})
                    expected_keys = list(params_schema.keys())
                    param_types = {
                        k: v.get("type")
                        for k, v in params_schema.items()
                        if isinstance(v, dict)
                    }
                    break

            aligned_params: Dict[str, Any] = {}
            raw_values = list(raw_params.values())

            for i, expected_key in enumerate(expected_keys):
                val: Any = ""
                if expected_key in raw_params:
                    val = raw_params[expected_key]
                elif i < len(raw_values):
                    val = raw_values[i]

                if isinstance(val, str):
                    val = val.strip()

                # SCHEMA-AWARE TYPE CASTING
                expected_type = param_types.get(expected_key)

                if expected_type == "number" and isinstance(
                    val, (int, float, str)
                ):
                    try:
                        aligned_params[expected_key] = float(val)
                    except ValueError:
                        aligned_params[expected_key] = val
                elif expected_type == "integer" and isinstance(
                    val, (int, float, str)
                ):
                    try:
                        aligned_params[expected_key] = int(float(val))
                    except ValueError:
                        aligned_params[expected_key] = val
                else:
                    aligned_params[expected_key] = val

            # 🛡️ PYDANTIC VALIDATION INTEGRATION
            validated_output = FunctionCallResult(
                prompt=prompt,
                name=func_name,
                parameters=aligned_params
            )

            return validated_output.model_dump()

        except (json.JSONDecodeError, ValidationError) as e:
            return {
                "prompt": prompt,
                "error": f"Validation/Parsing Error: {e}",
                "raw": json_result_str
            }
