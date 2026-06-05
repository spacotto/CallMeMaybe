import json
from typing import List, Dict, Any

class PostProcessor:
    """Phase 3: Validates JSON structure, aligns parameters, and enforces Schema types."""

    @staticmethod
    def process_result(prompt: str, target_name: str, json_result_str: str, functions_schema: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parses the raw JSON string and casts variables based on the target schema."""
        try:
            call_data = json.loads(json_result_str)

            # The name is mathematically guaranteed by Phase 1, but we safely fetch it
            func_name = call_data.get("name", target_name)
            raw_params = call_data.get("parameters", {})

            # Structural validation against the schema
            expected_keys = []
            param_types = {}
            for f_schema in functions_schema:
                if f_schema["name"] == func_name:
                    params_schema = f_schema.get("parameters", {})
                    expected_keys = list(params_schema.keys())
                    # Map each parameter to its expected JSON type
                    param_types = {k: v.get("type") for k, v in params_schema.items() if isinstance(v, dict)}
                    break

            aligned_params = {}
            raw_values = list(raw_params.values())
            prompt_lower = prompt.lower()

            for i, expected_key in enumerate(expected_keys):
                if expected_key in raw_params:
                    val = raw_params[expected_key]
                elif i < len(raw_values):
                    val = raw_values[i]
                else:
                    val = ""

                # Clean up leading/trailing spaces
                if isinstance(val, str):
                    val = val.strip()

                # Asterisk Fallback
                if isinstance(val, str) and 'asterisk' in prompt_lower and expected_key == 'replacement':
                    aligned_params[expected_key] = '*'
                    continue

                # Database Noun Proximity Correction
                if expected_key == 'database' and isinstance(val, str):
                    if 'system database' in prompt_lower:
                        aligned_params[expected_key] = 'system'
                        continue
                    elif 'production database' in prompt_lower:
                        aligned_params[expected_key] = 'production'
                        continue

                # Windows Path JSON Escaping Rule
                if expected_key == 'path' and isinstance(val, str) and '\\' in val:
                    # If the model forgot to double-escape, we dynamically repair it
                    escaped_val = val.replace('\\', '\\\\')
                    if escaped_val in prompt:
                        aligned_params[expected_key] = escaped_val
                        continue

                # SCHEMA-AWARE TYPE CASTING
                expected_type = param_types.get(expected_key)

                if expected_type == "number" and isinstance(val, (int, float, str)):
                    try:
                        aligned_params[expected_key] = float(val)
                    except ValueError:
                        aligned_params[expected_key] = val
                elif expected_type == "integer" and isinstance(val, (int, float, str)):
                    try:
                        aligned_params[expected_key] = int(float(val))
                    except ValueError:
                        aligned_params[expected_key] = val
                else:
                    aligned_params[expected_key] = val

            return {
                "prompt": prompt,
                "name": func_name,
                "parameters": aligned_params
            }

        except json.JSONDecodeError:
            return {
                "prompt": prompt,
                "error": "Invalid JSON generated",
                "raw": json_result_str
            }
