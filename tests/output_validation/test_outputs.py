import os
import json
import pytest

# Mock Paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_DATA_DIR = os.path.join(MODULE_DIR, "mock_data")
MOCK_SCHEMA_PATH = os.path.join(MOCK_DATA_DIR, "mock_schema.json")

# Real Paths
ROOT_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
REAL_OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "output", "function_calling_results.json")
REAL_SCHEMA_PATH = os.path.join(ROOT_DIR, "data", "input", "functions_definition.json")

def validate_item_against_schema(item: dict, schema: dict) -> list:
    """Core validation logic replicating the Moulinette constraints."""
    errors = []

    expected_keys = {"prompt", "name", "parameters"}
    if set(item.keys()) != expected_keys:
        errors.append(f"Invalid root keys. Found: {list(item.keys())}")
        return errors

    func_name = item["name"]
    target_schema = {s["name"]: s for s in schema}.get(func_name, {}).get("parameters", {})
    actual_params = item["parameters"]

    expected_params = set(target_schema.keys())
    provided_params = set(actual_params.keys())

    if provided_params != expected_params:
        extra = provided_params - expected_params
        missing = expected_params - provided_params
        if extra:
            errors.append(f"Extra parameters found: {extra}")
        if missing:
            errors.append(f"Missing required parameters: {missing}")

    for param_name, param_val in actual_params.items():
        if param_name in target_schema:
            expected_type = target_schema[param_name].get("type")
            type_map = {"string": str, "number": (int, float), "boolean": bool, "object": dict}

            # Special check to prevent Python booleans (True/False) from passing as numbers (1/0)
            req_type = type_map.get(expected_type, str)
            if expected_type == "number" and isinstance(param_val, bool):
                errors.append(f"Type mismatch on '{param_name}': Expected number, got boolean")
            elif not isinstance(param_val, req_type):
                errors.append(f"Type mismatch on '{param_name}': Expected {expected_type}")

    return errors

def load_mock_json(filename: str) -> dict:
    with open(os.path.join(MOCK_DATA_DIR, filename), "r") as f:
        return json.load(f)[0]

def load_schema() -> dict:
    with open(MOCK_SCHEMA_PATH, "r") as f:
        return json.load(f)

# --- MOCK TESTS (PROVING THE VALIDATOR WORKS) ---

def test_extra_root_keys():
    """Validator must reject objects with keys other than prompt, name, parameters."""
    item = load_mock_json("extra_keys.json")
    schema = load_schema()
    errors = validate_item_against_schema(item, schema)
    assert any("Invalid root keys" in e for e in errors)

def test_wrong_parameter_type():
    """Validator must reject strings masquerading as numbers."""
    item = load_mock_json("wrong_type.json")
    schema = load_schema()
    errors = validate_item_against_schema(item, schema)
    assert any("Type mismatch" in e for e in errors)

def test_missing_required_parameter():
    """Validator must reject payloads missing schema-defined parameters."""
    item = load_mock_json("missing_param.json")
    schema = load_schema()
    errors = validate_item_against_schema(item, schema)
    assert any("Missing required parameters" in e for e in errors)

# --- REAL OUTPUT TEST (VALIDATING THE ACTUAL ENGINE RESULTS) ---

def test_real_output_compliance():
    """Validates the actual generated output file against the real schema if it exists."""
    if not os.path.exists(REAL_OUTPUT_PATH):
        pytest.skip("Real output file not found. Run the engine first to generate data.")
    if not os.path.exists(REAL_SCHEMA_PATH):
        pytest.skip("Real schema file not found. Cannot validate real data.")

    with open(REAL_OUTPUT_PATH, "r", encoding="utf-8") as f:
        real_data = json.load(f)

    with open(REAL_SCHEMA_PATH, "r", encoding="utf-8") as f:
        real_schema = json.load(f)

    assert isinstance(real_data, list), "The root of the actual output JSON must be a list."

    all_errors = []
    for index, item in enumerate(real_data):
        item_errors = validate_item_against_schema(item, real_schema)
        for e in item_errors:
            func_name = item.get("name", "Unknown Function")
            all_errors.append(f"Item {index} [{func_name}]: {e}")

    # If all_errors is not empty, the assert will fail and print out the exact violations
    assert not all_errors, f"Moulinette Validation Failed for Real Output:\n" + "\n".join(all_errors)
