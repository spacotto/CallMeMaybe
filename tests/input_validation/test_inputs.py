import os
import subprocess
import pytest

# Paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
INVALID_DATA_DIR = os.path.join(MODULE_DIR, "invalid_data")
MAIN_SCRIPT = ["uv", "run", "python", "-m", "src"]

def run_engine_with_schema(schema_filename: str) -> subprocess.CompletedProcess:
    """Helper to run the engine with a specific broken schema file."""
    schema_path = os.path.join(INVALID_DATA_DIR, schema_filename)
    # Redirect stdout/stderr so it doesn't spam the test console
    return subprocess.run(
        MAIN_SCRIPT + ["--functions_definition", schema_path],
        capture_output=True,
        text=True
    )

def test_missing_file():
    """Engine should handle a file path that does not exist."""
    result = run_engine_with_schema("does_not_exist.json")
    # Engine should exit gracefully, not with a fatal Python crash (exit code 1 or 0 is fine depending on your error() implementation)
    assert "Schema file not found" in result.stderr or "Schema file not found" in result.stdout

def test_malformed_json():
    """Engine should catch JSONDecodeError."""
    result = run_engine_with_schema("malformed.json")
    assert "Failed to parse JSON" in result.stderr or "Failed to parse JSON" in result.stdout

def test_wrong_root_type():
    """Engine should reject JSON where the root is not a list."""
    result = run_engine_with_schema("wrong_root.json")
    assert "Invalid schema format" in result.stderr or "Invalid schema format" in result.stdout

def test_empty_file():
    """Engine should handle completely empty files gracefully."""
    result = run_engine_with_schema("empty.json")
    assert "Failed to parse JSON" in result.stderr or "Failed to parse JSON" in result.stdout
