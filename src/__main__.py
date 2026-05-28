"""

Usage:
    python3 main.py
    Example: python3 main.py
"""

import argparse
import json
import os
import sys
from pydantic import ValidationError

from utils import error, warning
from src.parser import load_function_definitions, load_input_prompts

def execute_pipeline(func_path: str, input_path: str, output_path: str) -> None:
    """Orchestrates the program flow using the parsing module."""

    # --- Parse and validate data structures

    try:
        functions = load_function_definitions(func_path)
        prompts = load_input_prompts(input_path)

    except (FileNotFoundError, ValueError, ValidationError) as e:
        error(f"Initialization Failed: {e}")
        sys.exit(1)

    print(f">>> Verified {len(functions)} system functions.")
    print(f">>> Processing {len(prompts)} validation prompts...")

    output_results = []

    # --- Iterate through isolated prompt streams

    for item in prompts:

        try:
            # item.prompt is guaranteed to exist and be a string thanks to Pydantic
            current_prompt = item.prompt

            # --- TODO: Invoke constrained decoding engine here ---
            # result_name, result_params = run_decoder(current_prompt, functions)

            # Temporary mock values for structural demonstration
            result_name = "fn_add_numbers"
            result_params = {"a": 2, "b": 3}

            output_results.append({
                "prompt": current_prompt,
                "name": result_name,
                "parameters": result_params
            })

        except Exception as prompt_err:
            # Keeps individual pipeline anomalies from crashing the program execution
            warning(f"Skipping prompt due to internal failure: {prompt_err}")
            continue

    # --- Secure output destination & dump results

    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_results, f, indent=2)
        print(f"✅ Run complete. Output safely recorded to: {output_path}")

    except IOError as e:
        error(f"Failed to write output file: {e}")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Call Me Maybe: LLM Function Calling")
    ap.add_argument("--functions_definition", default="data/input/functions_definition.json")
    ap.add_argument("--input", default="data/input/function_calling_tests.json")
    ap.add_argument("--output", default="data/output/function_calls.json")
    args = ap.parse_args()

    execute_pipeline(args.functions_definition, args.input, args.output)


if __name__ == "__main__":
    main()
