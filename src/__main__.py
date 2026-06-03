import time
import argparse
import json
import os
from src.engine import ConstrainedDecoder
from src.parser import SchemaParser
from src.utils import Formatter, error, warning

def main() -> None:

    # Catch the CLI arguments sent by the Makefile or use defaults
    cli_parser = argparse.ArgumentParser(description="Constrained Decoding LLM Engine")

    cli_parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to schema JSON (default: data/input/functions_definition.json)"
    )
    cli_parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON (default: data/input/function_calling_tests.json)"
    )
    cli_parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to save generated JSON results (default: data/output/function_calls.json)"
    )

    cli_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print real-time visualization of the generation state machine"
    )
    args = cli_parser.parse_args()

    print(Formatter.apply('bold', 'yellow',
                          ">>> Initializing Constrained Decoder Engine..."))
    start_time = time.time()

    try:
        engine = ConstrainedDecoder(model_name="Qwen/Qwen3-0.6B")
        elapsed = time.time() - start_time
        txt = f">>> Engine loaded in {elapsed:.2f} seconds.\n"
        print(Formatter.apply('bold', 'cyan', txt))
    except Exception as e:
        error(f"Failed to load model architecture: {e}")
        return

    # Load the dynamic schema specified by the Makefile
    schema_parser = SchemaParser(file_path=args.functions_definition)
    functions_schema = schema_parser.load_functions()

    if not functions_schema:
        error("Execution aborted: No valid functions schema found.")
        return

    # Load the Input Prompts File
    if not os.path.exists(args.input):
        error(f"Input file not found at: {args.input}")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    results = []

    # Iterate through the tests and execute constrained decoding
    for idx, test in enumerate(test_cases):
        prompt = test.get("prompt") if isinstance(test, dict) else str(test)

        # This print statement confirms the loop is running correctly
        txt = f">>> Processing Prompt [{idx+1}/{len(test_cases)}]: "
        print(Formatter.apply('bold', 'yellow', txt) + prompt)

        try:
            generation_start = time.time()
            json_result_str = engine.generate_function_call(
                user_prompt=prompt,
                functions=functions_schema,
                max_new_tokens=120,
                verbose=args.verbose
            )

            # Cast the guaranteed JSON string back into a Python dictionary
            try:
                call_data = json.loads(json_result_str)
                func_name = call_data.get("name", "MISSING_NAME")
                raw_params = call_data.get("parameters", {})

                # Fetch Expected Schema Keys for this specific function
                expected_keys = []
                for f_schema in functions_schema:
                    if f_schema["name"] == func_name:
                        # Directly extract the keys from parameters
                        expected_keys = list(f_schema.get("parameters", {}).keys())
                        break

                # Strict Schema Auto-Alignment & Pruning
                aligned_params = {}
                raw_values = list(raw_params.values())

                for i, expected_key in enumerate(expected_keys):
                    # Direct match: The model used the correct key from the schema
                    if expected_key in raw_params:
                        val = raw_params[expected_key]
                    # Positional match: The model hallucinated a key
                    # (like "string" instead of "s")
                    elif i < len(raw_values):
                        val = raw_values[i]
                    # Missing value fallback
                    else:
                        val = ""

                    # Enforce floats for numerical fields
                    aligned_params[expected_key] = float(val) if type(val) is int else val

                # Structure the final output strictly to the flat layout
                results.append({
                    "prompt": prompt,
                    "name": func_name,
                    "parameters": aligned_params
                })

            except json.JSONDecodeError:
                error(f"Model generated invalid JSON for prompt {idx+1}")
                results.append({
                    "prompt": prompt,
                    "error": "Invalid JSON generated",
                    "raw": json_result_str
                })

            gen_time = time.time() - generation_start
            txt = f">>> Valid JSON genrated in {gen_time:.2f}s\n"
            print(Formatter.apply('bold', 'cyan', txt))

        except Exception as e:
            error(f"Generation loop failed on prompt {idx+1}: {e}")

    # Save the final JSON array to the specified output path
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    txt = f">>> All results successfully saved to {args.output} 💾\n"
    print(Formatter.apply('bold', 'cyan', txt))


if __name__ == "__main__":
    main()
