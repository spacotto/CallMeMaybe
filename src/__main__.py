import time
import argparse
import json
import os

from src.engine import ConstrainedDecoder
from src.parser import SchemaParser
from src.utils import Formatter, error
from src.visualizer import Visualizer

def main() -> None:

    cli_parser = argparse.ArgumentParser(description="Constrained Decoding LLM Engine")
    cli_parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
    cli_parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
    cli_parser.add_argument("--output", type=str, default="data/output/function_calls.json")
    cli_parser.add_argument("--verbose", action="store_true")
    args = cli_parser.parse_args()

    print(Formatter.apply('bold', 'yellow', ">>> Initializing Constrained Decoder Engine..."))
    start_time = time.time()

    try:
        engine = ConstrainedDecoder(model_name="Qwen/Qwen3-0.6B")
        elapsed = time.time() - start_time
        print(Formatter.apply('bold', 'cyan', f">>> Engine loaded in {elapsed:.2f} seconds.\n"))
    except Exception as e:
        error(f"Failed to load model architecture: {e}")
        return

    schema_parser = SchemaParser(file_path=args.functions_definition)
    functions_schema = schema_parser.load_functions()

    if not functions_schema:
        error("Execution aborted: No valid functions schema found.")
        return

    if not os.path.exists(args.input):
        error(f"Input file not found at: {args.input}")
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    results = []

    for idx, test in enumerate(test_cases):
        prompt = test.get("prompt") if isinstance(test, dict) else str(test)

        # Delegate Header Render
        Visualizer.print_prompt_start(idx + 1, len(test_cases), prompt)

        try:
            generation_start = time.time()
            json_result_str = engine.generate_function_call(
                user_prompt=prompt,
                functions=functions_schema,
                max_new_tokens=120,
                verbose=args.verbose
            )

            is_valid_json = True
            try:
                call_data = json.loads(json_result_str)
                func_name = call_data.get("name", "MISSING_NAME")
                raw_params = call_data.get("parameters", {})

                expected_keys = []
                for f_schema in functions_schema:
                    if f_schema["name"] == func_name:
                        expected_keys = list(f_schema.get("parameters", {}).keys())
                        break

                aligned_params = {}
                raw_values = list(raw_params.values())

                for i, expected_key in enumerate(expected_keys):
                    if expected_key in raw_params:
                        val = raw_params[expected_key]
                    elif i < len(raw_values):
                        val = raw_values[i]
                    else:
                        val = ""
                    aligned_params[expected_key] = float(val) if type(val) is int else val

                results.append({
                    "prompt": prompt,
                    "name": func_name,
                    "parameters": aligned_params
                })

            except json.JSONDecodeError:
                is_valid_json = False
                error(f"Model generated invalid JSON for prompt {idx+1}")
                results.append({
                    "prompt": prompt,
                    "error": "Invalid JSON generated",
                    "raw": json_result_str
                })

            gen_time = time.time() - generation_start

            # Delegate Footer and Final Render
            Visualizer.print_generation_time(gen_time, is_valid=is_valid_json)
            Visualizer.print_json_render(results[-1])

        except Exception as e:
            error(f"Generation loop failed on prompt {idx+1}: {e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(Formatter.apply('bold', 'cyan', f"\n 💾 All results successfully saved to {args.output}\n"))


if __name__ == "__main__":
    main()
