import time
import argparse
import json
import os

from src.engine import ConstrainedDecoder
from src.parser import SchemaParser
from src.utils import Formatter, error
from src.visualizer import Visualizer

def main() -> None:

    start = time.time()

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

    # 1. Extract all prompts into a single list for batching
    prompts = [test.get("prompt") if isinstance(test, dict) else str(test) for test in test_cases]

    # --- BATCH EXECUTION ---
    print(Formatter.apply('bold', 'yellow', f">>> Generating batch of {len(prompts)} prompts simultaneously..."))
    print(Formatter.apply(None, 'gray', "-" * 70))
    generation_start = time.time()

    try:
        # 2. Call the new batched method ONCE for the entire dataset
        batch_results = engine.generate_batch(
            prompts=prompts,
            functions=functions_schema,
            max_new_tokens=120,
            verbose=args.verbose
        )
    except Exception as e:
        error(f"Batch generation failed: {e}")
        return

    gen_time = time.time() - generation_start
    avg_gen_time = gen_time / len(prompts) # Calculate average time per prompt for the visualizer

    # --- PARSING AND VALIDATION ---
    # 3. Loop over the generated results instead of generating them sequentially
    for idx, (prompt, json_result_str) in enumerate(zip(prompts, batch_results)):
        Visualizer.print_prompt_start(idx + 1, len(prompts), prompt)

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

        # Delegate Footer and Final Render
        Visualizer.print_generation_time(avg_gen_time, is_valid=is_valid_json)
        Visualizer.print_json_render(results[-1])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    txt = f"\n 💾 All results successfully saved to {args.output}"
    print(Formatter.apply('bold', 'cyan', txt))
    stopwatch = f' ⏳ Output generated in {time.time() - start:.1f}s\n'
    print(Formatter.apply('bold', 'lime', stopwatch))

if __name__ == "__main__":
    main()
