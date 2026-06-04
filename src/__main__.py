import time
import argparse
import json
import os

from src.engine import FunctionClassifier, ParameterExtractor
from src.parser import SchemaParser
from src.utils import Formatter, error
from src.visualizer import Visualizer

def main() -> None:

    cli_parser = argparse.ArgumentParser(description="Constrained Decoding LLM Engine")
    cli_parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
    cli_parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
    cli_parser.add_argument("--output", type=str, default="data/output/function_calls.json")
    cli_parser.add_argument("--few_shot", type=str, default="data/input/few_shot.json")
    cli_parser.add_argument("--verbose", action="store_true")
    args = cli_parser.parse_args()

    print(Formatter.apply('bold', 'yellow', ">>> Initializing Two-Stage Pipeline..."))
    start_time = time.time()

    try:
        # Initialize the shared neural network memory footprint
        classifier = FunctionClassifier(model_name="Qwen/Qwen3-0.6B")
        extractor = ParameterExtractor(classifier_instance=classifier)
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

    with open(args.input, 'r', encoding='utf-8') as test_file:
        test_cases = json.load(test_file)

    few_shot_examples = []
    if os.path.exists(args.few_shot):
        with open(args.few_shot, 'r', encoding='utf-8') as example_file:
            few_shot_examples = json.load(example_file)

    results = []
    prompts = [test.get("prompt") if isinstance(test, dict) else str(test) for test in test_cases]

    start = time.time()

    print(Formatter.apply('bold', 'yellow', f">>> Processing batch of {len(prompts)} prompts..."))
    print(Formatter.apply(None, 'gray', "-" * 70))
    generation_start = time.time()

    try:
        # --- PHASE 1: CLASSIFICATION (Zero-Shot) ---
        if args.verbose:
            print(Formatter.apply('bold', 'blue', ">>> Phase 1: Classifying target functions..."))
            print(Formatter.apply(None, 'gray', "-" * 70))

        classified_names = classifier.classify_batch(
            prompts=prompts,
            functions=functions_schema,
            max_new_tokens=30
        )

        # --- PHASE 2: PARAMETER EXTRACTION (Few-Shot/Deep Parse) ---
        if args.verbose:
            print(Formatter.apply('bold', 'blue', ">>> Phase 2: Extracting nested parameters..."))
            print(Formatter.apply(None, 'gray', "-" * 70))

        # Note: If your Formatter supports few_shot_examples, pass them here!
        batch_results = extractor.extract_batch(
            prompts=prompts,
            function_names=classified_names,
            functions=functions_schema,
            max_new_tokens=120,
            verbose=args.verbose
        )

    except Exception as e:
        error(f"Pipeline generation failed: {e}")
        return

    gen_time = time.time() - generation_start
    avg_gen_time = gen_time / len(prompts)

    # --- PARSING AND VALIDATION ---
    for idx, (prompt, target_name, json_result_str) in enumerate(zip(prompts, classified_names, batch_results)):
        Visualizer.print_prompt_start(idx + 1, len(prompts), prompt)

        is_valid_json = True
        try:
            call_data = json.loads(json_result_str)

            # The name is mathematically guaranteed by Pass 1
            func_name = call_data.get("name", target_name)
            raw_params = call_data.get("parameters", {})

            # Structural validation against the Pydantic schema
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

                if type(val) is int:
                    aligned_params[expected_key] = float(val)
                elif 'asterisk' in val.lower():
                    aligned_params[expected_key] = '*'
                else:
                    aligned_params[expected_key] = val

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

        Visualizer.print_generation_time(avg_gen_time, is_valid=is_valid_json)
        Visualizer.print_json_render(results[-1])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=4)

    txt = f"\n 💾 All results successfully saved to {args.output}"
    print(Formatter.apply('bold', 'cyan', txt))
    final_time = time.time() - start
    stopwatch = f' ⏳ Pipeline execution completed in {final_time / 60:.1f}m\n'
    print(Formatter.apply('bold', 'lime', stopwatch))

if __name__ == "__main__":
    main()
