import time
import argparse
import json
import os
from typing import List

from src.engine import FunctionClassifier, ParameterExtractor, NestedExtractor, PostProcessor
from src.parser import SchemaParser
from src.utils import Formatter, error
from src.visualizer import Visualizer


def main() -> None:

    start = time.time()

    # --- CLI SETUP ---
    try:
        cli_parser = argparse.ArgumentParser(
            description="Constrained Decoding LLM Engine"
        )
        cli_parser.add_argument(
            "--functions_definition",
            type=str,
            default="data/input/functions_definition.json"
        )
        cli_parser.add_argument(
            "--input",
            type=str,
            default="data/input/function_calling_tests.json"
        )
        cli_parser.add_argument(
            "--output",
            type=str,
            default="data/output/function_calls.json"
        )
        cli_parser.add_argument("--verbose", action="store_true")
        args = cli_parser.parse_args()

    except Exception as e:
        error(f"CLI Initialization failed: {e}")
        return

    print(
        Formatter.apply(
            'bold', 'yellow', ">>> Initializing Three-Stage Pipeline..."
        )
    )
    start_time = time.time()

    # --- MODEL INITIALIZATION ---
    try:
        classifier = FunctionClassifier(model_name="Qwen/Qwen3-0.6B")

        # Instantiate BOTH extraction paths
        fast_extractor = ParameterExtractor(classifier_instance=classifier)
        slow_extractor = NestedExtractor(classifier_instance=classifier)

        elapsed = time.time() - start_time
        print(
            Formatter.apply(
                'bold',
                'cyan',
                f">>> Engine loaded in {elapsed:.2f} seconds.\n\n"
            )
        )

    except Exception as e:
        error(f"Failed to load model architecture: {e}")
        return

    # --- SCHEMA PARSING ---
    schema_parser = SchemaParser(file_path=args.functions_definition)
    functions_schema = schema_parser.load_functions()

    if not functions_schema:
        error("Execution aborted: No valid functions schema found.")
        return

    # --- PHASE 1: CLASSIFICATION ---
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        prompts: List[str] = [
            str(item.get("prompt", ""))
            for item in input_data
            if isinstance(item, dict) and "prompt" in item
        ]

        print(Formatter.apply(None, 'gray', "-" * 70))
        print(Formatter.apply('bold', 'yellow', f">>> Phase 1: Classifying {len(prompts)} prompts..."))

        target_names: List[str] = classifier.classify_batch(
            prompts, functions_schema
        )

    except Exception as e:
        error(f"Phase 1 (Classification) failed: {e}")
        return

    # --- PHASE 2: PARAMETER EXTRACTION (ROUTED) ---
    try:
        print(Formatter.apply(None, 'gray', "-" * 70))
        print(Formatter.apply('bold', 'yellow', ">>> Phase 2: Extracting parameters (Routed)..."))
        print(Formatter.apply(None, 'gray', "-" * 70))

        generation_start = time.time()

        # Step 1: Split the Batch using the SchemaParser
        flat_prompts, flat_names, flat_indices = [], [], []
        nested_prompts, nested_names, nested_indices = [], [], []

        for idx, target_name in enumerate(target_names):
            if SchemaParser.is_nested(target_name, functions_schema):
                nested_prompts.append(prompts[idx])
                nested_names.append(target_name)
                nested_indices.append(idx)
            else:
                flat_prompts.append(prompts[idx])
                flat_names.append(target_name)
                flat_indices.append(idx)

        # Step 2: Process Fast Path (Flat Schemas)
        flat_results = []
        if flat_prompts:
            flat_results = fast_extractor.extract_batch(
                prompts=flat_prompts,
                function_names=flat_names,
                functions=functions_schema,
                max_new_tokens=120,
                verbose=args.verbose
            )

        # Step 3: Process Slow Path (Nested Schemas)
        nested_results = []
        if nested_prompts:
            nested_results = slow_extractor.extract_batch(
                prompts=nested_prompts,
                function_names=nested_names,
                functions=functions_schema,
                max_new_tokens=180, # Nested logic takes more tokens!
                verbose=args.verbose
            )

        # Step 4: Reconstruct Original Batch Order
        json_results = [""] * len(prompts)
        for original_idx, result in zip(flat_indices, flat_results):
            json_results[original_idx] = result
        for original_idx, result in zip(nested_indices, nested_results):
            json_results[original_idx] = result

        avg_gen_time = (time.time() - generation_start) / max(1, len(prompts))

    except Exception as e:
        error(f"Phase 2 (Extraction) failed: {e}")
        return

    # --- PHASE 3: POST-PROCESSING ---
    results = []
    try:
        print(Formatter.apply('bold', 'yellow', ">>> Phase 3: Validating output..."))
        print(Formatter.apply(None, 'gray', "-" * 70))

        for idx, (prompt, target_name, json_result_str) in enumerate(
            zip(prompts, target_names, json_results)
        ):
            try:
                if args.verbose:
                    Visualizer.print_prompt_start(idx + 1, len(prompts), prompt)

                final_item = PostProcessor.process_result(
                    prompt,
                    target_name,
                    json_result_str,
                    functions_schema
                )
                results.append(final_item)

                is_valid_json = "error" not in final_item
                if not is_valid_json:
                    error(f"Model generated invalid JSON for prompt {idx+1}")

                if args.verbose:
                    Visualizer.print_generation_time(avg_gen_time, is_valid=is_valid_json)
                    Visualizer.print_json_render(final_item)

            except Exception as item_e:
                error(f"Critical Parsing Error on item {idx+1}: {item_e}")
                results.append({
                    "prompt": prompt,
                    "error": f"Internal Processing Error: {item_e}",
                    "raw": json_result_str
                })

    except Exception as e:
        error(f"Phase 3 (Post-Processing loop) failed: {e}")

    # --- OUTPUT SAVING ---
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as output_file:
            json.dump(results, output_file, indent=4)

        print(Formatter.apply('bold', 'cyan', f"\n\n 💾 All results successfully saved to {args.output}"))
        final_time = time.time() - start
        if final_time < 60:
            stopwatch = f' ⏳ Pipeline execution completed in {final_time:.1f}s\n'
        else:
            stopwatch = f' ⏳ Pipeline execution completed in {final_time / 60:.1f}m\n'
        print(Formatter.apply('bold', 'lime', stopwatch))

    except Exception as e:
        error(f"Failed to save results: {e}")

if __name__ == "__main__":
    main()
