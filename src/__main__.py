import time
import argparse
import json
import os
from typing import List, Dict, Any

from src.engine import (
    FunctionClassifier,
    SchemaExtractor,
    PostProcessor
)
from src.parser import SchemaParser
from src.utils import Formatter as fmt
from src.utils import error
from src.visualizer import Visualizer


def calculate_prompt_limit(func_name: str,
                           functions_schema: List[Dict[str, Any]]) -> int:
    """Calculates a safe token limit based on the target schema's types."""
    schema = next(
        (f for f in functions_schema if f["name"] == func_name), {}
    )
    params = schema.get("parameters", {})

    if not params:
        return 20

    has_nesting = any(
        isinstance(v, dict) and v.get("type") == "object"
        for v in params.values()
    )

    if has_nesting:
        return 120

    types = [v.get("type") for v in params.values() if isinstance(v, dict)]

    if "string" in types:
        if func_name in ["fn_execute_sql_query",
                         "fn_substitute_string_with_regex"]:
            return 80
        return 60

    num_params = len(params)
    if num_params >= 3:
        return 65

    return 42


def main() -> None:

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
            default="data/output/function_calling_results.json"
        )
        cli_parser.add_argument("--verbose", action="store_true")
        args = cli_parser.parse_args()

    except Exception as e:
        error(f"CLI Initialization failed: {e}")
        return

    print(fmt.apply(
        'bold', 'yellow', ">>> Initializing Pipeline..."
    ))
    start_init = time.time()

    # --- MODEL INITIALIZATION ---
    try:
        classifier = FunctionClassifier(model_name="Qwen/Qwen3-0.6B")
        schema_extractor = SchemaExtractor(classifier_instance=classifier)

        init = time.time() - start_init
        print(fmt.apply(
            'bold',
            'cyan',
            f">>> Engine loaded in {init:.2f} seconds.\n\n"
        ))

    except Exception as e:
        error(f"Failed to load model architecture: {e}")
        return

    # --- SCHEMA PARSING ---
    schema_parser = SchemaParser(file_path=args.functions_definition)
    functions_schema = schema_parser.load_functions()

    if not functions_schema:
        error("Execution aborted: No valid functions schema found.")
        return

    start_pipeline = time.time()

    # --- PHASE 1: CLASSIFICATION ---
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        prompts: List[str] = [
            str(item.get("prompt", ""))
            for item in input_data
            if isinstance(item, dict) and "prompt" in item
        ]

        print(fmt.apply(None, 'gray', "-" * 70))
        msg = f">>> Phase 1: Classifying {len(prompts)} prompts..."
        print(fmt.apply('bold', 'yellow', msg))

        target_names: List[str] = classifier.classify_batch(
            prompts, functions_schema
        )

    except Exception as e:
        error(f"Phase 1 (Classification) failed: {e}")
        return

    # --- PHASE 2: UNIFIED PARAMETER EXTRACTION ---
    try:
        print(fmt.apply(None, 'gray', "-" * 70))
        print(fmt.apply(
                'bold',
                'yellow',
                ">>> Phase 2: Extracting parameters via Schema Engine..."
            ))
        print(fmt.apply(None, 'gray', "-" * 70))

        generation_start = time.time()

        limits = []
        for target_name in target_names:
            limit = calculate_prompt_limit(target_name, functions_schema)
            if SchemaParser.is_nested(target_name, functions_schema):
                limits.append(limit + 50)
            else:
                limits.append(limit)

        json_results = schema_extractor.extract_batch(
            prompts=prompts,
            function_names=target_names,
            functions=functions_schema,
            max_new_tokens_list=limits,
        )

        avg_gen_time = (time.time() - generation_start) / max(1, len(prompts))

    except Exception as e:
        error(f"Phase 2 (Extraction) failed: {e}")
        return

    # --- PHASE 3: POST-PROCESSING ---
    results = []
    try:
        print(fmt.apply('bold', 'yellow', ">>> Phase 3: Validating..."))
        print(fmt.apply(None, 'gray', "-" * 70))

        for idx, (prompt, target_name, json_result_str) in enumerate(
            zip(prompts, target_names, json_results)
        ):
            try:
                if args.verbose:
                    Visualizer.print_prompt_start(
                        idx + 1, len(prompts), prompt
                    )

                final_item = PostProcessor.process_result(
                    prompt,
                    target_name,
                    json_result_str,
                    functions_schema
                )
                results.append(final_item)

                is_valid = "error" not in final_item
                if not is_valid:
                    error(f"Invalid JSON for prompt {idx+1}")

                if args.verbose:
                    Visualizer.print_generation_time(
                        avg_gen_time, is_valid=is_valid
                    )
                    Visualizer.print_json_render(final_item)

            except Exception as item_e:
                error(f"Critical Parsing Error on item {idx+1}: {item_e}")
                results.append({
                    "prompt": prompt,
                    "error": f"Internal Error: {item_e}",
                    "raw": json_result_str
                })

    except Exception as e:
        error(f"Phase 3 (Post-Processing) failed: {e}")

    # --- OUTPUT SAVING ---
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as output_file:
            json.dump(results, output_file, indent=4)

        msg = f"\n\n 💾 All results saved to {args.output}"
        print(fmt.apply('bold', 'cyan', msg))

        end = time.time() - start_pipeline
        if end < 60:
            stopwatch = f' ⏳ Pipeline completed in {end:.1f}s\n'
        else:
            stopwatch = f' ⏳ Pipeline completed in {end/60:.1f}m\n'
        print(fmt.apply('bold', 'lime', stopwatch))

    except Exception as e:
        error(f"Failed to save results: {e}")


if __name__ == "__main__":
    main()
