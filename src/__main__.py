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


def chunk_data(data: List[Any], chunk_size: int):
    """Yields successive n-sized chunks from data to prevent OOM errors."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def calculate_prompt_limit(schema: Dict[str, Any]) -> int:
    """Calculates a safe token limit dynamically based on the schema's shape."""
    params = schema.get("parameters", {})
    if not params:
        return 20

    # Add tokens based on parameter count
    base_tokens = 42
    base_tokens += len(params) * 15

    # Allocate more tokens if string generation is required
    types = [v.get("type") for v in params.values() if isinstance(v, dict)]
    if "string" in types:
        base_tokens += 20

    return min(base_tokens, 150)


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

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        prompts: List[str] = [
            str(item.get("prompt", ""))
            for item in input_data
            if isinstance(item, dict) and "prompt" in item
        ]

        print(fmt.apply(None, 'gray', "-" * 70))
        msg = f">>> Phase 1 & 2: Processing {len(prompts)} prompts..."
        print(fmt.apply('bold', 'yellow', msg))
        print(fmt.apply(None, 'gray', "-" * 70))

        BATCH_SIZE = 32
        target_names: List[str] = []
        json_results: List[str] = []

        generation_start = time.time()

        # Batch processing prevents Memory exhaustion (OOM) on large test sets
        for batch_idx, batch_prompts in enumerate(chunk_data(prompts, BATCH_SIZE)):

            # Phase 1: Classification
            batch_targets = classifier.classify_batch(batch_prompts, functions_schema)
            target_names.extend(batch_targets)

            # Calculate limits dynamically per prompt
            batch_limits = []
            for target_name in batch_targets:
                target_schema = next(
                    (f for f in functions_schema if f["name"] == target_name), {}
                )
                limit = calculate_prompt_limit(target_schema)
                if SchemaParser.is_nested(target_name, functions_schema):
                    limit += 50
                batch_limits.append(limit)

            # Phase 2: Extraction
            batch_results = schema_extractor.extract_batch(
                prompts=batch_prompts,
                function_names=batch_targets,
                functions=functions_schema,
                max_new_tokens_list=batch_limits,
            )
            json_results.extend(batch_results)

        avg_gen_time = (time.time() - generation_start) / max(1, len(prompts))

    except Exception as e:
        error(f"Pipeline Generation failed: {e}")
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

                if args.verbose:
                    Visualizer.print_generation_time(
                        avg_gen_time, is_valid=True
                    )
                    Visualizer.print_json_render(final_item)

            except Exception as item_e:
                error(f"Critical Parsing Error on item {idx+1}: {item_e}")
                results.append({
                    "prompt": prompt,
                    "name": target_name,
                    "parameters": {}
                })

    except Exception as e:
        error(f"Phase 3 (Post-Processing) failed: {e}")

    # --- OUTPUT SAVING ---
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as output_file:
            json.dump(results, output_file, indent=4)

        print(fmt.apply('bold', 'cyan',
                        f"\n\n 💾 All results saved to {args.output}"))

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
