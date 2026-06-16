"""
Main orchestration module for the constrained decoding pipeline.

This module serves as the entry point for the function calling engine.
It manages the ingestion of CLI arguments, handles dynamic environment
variable configurations (such as `LLM_MODEL_NAME`), and orchestrates
the batching and execution phases of the constrained decoding process.
"""

# ----------------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------------

import time
import argparse
import json
import os
from typing import List, Dict, Any, Iterator

from src.engine import (
    FunctionClassifier,
    SchemaExtractor,
    PostProcessor
)
from src.parser import SchemaParser
from src.utils import Formatter as fmt
from src.utils import error
from src.visualizer import Visualizer


# ----------------------------------------------------------------------------
#  Helper functions
# ----------------------------------------------------------------------------

def chunk_data(data: List[str], batch_size: int) -> Iterator[List[str]]:
    """
    Yields successive batches from a list of strings.

    # ------------------------------------------------------------------
    # [BATCHING MECHANISM]: Dynamic Memory Protection
    # This generator safely slices the dataset. Pushing 10,000 prompts
    # into the model simultaneously will cause a fatal CUDA Out-Of-Memory
    # (OOM) exception. Batching bounds the VRAM footprint to a safe,
    # constant size regardless of the total input file length.
    # ------------------------------------------------------------------

    Args:
        data (List[str]): The complete list of prompts.
        batch_size (int): The maximum number of elements per batch.

    Yields:
        Iterator[List[str]]: A single batched slice of the original data.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def calculate_prompt_limit(schema: Dict[str, Any]) -> int:
    """
    Calculates a safe token limit dynamically based on the schema's shape.

    By analyzing the parameter count and types (specifically looking for
    unbounded strings), the engine assigns a strict cut-off point to
    prevent the LLM from hallucinating endless loops.

    Args:
        schema (Dict[str, Any]): The target function's parameter schema.

    Returns:
        int: The maximum number of tokens allowed for this specific prompt.
    """
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


# ----------------------------------------------------------------------------
#  Entry point
# ----------------------------------------------------------------------------

def main() -> None:
    """
    Executes the primary constrained decoding pipeline.

    Initializes the argument parser to retrieve file paths, validates
    the environment variables, initializes the engine phases, and drives
    the text data through Classification, Extraction, and Validation.
    """

    # ----------------------------------------------------------------------
    # [ERROR CATCHING]: CLI Initialization
    # Prevents malformed arguments from causing an unhandled crash before
    # the engine even begins to spin up.
    # ----------------------------------------------------------------------
    try:
        cli_parser = argparse.ArgumentParser(
            description="Constrained Decoding LLM Engine"
        )
        cli_parser.add_argument(
            "--functions_definition",
            type=str,
            required=True,
            help="Path to the functions definition JSON file (Required)"
        )
        cli_parser.add_argument(
            "--input",
            type=str,
            default="data/input/function_calling_tests.json",
            help="Path to the input prompts JSON file (Optional)"
        )
        cli_parser.add_argument(
            "--output",
            type=str,
            default="data/output/function_calling_results.json",
            help="Path to save the generated JSON results (Optional)"
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
    model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-0.6B")

    # ----------------------------------------------------------------------
    # [ERROR CATCHING]: Model Loading
    # Catches missing weights, incorrect paths, or architecture mismatches
    # gracefully rather than throwing raw device-side tracebacks.
    # ----------------------------------------------------------------------
    try:
        classifier = FunctionClassifier(model_name=model_name)
        schema_extractor = SchemaExtractor(classifier_instance=classifier)

        init = time.time() - start_init
        print(fmt.apply(
            'bold',
            'cyan',
            f">>> Engine loaded in {init:.2f} seconds."
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

    # ----------------------------------------------------------------------
    # [ERROR CATCHING]: Main Pipeline Execution Loop
    # Wraps the entire generation sequence to ensure that if a fatal OOM
    # or dataset corruption occurs, the pipeline fails cleanly without
    # leaving orphaned processes.
    # ----------------------------------------------------------------------
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        prompts: List[str] = [
            str(item.get("prompt", ""))
            for item in input_data
            if isinstance(item, dict) and "prompt" in item
        ]

        print(fmt.apply('bold', 'yellow',
                        f">>> Processing {len(prompts)} prompts..."))

        BATCH_SIZE = 1 if args.verbose else 32
        results: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # [BATCHING MECHANISM]: Loop Orchestration
        # Drives the sequential batches through Phase 1 and Phase 2. If
        # verbose mode is on, it forces a batch size of 1 to allow the
        # visualizer to cleanly print the state machine to the terminal.
        # ------------------------------------------------------------------
        for batch_idx, batch_prompts in enumerate(
            chunk_data(prompts, BATCH_SIZE)
        ):
            if args.verbose and BATCH_SIZE != 1:
                print(fmt.apply(
                    'bold', 'cyan', f"Processing Batch {batch_idx + 1}..."
                ))

            # Phase 1: Classification
            batch_targets = classifier.classify_batch(
                batch_prompts, functions_schema
            )

            batch_limits = []
            for target_name in batch_targets:
                target_schema = next(
                    (f for f in functions_schema
                     if f["name"] == target_name), {}
                )
                limit = calculate_prompt_limit(target_schema)
                if SchemaParser.is_nested(target_name, functions_schema):
                    limit += 50
                batch_limits.append(limit)

            if args.verbose and BATCH_SIZE == 1:
                abs_idx = len(results) + 1
                Visualizer.print_prompt_start(
                    abs_idx, len(prompts), batch_prompts[0]
                )

            # Phase 2: Extraction
            t0 = time.time()
            batch_results = schema_extractor.extract_batch(
                prompts=batch_prompts,
                function_names=batch_targets,
                functions=functions_schema,
                max_new_tokens_list=batch_limits,
                verbose=args.verbose
            )
            batch_time = time.time() - t0

            # Phase 3: Immediate Post-Processing & Validation
            for idx_in_batch, result_tuple in enumerate(
                zip(batch_prompts, batch_targets, batch_results)
            ):
                prompt, target_name, json_result_str = result_tuple

                # ----------------------------------------------------------
                # [ERROR CATCHING]: Item-Level Degradation
                # If a specific prompt fails validation, we catch it here,
                # log it, inject an empty safe parameter block, and ALLOW
                # the batch to continue. One bad prompt will not ruin the
                # entire JSON output file.
                # ----------------------------------------------------------
                try:
                    final_item = PostProcessor.process_result(
                        prompt,
                        target_name,
                        json_result_str,
                        functions_schema
                    )
                    results.append(final_item)

                    if args.verbose and BATCH_SIZE == 1:
                        Visualizer.print_generation_time(
                            batch_time, is_valid=True
                        )
                        Visualizer.print_json_render(final_item)

                except Exception as item_e:
                    error(
                        f"Critical Parsing Error on item {len(results) + 1}: "
                        f"{item_e}"
                    )
                    # Strictly compliant fallback
                    fallback = {
                        "prompt": prompt,
                        "name": target_name,
                        "parameters": {}
                    }
                    results.append(fallback)

                    if args.verbose and BATCH_SIZE == 1:
                        Visualizer.print_generation_time(
                            batch_time, is_valid=False
                        )
                        Visualizer.print_json_render(fallback)

    except Exception as e:
        error(f"Pipeline Execution failed: {e}")
        return

    # ----------------------------------------------------------------------
    # [ERROR CATCHING]: Output Saving
    # Safely attempts to create missing directories and write the final
    # JSON. Caught specifically to prevent losing hours of generation
    # due to a simple permissions error.
    # ----------------------------------------------------------------------
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
