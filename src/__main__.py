import time
import argparse
import json
import os

from src.engine import FunctionClassifier, ParameterExtractor, PostProcessor
from src.parser import SchemaParser
from src.utils import Formatter, error
from src.visualizer import Visualizer

def main() -> None:

    # --- CLI SETUP ---

    try:
        cli_parser = argparse.ArgumentParser(description="Constrained Decoding LLM Engine")
        cli_parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
        cli_parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
        cli_parser.add_argument("--output", type=str, default="data/output/function_calls.json")
        cli_parser.add_argument("--verbose", action="store_true")
        args = cli_parser.parse_args()

    except Exception as e:
        error(f"CLI Initialization failed: {e}")
        return

    print(Formatter.apply('bold', 'yellow', ">>> Initializing Three-Stage Pipeline..."))
    start_time = time.time()

    # --- MODEL INITIALIZATION ---

    try:
        classifier = FunctionClassifier(model_name="Qwen/Qwen3-0.6B")
        extractor = ParameterExtractor(classifier_instance=classifier)
        elapsed = time.time() - start_time
        print(Formatter.apply('bold', 'cyan', f">>> Engine loaded in {elapsed:.2f} seconds.\n"))

    except Exception as e:
        error(f"Failed to load model architecture: {e}")
        return

    # --- SCHEMA LOADING ---

    try:
        schema_parser = SchemaParser(file_path=args.functions_definition)
        functions_schema = schema_parser.load_functions()

        if not functions_schema:
            error("Execution aborted: No valid functions schema found.")
            return

    except Exception as e:
        error(f"Failed to parse functions schema: {e}")
        return

    # --- INPUT PROMPT LOADING ---

    try:
        if not os.path.exists(args.input):
            error(f"Input file not found at: {args.input}")
            return

        with open(args.input, 'r', encoding='utf-8') as test_file:
            test_cases = json.load(test_file)

        prompts = [test.get("prompt") if isinstance(test, dict) else str(test) for test in test_cases]

    except Exception as e:
        error(f"Failed to load input prompts: {e}")
        return

    results = []
    start = time.time()

    print(Formatter.apply('bold', 'yellow', f">>> Processing batch of {len(prompts)} prompts..."))
    print(Formatter.apply(None, 'gray', "-" * 70))
    generation_start = time.time()

    # --- PHASE 1: CLASSIFICATION (Zero-Shot) ---

    try:
        if args.verbose:
            print(Formatter.apply('bold', 'blue', ">>> Phase 1: Classifying target functions..."))
            print(Formatter.apply(None, 'gray', "-" * 70))

        classified_names = classifier.classify_batch(
            prompts=prompts,
            functions=functions_schema,
            max_new_tokens=30
        )

    except Exception as e:
        error(f"Phase 1 (Classification) failed catastrophically: {e}")
        return

    # --- PHASE 2: PARAMETER EXTRACTION (Few-Shot/Deep Parse) ---

    try:
        if args.verbose:
            print(Formatter.apply('bold', 'blue', ">>> Phase 2: Extracting nested parameters..."))
            print(Formatter.apply(None, 'gray', "-" * 70))

        batch_results = extractor.extract_batch(
            prompts=prompts,
            function_names=classified_names,
            functions=functions_schema,
            max_new_tokens=120,
            verbose=args.verbose
        )

    except Exception as e:
        error(f"Phase 2 (Extraction) failed catastrophically: {e}")
        return

    gen_time = time.time() - generation_start
    avg_gen_time = gen_time / len(prompts) if len(prompts) > 0 else 0

    # --- PHASE 3: POST-PROCESSING & VALIDATION ---

    try:
        print(Formatter.apply('bold', 'blue', ">>> Phase 3: Post-processing & Validation..."))
        print(Formatter.apply(None, 'gray', "-" * 70))

        for idx, (prompt, target_name, json_result_str) in enumerate(zip(prompts, classified_names, batch_results)):
            # Item-Level Try/Except: Prevents a single corrupted generation from crashing the entire batch loop

            try:
                Visualizer.print_prompt_start(idx + 1, len(prompts), prompt)

                final_item = PostProcessor.process_result(prompt, target_name, json_result_str, functions_schema)
                results.append(final_item)

                is_valid_json = "error" not in final_item
                if not is_valid_json:
                    error(f"Model generated invalid JSON for prompt {idx+1}")

                Visualizer.print_generation_time(avg_gen_time, is_valid=is_valid_json)
                Visualizer.print_json_render(final_item)

            except Exception as item_e:
                error(f"Critical Parsing Error on item {idx+1}: {item_e}")
                results.append({"prompt": prompt, "error": f"Internal Processing Error: {item_e}", "raw": json_result_str})

    except Exception as e:
        error(f"Phase 3 (Post-Processing loop) failed: {e}")

    # --- OUTPUT SAVING ---

    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as output_file:
            json.dump(results, output_file, indent=4)

        txt = f"\n 💾 All results successfully saved to {args.output}"
        print(Formatter.apply('bold', 'cyan', txt))
        final_time = time.time() - start
        stopwatch = f' ⏳ Pipeline execution completed in {final_time / 60:.1f}m\n'
        print(Formatter.apply('bold', 'lime', stopwatch))

    except Exception as e:
        error(f"Failed to save results to {args.output}: {e}")

if __name__ == "__main__":
    main()
