import json
import os
import argparse
from typing import Set, Dict, Any, List
from src.utils import Formatter, error

class Visualizer:
    @staticmethod
    def print_prompt_start(idx: int, total: int, prompt: str) -> None:
        """Renders the header for a new prompt evaluation."""
        txt = f">>> Processing Prompt [{idx}/{total}]: "
        print("\n" + Formatter.apply('bold', 'yellow', txt) + prompt)
        print(Formatter.apply(None, 'gray', "-" * 70))

    @staticmethod
    def print_step(step: int, token_str: str, allowed_chars: Set[str], state_name: str) -> None:
        """Handles the live-updating carriage return for token generation."""
        clean_token = token_str.replace('\n', '\\n').replace('\r', '\\r')
        line = (
            Formatter.apply('bold', 'blue', f"[{step+1:03d}] ") +
            Formatter.apply(None, 'cyan', f"State: ") +
            Formatter.apply('bold', 'white', f"{state_name:<16}") +
            Formatter.apply(None, 'gray', " | ") +
            Formatter.apply(None, 'yellow', f"Mask: ") +
            Formatter.apply('bold', 'yellow', f"{len(allowed_chars):02d} allowed chars") +
            Formatter.apply(None, 'gray', " | ") +
            Formatter.apply(None, 'lime', f"Token Generated: ") +
            Formatter.apply('bold', 'lime', f"'{clean_token}'")
        )
        print(f"\r\033[K{line}", end="", flush=True)

    @staticmethod
    def print_step_complete() -> None:
        """Closes the carriage return line securely."""
        print() # Move past the \r line
        print(Formatter.apply(None, 'gray', "-" * 70))

    @staticmethod
    def print_generation_time(gen_time: float, is_valid: bool = True) -> None:
        """Displays the elapsed time and validation status."""
        if is_valid:
            txt = f">>> Valid JSON genrated in {gen_time:.2f}s"
            print(Formatter.apply('bold', 'cyan', txt))
        else:
            txt = f">>> ERROR! Model generated invalid JSON in {gen_time:.2f}s"
            print(Formatter.apply('bold', 'red', txt))
        print(Formatter.apply(None, 'gray', "-" * 70))

    @staticmethod
    def print_json_render(item: Dict[str, Any]) -> None:
        """Renders the final parsed data in a clean tree structure."""
        print("JSON render")
        prompt = item.get("prompt", "Unknown Prompt")
        print(Formatter.apply(None, 'gray', f" ├─ Prompt: ") + prompt)

        if "error" in item:
            print(Formatter.apply('bold', 'red', f" ├─ ❌ Generation Error: {item['error']}"))
            return

        name = item.get("name", "MISSING_NAME")
        args_dict = item.get("parameters", {})

        print(Formatter.apply(None, 'gray', f" ├─ Name: ") + Formatter.apply('bold', 'lime', name))

        if not args_dict:
            print(Formatter.apply(None, 'gray', f" └─ Parameters: ") + Formatter.apply('bold', 'yellow', "{ } (Empty)"))
        else:
            print(Formatter.apply(None, 'gray', f" └─ Parameters: "))
            arg_items = list(args_dict.items())
            for i, (key, val) in enumerate(arg_items):
                connector = "    └─" if i == len(arg_items) - 1 else "    ├─"
                print(Formatter.apply(None, 'gray', f"{connector} {key}: ") + Formatter.apply(None, 'white', str(val)))


# -------------------------------------------------------------------------
# Standalone execution support for rendering existing JSON files
# -------------------------------------------------------------------------
def render_dashboard(input_path: str) -> None:
    if not os.path.exists(input_path):
        error(f"Results file not found at: {input_path}. Run the engine first!")
        return
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        error(f"Failed to load results JSON: {e}")
        return

    print(Formatter.apply('bold', 'white', "\n" + "="*80))
    print(Formatter.apply('bold', 'cyan', " JSON OUTPUT VISUALISATION"))
    print(Formatter.apply('bold', 'white', "="*80 + "\n"))

    for item in results:
        Visualizer.print_json_render(item)
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constrained Decoding Results Visualizer")
    parser.add_argument("--input", type=str, default="data/output/function_calls.json")
    args = parser.parse_args()
    render_dashboard(args.input)
