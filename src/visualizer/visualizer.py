import json
import os
import argparse
from src.utils import Formatter, error, warning

def render_dashboard(input_path: str) -> None:
    """Renders a colorful terminal dashboard from a constrained decoding JSON output."""
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

    success_count = 0

    for idx, item in enumerate(results):
        prompt = item.get("prompt", "Unknown Prompt")
        call = item.get("function_call", {})

        name = call.get("name", "MISSING_NAME")
        args_dict = call.get("arguments", {})

        print(Formatter.apply('bold', 'blue', f"[{idx + 1:02d}] Prompt: ") + prompt)

        if "error" in call:
            print(Formatter.apply('bold', 'red', f" ├─ ❌ Generation Error: {call['error']}"))
        else:
            success_count += 1
            print(Formatter.apply(None, 'gray', f" ├─ Function : ") + Formatter.apply('bold', 'lime', name))

            if not args_dict:
                print(Formatter.apply(None, 'gray', f" └─ Arguments: ") + Formatter.apply('bold', 'yellow', "{ } (Empty)"))
            else:
                print(Formatter.apply(None, 'gray', f" └─ Arguments: "))
                arg_items = list(args_dict.items())
                for i, (key, val) in enumerate(arg_items):
                    connector = "    └─" if i == len(arg_items) - 1 else "    ├─"
                    print(Formatter.apply(None, 'gray', f"{connector} {key}: ") + Formatter.apply(None, 'white', str(val)))

        print(Formatter.apply(None, 'gray', "-" * 80))

    print("\n" + Formatter.apply('bold', 'cyan', "="*80))
    summary_color = 'lime' if success_count == len(results) else 'yellow'
    print(Formatter.apply('bold', summary_color, f" 📈 SUMMARY: {success_count}/{len(results)} outputs achieved perfect JSON syntax."))
    print(Formatter.apply('bold', 'cyan', "="*80 + "\n"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constrained Decoding Results Visualizer")
    parser.add_argument("--input", type=str, default="data/output/function_calls.json")
    args = parser.parse_args()
    render_dashboard(args.input)
