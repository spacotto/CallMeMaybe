from typing import Set, Dict, Any


from src.utils import Formatter


class Visualizer:
    @staticmethod
    def print_prompt_start(idx: int, total: int, prompt: str) -> None:
        """Renders the header for a new prompt evaluation."""
        txt = f">>> Prompt [{idx}/{total}]: "
        print("\n" + Formatter.apply('bold', 'yellow', txt) + prompt)
        print(Formatter.apply(None, 'gray', "-" * 70))

    @staticmethod
    def print_status(
        step: int, token_str: str, allowed_chars: Set[str], state_name: str
    ) -> None:
        """Handles the live-updating carriage return for token generation."""
        clean_token = token_str.replace('\n', '\\n').replace('\r', '\\r')
        line = (
            Formatter.apply('bold', 'blue', f"[{step+1:03d}] ") +
            Formatter.apply(None, 'cyan', "State: ") +
            Formatter.apply('bold', 'white', f"{state_name:<16}") +
            Formatter.apply(None, 'gray', " | ") +
            Formatter.apply(None, 'yellow', "Mask: ") +
            Formatter.apply(
                'bold', 'yellow', f"{len(allowed_chars):02d} allowed chars"
            ) +
            Formatter.apply(None, 'gray', " | ") +
            Formatter.apply(None, 'lime', "Token Generated: ") +
            Formatter.apply('bold', 'lime', f"'{clean_token}'")
        )
        print(f"\r\033[K{line}", end="", flush=True)

    @staticmethod
    def print_div() -> None:
        """Closes the carriage return line securely."""
        print()
        print(Formatter.apply(None, 'gray', "-" * 70))

    @staticmethod
    def print_generation_time(gen_time: float, is_valid: bool = True) -> None:
        """Displays the elapsed time and validation status."""
        if is_valid:
            txt = f">>> Valid JSON genrated in {gen_time:.2f}s"
            print(Formatter.apply('bold', 'cyan', txt))
        else:
            txt = f">>> ERROR! Model invalid JSON in {gen_time:.2f}s"
            print(Formatter.apply('bold', 'red', txt))
        print(Formatter.apply(None, 'gray', "-" * 70))

    @staticmethod
    def print_json_render(item: Dict[str, Any]) -> None:
        """Renders the final parsed data in a clean tree structure."""
        print("JSON render")
        prompt = item.get("prompt", "Unknown Prompt")
        print(Formatter.apply(None, 'gray', " ├─ Prompt: ") + prompt)

        if "error" in item:
            err = item['error']
            print(Formatter.apply('bold', 'red', f" ├─ ❌ Error: {err}"))
            return

        name = item.get("name", "MISSING_NAME")
        args_dict = item.get("parameters", {})

        lbl = Formatter.apply(None, 'gray', " ├─ Name: ")
        val = Formatter.apply('bold', 'lime', name)
        print(lbl + val)

        if not args_dict:
            print(Formatter.apply(None, 'gray', " └─ Parameters: ") +
                  Formatter.apply('bold', 'yellow', "{ } (Empty)"))
        else:
            print(Formatter.apply(None, 'gray', " └─ Parameters: "))
            arg_items = list(args_dict.items())
            for i, (key, val) in enumerate(arg_items):
                conn = "    └─" if i == len(arg_items) - 1 else "    ├─"
                lbl_key = Formatter.apply(None, 'gray', f"{conn} {key}: ")
                val_str = Formatter.apply(None, 'white', str(val))
                print(lbl_key + val_str)
