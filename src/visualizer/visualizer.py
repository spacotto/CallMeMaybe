"""
Handles terminal visualizations for verbose pipeline execution.

This module provides clear, heavily formatted live feedback during
the autoregressive generation loop. It relies on the project's internal
`Formatter` to apply ANSI colors to tracking states, mask sizes, and
final tree renderings.
"""

from typing import Set, Dict, Any
from src.utils import Formatter


class Visualizer:
    """
    Static utility class for rendering terminal UI elements.

    Provides methods to render prompt headers, live-update the carriage
    return with the current CFG state, and neatly print the final
    validated JSON output as an ASCII tree.
    """

    @staticmethod
    def print_prompt_start(idx: int, total: int, prompt: str) -> None:
        """
        Renders the header for a new prompt evaluation.

        Args:
            idx (int): The current prompt number.
            total (int): The total number of prompts in the dataset.
            prompt (str): The natural language query.
        """
        txt = f">>> Prompt [{idx}/{total}]: "
        print("\n" + Formatter.apply('bold', 'yellow', txt) + prompt)
        print(Formatter.apply(None, 'gray', "-" * 70))

    @staticmethod
    def print_status(
        step: int, token_str: str, allowed_chars: Set[str], state_name: str
    ) -> None:
        """
        Handles the live-updating carriage return for token generation.

        This allows the user to watch the extractor's state machine
        (e.g., IN_KEY, EXPECT_COLON) and mask size fluctuate in real-time
        without filling the terminal history with hundreds of lines.

        Args:
            step (int): The current generation step (token index).
            token_str (str): The decoded string representation of the token.
            allowed_chars (Set[str]): Number of valid characters remaining.
            state_name (str): The current rule in the JSON parser state.
        """
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
        """Closes the carriage return line securely with a divider."""
        print()
        print(Formatter.apply(None, 'gray', "-" * 70))

    @staticmethod
    def print_generation_time(gen_time: float, is_valid: bool = True) -> None:
        """
        Displays the elapsed time and phase 3 validation status.

        Args:
            gen_time (float): Seconds taken to generate the JSON.
            is_valid (bool): Whether it passed strict Pydantic checks.
        """
        if is_valid:
            txt = f">>> Valid JSON genrated in {gen_time:.2f}s"
            print(Formatter.apply('bold', 'cyan', txt))
        else:
            txt = f">>> ERROR! Model invalid JSON in {gen_time:.2f}s"
            print(Formatter.apply('bold', 'red', txt))
        print(Formatter.apply(None, 'gray', "-" * 70))

    @staticmethod
    def print_json_render(item: Dict[str, Any]) -> None:
        """
        Renders the final parsed data in a clean ASCII tree structure.

        Args:
            item (Dict[str, Any]): The validated output dictionary.
        """
        print("JSON render")
        prompt = item.get("prompt", "Unknown Prompt")
        print(Formatter.apply(None, 'gray', " ├─ Prompt: ") + prompt)

        # ------------------------------------------------------------------
        # [ERROR CATCHING]: Visualizing Graceful Degradation
        # If the PostProcessor caught an error and injected a fallback
        # object, it attaches an 'error' key. The visualizer flags this
        # clearly in red before halting rendering for this specific item.
        # ------------------------------------------------------------------
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
