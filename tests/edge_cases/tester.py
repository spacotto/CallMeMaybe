import os
import json
import subprocess
from typing import List, Dict, Any
from src.utils import Formatter as clr
from src.utils import error as err

class EdgeCaseTester:
    def __init__(self, test_input_dir: str, test_output_dir: str) -> None:
        self.test_input_dir = test_input_dir
        self.test_output_dir = test_output_dir
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Isolate edge case tests so they don't overwrite standard batch data
        # Now pointing to the centralized data/input/tests and data/output/tests directories
        self.test_input_path = os.path.join(self.test_input_dir, "edge_case_inputs.json")
        self.test_output_path = os.path.join(self.test_output_dir, "edge_case_results.json")
        self.schema_file = os.path.join(self.root_dir, "data", "input", "functions_definition.json")

        # The adversarial prompt pool
        self.edge_cases = {
            "fn_add_numbers": [
                {"prompt": "What is the sum of -9999999 and 9999999?"},   # Extreme negatives/positives
                {"prompt": "Add 0.000000001 to 3.1415926535"},            # Extreme float precision
                {"prompt": "Can you add two and two together?"}           # Written numerals
            ],
            "fn_greet": [
                {"prompt": "Greet ''"},                                   # Empty string
                {"prompt": "Say hello to Mr. O'Connor-Smith_99!"},        # Special characters
                {"prompt": "Greet 1337"}                                  # Number masquerading as a string
            ],
            "fn_reverse_string": [
                {"prompt": "Reverse this string: !@#$%^&*()_+"},          # Pure symbols
                {"prompt": "Reverse the string '  spaces  '"},            # Trailing/leading whitespace
                {"prompt": "Reverse the string ''"}                       # Empty string extraction
            ],
            "fn_get_square_root": [
                {"prompt": "Calculate the square root of 0"},             # Zero boundary
                {"prompt": "What is the square root of -100?"},           # Negative boundary (tests LLM logic extraction)
                {"prompt": "Find the square root of 9999999999.99"}       # Large floats
            ],
            "fn_substitute_string_with_regex": [
                {"prompt": "Replace all '\\\\' with '/' in 'C:\\\\path\\\\file'"}, # Escaping hell
                {"prompt": "Substitute '' with 'X' in 'Nothing'"},                 # Empty regex pattern
                {"prompt": "In the text '12345', switch the word '\\d' to '*'"}    # Literal string vs Regex syntax conflict
            ]
        }

    def _select_target(self) -> List[Dict[str, str]]:
        """Displays UI to select which function to stress test."""
        functions = list(self.edge_cases.keys())
        print('\n' + ' ' + '=' * 60)
        print(clr.apply('bold', 'white', "  Edge Case Selection"))
        print(' ' + '=' * 60)

        print(clr.apply('bold', 'white', f'  {"n.":<3}Description'))
        print(' ' + '-' * 60)

        print(f"  {'0':<3}All Functions (Full Stress Test)")
        for i, func in enumerate(functions, 1):
            print(f'  {i:<3}{func}')

        print(' ' + '-' * 60)
        raw = input(clr.apply('bold', 'white', "  Select a target to test: "))

        try:
            choice = int(raw.strip())
            if choice == 0:
                print(clr.apply('bold', 'yellow', ">>> Compiling ALL edge cases..."))
                all_tests = []
                for cases in self.edge_cases.values():
                    all_tests.extend(cases)
                return all_tests
            elif 1 <= choice <= len(functions):
                target_func = functions[choice - 1]
                print(clr.apply('bold', 'yellow', f">>> Compiling edge cases for {target_func}..."))
                return self.edge_cases[target_func]
            else:
                err("  Invalid selection.")
                return []
        except ValueError:
            err("  Please enter a valid number.")
            return []

    def _setup_files(self, test_prompts: List[Dict[str, str]]) -> None:
        """Writes the selected edge cases to a temporary input file."""
        with open(self.test_input_path, 'w', encoding='utf-8') as f:
            json.dump(test_prompts, f, indent=2)

    def _run_engine(self) -> bool:
        """Executes the pipeline against the isolated edge case files."""
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)

        command = [
            "uv", "run", "python", "-m", "src",
            "--functions_definition", self.schema_file,
            "--input", self.test_input_path,
            "--output", self.test_output_path,
            "--verbose"
        ]

        result = subprocess.run(command, capture_output=False, text=True)

        if result.returncode != 0:
            err(f"  [FAILED] Engine crashed during edge cases with exit code {result.returncode}")
            print(f"  --- Engine Stderr ---\n{result.stderr}\n---------------------")
            return False
        return True

    def _verify_output(self, expected_count: int) -> None:
        """Validates that the engine didn't crash and produced parseable JSON."""
        if not os.path.exists(self.test_output_path):
            err("  [FAILED] Output file was not created. The engine likely silently crashed.")
            return

        try:
            with open(self.test_output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if len(data) != expected_count:
                err(f"  [WARNING] Expected {expected_count} results, but generated {len(data)}.")
            else:
                print(clr.apply('bold', 'lime', f"\n  [SUCCESS] Successfully extracted {len(data)} edge cases.\n"))

        except json.JSONDecodeError as e:
            err(f"  [FAILED] Engine generated invalid JSON under edge case stress: {e}")

    def _cleanup(self) -> None:
        """Removes the temporary test files."""
        if os.path.exists(self.test_input_path):
            os.remove(self.test_input_path)
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)

    def run(self) -> None:
        """Main execution flow."""

        test_prompts = self._select_target()
        if not test_prompts:
            return

        try:
            self._setup_files(test_prompts)
            if self._run_engine():
                self._verify_output(len(test_prompts))
        finally:
            self._cleanup()
