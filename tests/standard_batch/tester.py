import os
import json
import subprocess
from src.utils import Formatter as clr
from src.utils import error as err

class StandardBatchTester:
    def __init__(self) -> None:
        # Standard paths used by the main engine
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.input_dir = os.path.join(self.root_dir, "data", "input")
        self.output_dir = os.path.join(self.root_dir, "data", "output")

        self.input_file = os.path.join(self.input_dir, "function_calling_tests.json")
        self.schema_file = os.path.join(self.input_dir, "functions_definition.json")
        self.output_file = os.path.join(self.output_dir, "function_calling_results.json")

        # Ground truth file (if it exists) to compare against
        self.expected_output_file = os.path.join(self.root_dir, "function_calls.json")

    def _setup_files(self) -> None:
        """Ensures the standard input directories and files exist, creating them if necessary."""
        os.makedirs(self.input_dir, exist_ok=True)

        # 1. Reconstruct functions_definition.json if missing
        if not os.path.exists(self.schema_file):
            print(clr.apply('bold', 'yellow', f">>> Recreating missing {os.path.basename(self.schema_file)}..."))
            schema_data = [
                {
                    "name": "fn_add_numbers",
                    "description": "Add two numbers together and return their sum.",
                    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "returns": {"type": "number"}
                },
                {
                    "name": "fn_greet",
                    "description": "Generate a greeting message for a person by name.",
                    "parameters": {"name": {"type": "string"}},
                    "returns": {"type": "string"}
                },
                {
                    "name": "fn_reverse_string",
                    "description": "Reverse a string and return the reversed result.",
                    "parameters": {"s": {"type": "string"}},
                    "returns": {"type": "string"}
                },
                {
                    "name": "fn_get_square_root",
                    "description": "Calculate the square root of a number.",
                    "parameters": {"a": {"type": "number"}},
                    "returns": {"type": "number"}
                },
                {
                    "name": "fn_substitute_string_with_regex",
                    "description": "Replace all occurrences matching a regex pattern in a string.",
                    "parameters": {
                        "source_string": {"type": "string"},
                        "regex": {"type": "string"},
                        "replacement": {"type": "string"}
                    },
                    "returns": {"type": "string"}
                }
            ]
            with open(self.schema_file, 'w', encoding='utf-8') as f:
                json.dump(schema_data, f, indent=2)

        # 2. Reconstruct function_calling_tests.json if missing
        if not os.path.exists(self.input_file):
            print(clr.apply('bold', 'cyan', f"  -> Recreating missing {os.path.basename(self.input_file)}..."))
            tests_data = [
                {"prompt": "What is the sum of 2 and 3?"},
                {"prompt": "What is the sum of 265 and 345?"},
                {"prompt": "Greet shrek"},
                {"prompt": "Greet john"},
                {"prompt": "Reverse the string 'hello'"},
                {"prompt": "Reverse the string 'world'"},
                {"prompt": "What is the square root of 16?"},
                {"prompt": "Calculate the square root of 144"},
                {"prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"},
                {"prompt": "Replace all vowels in 'Programming is fun' with asterisks"},
                {"prompt": "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'"}
            ]
            with open(self.input_file, 'w', encoding='utf-8') as f:
                json.dump(tests_data, f, indent=2)

    def _run_engine(self) -> bool:
        """Executes the main LLM pipeline."""

        # Ensure output directory exists and is clean
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        command = [
            "uv", "run", "python", "-m", "src",
            "--functions_definition", self.schema_file,
            "--input", self.input_file,
            "--output", self.output_file,
            "--verbose"
            ]

        result = subprocess.run(command, capture_output=False, text=True)

        if result.returncode != 0:
            err(f"  [FAILED] Engine crashed with exit code {result.returncode}")
            print(f"  --- Engine Stderr ---\n{result.stderr}\n---------------------")
            return False

        return True

    def _verify_output(self) -> None:
        """Verifies the generated JSON against the expected standard."""
        print(clr.apply('bold', 'yellow', ">>> Verifying generated output..."))

        if not os.path.exists(self.output_file):
            err("  [FAILED] Engine completed but output file was not created.")
            return

        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                generated_data = json.load(f)

            if not isinstance(generated_data, list):
                err("  [FAILED] Generated output is not a JSON list.")
                return

            # If we have the ground truth file, do a strict count comparison
            if os.path.exists(self.expected_output_file):
                with open(self.expected_output_file, 'r', encoding='utf-8') as f:
                    expected_data = json.load(f)

                if len(generated_data) != len(expected_data):
                    err(f"  [WARNING] Expected {len(expected_data)} items, got {len(generated_data)}.")

            # Verify schema shape (quick sanity check)
            malformed_count = 0
            for item in generated_data:
                if not all(k in item for k in ("prompt", "name", "parameters")):
                    malformed_count += 1

            if malformed_count > 0:
                err(f"  [FAILED] {malformed_count} items missing required keys.")
            else:
                print(clr.apply('bold', 'cyan', ">>> SUCCESS! Standard prompt batch executed flawlessly!"))

        except json.JSONDecodeError as e:
            err(f"  [FAILED] Generated file is not valid JSON: {e}")

    def run(self) -> None:
        """Main entry point."""

        # Safely create prerequisites before running
        self._setup_files()

        if self._run_engine():
            self._verify_output()
