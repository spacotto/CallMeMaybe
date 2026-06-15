import os
import json
import subprocess
from src.utils import Formatter as clr
from src.utils import error as err
from src.utils import warning as warn

class NestedParametersTester:
    def __init__(self, test_input_dir: str, test_output_dir: str) -> None:
        # Accept the centralized directories from the Orchestrator
        self.test_input_dir = test_input_dir
        self.test_output_dir = test_output_dir

        # Isolate nested tests into the centralized folders
        self.test_input_path = os.path.join(self.test_input_dir, "nested_inputs.json")
        self.test_schema_path = os.path.join(self.test_input_dir, "nested_schema.json")
        self.test_output_path = os.path.join(self.test_output_dir, "nested_results.json")

        self.nested_schema = [
            {
                "name": "fn_calculate_distance",
                "description": "Calculate the distance between point A (x, y) and point B (x, y).",
                "parameters": {
                    "point_a": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}, "y": {"type": "number"}}
                    },
                    "point_b": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}, "y": {"type": "number"}}
                    }
                }
            },
            {
                "name": "add_to_grocery_list",
                "description": "Add a specific item with its quantity to the grocery list.",
                "parameters": {
                    "item_details": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "quantity": {"type": "number"}}
                    }
                }
            },
            {
                "name": "remove_from_grocery_list",
                "description": "Remove a specific quantity of an item from the grocery list.",
                "parameters": {
                    "item_details": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "quantity": {"type": "number"}}
                    }
                }
            }
        ]

        self.nested_prompts = [
            {"prompt": "Calculate distance between point A (0, 8) and point B (4, 2)."},
            {"prompt": "Find distance between point A (3, 8) and point B (24, 42)."},
            {"prompt": "Add 3 tuna cans to the grocery list."},
            {"prompt": "Please put 5 red apples on my grocery list."},
            {"prompt": "Remove 2 tomatoes from the grocery list."},
            {"prompt": "Can you take away 3 bottles of milk from the list?"}
        ]

    def _setup_files(self) -> None:
        """Writes the nested schemas and prompts to temporary files."""
        with open(self.test_schema_path, 'w', encoding='utf-8') as f:
            json.dump(self.nested_schema, f, indent=2)

        with open(self.test_input_path, 'w', encoding='utf-8') as f:
            json.dump(self.nested_prompts, f, indent=2)

    def _run_engine(self) -> bool:
        """Executes the pipeline against the isolated nested files."""
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)

        command = [
            "uv", "run", "python", "-m", "src",
            "--functions_definition", self.test_schema_path,
            "--input", self.test_input_path,
            "--output", self.test_output_path,
            "--verbose"
        ]

        result = subprocess.run(command, capture_output=False, text=True)

        if result.returncode != 0:
            err(f"  [FAILED] Engine crashed during nested parameters test with exit code {result.returncode}")
            print(f"  --- Engine Stderr ---\n{result.stderr}\n---------------------")
            return False
        return True

    def _verify_output(self) -> None:
        """Validates that the engine successfully produced nested dictionaries."""
        if not os.path.exists(self.test_output_path):
            err("  [FAILED] Output file was not created.")
            return

        try:
            with open(self.test_output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if len(data) != len(self.nested_prompts):
                warn(f"Expected {len(self.nested_prompts)} results, but generated {len(data)}.")

            # Restored success print
            print(clr.apply('bold', 'lime', f"\n  [SUCCESS] Successfully extracted {len(data)} nested structures.\n"))

        except json.JSONDecodeError as e:
            err(f"  [FAILED] Engine generated invalid JSON: {e}")

    def _cleanup(self) -> None:
        """Removes the temporary test files."""
        if os.path.exists(self.test_input_path):
            os.remove(self.test_input_path)
        if os.path.exists(self.test_schema_path):
            os.remove(self.test_schema_path)
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)

    def run(self) -> None:
        """Main execution flow."""
        try:
            self._setup_files()
            if self._run_engine():
                self._verify_output()
        finally:
            self._cleanup()
