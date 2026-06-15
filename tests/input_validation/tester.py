import os
import json
import shutil
import pytest
from src.utils import Formatter as clr
from src.utils import error as err

class InputValidationTester:
    def __init__(self, test_input_dir: str, test_output_dir: str) -> None:
        self.test_input_dir = test_input_dir
        self.test_output_dir = test_output_dir
        self.test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_inputs.py")

        # Route invalid mock data into the centralized data/input/tests/ folder
        self.invalid_data_dir = os.path.join(self.test_input_dir, "input_validation_mock_data")

    def _setup_invalid_files(self) -> None:
        """Creates a temporary folder with malformed and invalid JSON files."""
        print(clr.apply('bold', 'yellow', ">>> Generating invalid test files..."))
        os.makedirs(self.invalid_data_dir, exist_ok=True)

        # Export the path so pytest knows where to look for the generated files
        os.environ["INPUT_VALIDATOR_MOCK_DIR"] = self.invalid_data_dir

        # 1. Malformed JSON (Syntax Error)
        with open(os.path.join(self.invalid_data_dir, "malformed.json"), "w", encoding="utf-8") as f:
            f.write("[\n  {\"name\": \"fn_test\",\n  \"parameters\": {} \n]") # Missing closing brace

        # 2. Wrong Root Type (Dict instead of List)
        with open(os.path.join(self.invalid_data_dir, "wrong_root.json"), "w", encoding="utf-8") as f:
            json.dump({"functions": [{"name": "fn_test"}]}, f)

        # 3. Empty File
        with open(os.path.join(self.invalid_data_dir, "empty.json"), "w", encoding="utf-8") as f:
            f.write("")

    def _cleanup(self) -> None:
        """Removes the temporary test folder after execution."""
        if os.path.exists(self.invalid_data_dir):
            shutil.rmtree(self.invalid_data_dir)

    def run(self) -> None:
        """Main entry point called by the Orchestrator."""
        try:
            self._setup_invalid_files()

            # Programmatically run pytest on the specific test file
            exit_code = pytest.main(["-v", self.test_file])

            if exit_code == pytest.ExitCode.OK:
                print(clr.apply('bold', 'cyan', ">>> SUCCESS! All input validation tests passed!"))
            else:
                err(f"\n [FAILED] Pytest returned exit code: {exit_code}")

        except Exception as e:
            err(f"  Fatal error during input validation testing: {e}")

#       finally:
#           Ensure cleanup happens even if tests crash
#           self._cleanup()
