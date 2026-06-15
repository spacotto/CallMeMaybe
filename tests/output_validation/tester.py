import os
import json
import shutil
import pytest
from src.utils import Formatter as clr
from src.utils import error as err

class OutputValidationTester:
    def __init__(self, test_input_dir: str, test_output_dir: str) -> None:
        # Centralized directories passed from the Orchestrator
        self.test_input_dir = test_input_dir
        self.test_output_dir = test_output_dir

        self.module_dir = os.path.dirname(os.path.abspath(__file__))

        # Route mock data into the centralized data/output/tests/ folder
        self.mock_data_dir = os.path.join(self.test_output_dir, "output_validation_mock_data")
        self.test_file = os.path.join(self.module_dir, "test_outputs.py")
        self.mock_schema_path = os.path.join(self.mock_data_dir, "mock_schema.json")

    def _setup_mock_files(self) -> None:
        """Creates a temporary folder with a mock schema and invalid output JSON files."""
        print(clr.apply('bold', 'yellow', ">>> Generating mock output test files..."))
        os.makedirs(self.mock_data_dir, exist_ok=True)

        # Export the path so pytest knows where to look for the generated files
        os.environ["OUTPUT_VALIDATOR_MOCK_DIR"] = self.mock_data_dir

        # 1. Create a valid mock schema to test against
        mock_schema = [{
            "name": "fn_test",
            "parameters": {
                "user_id": {"type": "number"},
                "message": {"type": "string"}
            }
        }]
        with open(self.mock_schema_path, "w", encoding="utf-8") as f:
            json.dump(mock_schema, f)

        # 2. Invalid Output: Extra Keys (Forbidden by Moulinette)
        with open(os.path.join(self.mock_data_dir, "extra_keys.json"), "w", encoding="utf-8") as f:
            json.dump([{
                "prompt": "test", "name": "fn_test", "parameters": {"user_id": 1, "message": "hi"},
                "forbidden_key": "this should fail"
            }], f)

        # 3. Invalid Output: Wrong Parameter Type (String instead of Number)
        with open(os.path.join(self.mock_data_dir, "wrong_type.json"), "w", encoding="utf-8") as f:
            json.dump([{
                "prompt": "test", "name": "fn_test",
                "parameters": {"user_id": "ONE", "message": "hi"}
            }], f)

        # 4. Invalid Output: Missing Required Parameter
        with open(os.path.join(self.mock_data_dir, "missing_param.json"), "w", encoding="utf-8") as f:
            json.dump([{
                "prompt": "test", "name": "fn_test",
                "parameters": {"message": "hi"}
            }], f)

    def _cleanup(self) -> None:
        """Removes the temporary test folder after execution."""
        if os.path.exists(self.mock_data_dir):
            shutil.rmtree(self.mock_data_dir)

    def run(self) -> None:
        """Main entry point called by the Orchestrator."""
        try:
            self._setup_mock_files()
            exit_code = pytest.main(["-v", self.test_file])

            if exit_code == pytest.ExitCode.OK:
                print(clr.apply('bold', 'cyan', ">>> SUCCESS! All output validation tests passed!"))
            else:
                err(f"\n  [FAILED] Pytest returned exit code: {exit_code}")

        except Exception as e:
            err(f"  Fatal error during output validation testing: {e}")
#        finally:
#            self._cleanup()
