import subprocess
from src.utils import Formatter as clr
from src.utils import error as err

# --- Import specialised tester class
from tests.input_validation.tester import InputValidationTester
from tests.output_validation.tester import OutputValidationTester
from tests.standard_batch.tester import StandardBatchTester
from tests.edge_cases.tester import EdgeCaseTester
from tests.nested_parameters.tester import NestedParametersTester


class CallMeMaybeTester:
    def __init__(self) -> None:
        self.input_tester = InputValidationTester()
        self.output_tester = OutputValidationTester()
        self.standard_batch_tester = StandardBatchTester()
        self.edge_case_tester = EdgeCaseTester()
        self.nested_tester = NestedParametersTester()

    # --- Dispatch Methods ---
    def _test_input_handling(self) -> None:
        print(clr.apply('bold', 'yellow', ">>> Running Input Validation..."))
        self.input_tester.run()

    def _test_output_validation(self) -> None:
        print(clr.apply('bold', 'yellow', ">>> Running Output Validation..."))
        self.output_tester.run()

    def _test_standard_batch(self) -> None:
        print(clr.apply('bold', 'yellow', ">>> Running Standard Prompt Batch..."))
        self.standard_batch_tester.run()

    def _test_edge_cases(self) -> None:
        print(clr.apply('bold', 'yellow', ">>> Running Edge Cases Prompt Batch..."))
        self.edge_case_tester.run()

    def _test_nested_parameters(self) -> None:
        print(clr.apply('bold', 'yellow', ">>> Running Nested Parameters Prompt Batch..."))
        self.nested_tester.run()

    # --- Orchestrator core
    def run(self) -> None:
        print('\n' + ' ' + '=' * 60)
        print(clr.apply('bold', 'white', '  Call Me Maybe: Test Suite'))
        print(' ' + '=' * 60)

        # Dictionary structure: { Key: ['Description', self.method_reference] }
        options = {
            1: ['Test input validation', self._test_input_handling],
            2: ['Test output validation', self._test_output_validation],
            3: ['Test standard prompt batch', self._test_standard_batch],
            4: ['Test edge cases prompt batch', self._test_edge_cases],
            5: ['Test nested parameters prompt batch', self._test_nested_parameters]
        }

        print(clr.apply('bold', 'white', f'  {"n.":<3}Description'))
        print(' ' + '-' * 60)

        for k, v in options.items():
            print(f'  {k:<3}{v[0]}')

        print(' ' + '-' * 60)
        raw = input(clr.apply('bold', 'white', f'  Pick an option: '))
        print()

        try:
            choice = int(raw.strip())
            print()

            if choice in options:
                target_method = options[choice][1]
                target_method()

            else:
                 err(' Invalid option.\n')

            print()

        except ValueError:
            err(' Please enter a valid number.\n')
        except subprocess.CalledProcessError as e:
            err(f' The pipeline crashed or returned an error code: {e}\n')
        except Exception as e:
            err(f' An unexpected error occurred: {e}\n')


if __name__ == "__main__":
    tester = CallMeMaybeTester()
    tester.run()
