import subprocess
from src.utils import Formatter as clr
from src.utils import error as err


def run() -> None:
    print('\n' + ' ' + '=' * 60)
    print(clr.apply('bold', 'white', '  Call Me Maybe: Test Suite'))
    print(' ' + '=' * 60)

    options = {
        1: ['Standard prompt batch'],
        2: ['Input handling'],
        3: ['Output validation'],
        4: ['Edge cases']
        }

    print(clr.apply('bold', 'white', f'  {"n.":<3}Description'))
    print(' ' + '-' * 60)

    for k, v in options.items():
        print(f'  {k:<3}{v[0]}')

    print(' ' + '-' * 60)
    choice = input(clr.apply('bold', 'white', f'  Pick an option: '))
    print()

    try:
        choice_num = int(choice.strip())
        print()

        if choice_num == 1:
            print(clr.apply('bold', 'yellow', '>>> Launching Standard prompt batch...\n'))

            command = [
                "uv", "run", "python", "-m", "src",
                "--functions_definition", "data/input/functions_definition.json",
                "--input", "data/input/function_calling_tests.json",
                "--output", "data/output/function_calls.json"
            ]

            subprocess.run(command, check=True)

        elif choice_num in options:
             print(clr.apply('bold', 'yellow', f' Option {choice_num} is not yet implemented.'))
        else:
             err(' Invalid option.\n')

    except ValueError:
        err(' Please enter a valid number.\n')
    except subprocess.CalledProcessError as e:
        err(f' The pipeline crashed or returned an error code: {e}\n')
    except Exception as e:
        err(f' An unexpected error occurred: {e}\n')


if __name__ == "__main__":
    run()
