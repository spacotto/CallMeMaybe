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
        print()

    except Exception:
        err(' Invalid option.\n')


if __name__ == "__main__":
    run()
