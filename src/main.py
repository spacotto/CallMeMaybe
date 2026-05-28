"""

Usage:
    python3 main.py
    Example: python3 main.py
"""

# ----------------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------------

from utils import error, warning

# ----------------------------------------------------------------------------
#  Main entry point
# ----------------------------------------------------------------------------


def main() -> None:

    args = sys.argv[1:]

    try:
        if '--functions_definition' not in args:
            raise FileNotFoundError('No functions definition provided')

    except FileNotFoundError as e:
        error(e)
        warning("Usage: make run ...")
        return

    except Exception as e:
        error(e)
        return


if __name__ == "__main__":
    main()
