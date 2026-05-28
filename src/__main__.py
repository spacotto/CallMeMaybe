"""

Usage:
    python3 main.py
    Example: python3 main.py
"""

# ----------------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------------

import os
import sys
import json
import argparse
from pydantic import ValidationError
from utils import error, warning

# ----------------------------------------------------------------------------
#  Main entry point
# ----------------------------------------------------------------------------

def main() -> None:

    try:
        ap = ArgumentParser(description="Call Me Maybe: LLM Function Calling")
        ap.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
        ap.add_argument("--input",
                        default="data/input/function_calling_tests.json")
        ap.add_argument("--output",
                        default="data/output/function_calls.json")

        args = ap.parse_args()

        if '--functions_definition' not in args:
            raise FileNotFoundError('No functions definition provided')

    except FileNotFoundError as e:
        error(e)
        warning("Usage: make run ...")
        return

    except Exception as e:
        error(e)
        return

    # Define the arguments

    print(f"Loading definitions from: {args.functions_definition}")
    print(f"Reading prompts from: {args.input}")
    print(f"Writing outputs to: {args.output}")

    # TODO: Pydantic validation, LLM SDK, and Constrained Decoder here!


if __name__ == "__main__":
    main()
