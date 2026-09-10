"""Entry point: launch the Wheat Biomass Toolkit (Augment -> Features -> ML).

Run with the project interpreter:

    .venv\\Scripts\\python.exe main.py
"""

import sys

from gui import run

if __name__ == "__main__":
    sys.exit(run())
