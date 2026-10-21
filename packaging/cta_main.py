"""PyInstaller entry point (GUI by default, CLI when given a command)."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from cta.cli import main

    sys.exit(main())
