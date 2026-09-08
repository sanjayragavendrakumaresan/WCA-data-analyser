"""Run the WCA suite from inside the wca_analysis directory.

Usage from this directory:
    python run.py
"""

from pathlib import Path
import sys


# Add the package's parent directory so ``wca_analysis`` can be imported
# even when this file is executed from inside the package directory.
PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from wca_analysis.app import WCAMenu  # noqa: E402


if __name__ == "__main__":
    WCAMenu().run()
