# Lets pytest import the top-level packages (core, scrapers, ...) whether it
# is run from this directory or from the repository root.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
