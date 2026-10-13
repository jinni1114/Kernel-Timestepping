"""Put the repository root on the import path, so the tests can import problems/ (not installed with the package)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
