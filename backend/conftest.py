import sys
from pathlib import Path

# Ensure the backend package root is on sys.path for all test imports.
sys.path.insert(0, str(Path(__file__).parent))
