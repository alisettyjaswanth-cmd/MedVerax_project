import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

for p in [parent_dir, current_dir, os.getcwd()]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.main import app
except ImportError:
    from main import app
