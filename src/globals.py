import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_repo_root():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT
