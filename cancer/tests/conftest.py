import sys
from pathlib import Path


def _ensure_project_on_path():
    project_dir = Path(__file__).resolve().parents[1]
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))


_ensure_project_on_path()
