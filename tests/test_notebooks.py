"""Test that Jupyter notebooks in notebooks/ run cleanly top-to-bottom without error."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

NOTEBOOKS_DIR = Path("notebooks")


def run_notebook_cells(nb_path: Path):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # Isolated execution namespace
    global_env = {}
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            code = "".join(cell.get("source", []))
            try:
                exec(code, global_env)
            except Exception as e:
                raise RuntimeError(
                    f"Notebook {nb_path.name} failed at cell {idx + 1}:\n{code}\nError: {e}"
                ) from e


def test_01_eda_notebook_runs():
    nb_path = NOTEBOOKS_DIR / "01_eda.ipynb"
    assert nb_path.exists(), "01_eda.ipynb must exist"
    run_notebook_cells(nb_path)


def test_02_modeling_notebook_runs():
    nb_path = NOTEBOOKS_DIR / "02_modeling_and_validation.ipynb"
    assert nb_path.exists(), "02_modeling_and_validation.ipynb must exist"
    run_notebook_cells(nb_path)
