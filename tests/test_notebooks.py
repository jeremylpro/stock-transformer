"""Guards for the Colab/Kaggle training notebook and its docs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "04_colab_kaggle_training.ipynb"
COLAB_REQUIREMENTS = ROOT / "requirements-colab.txt"
DOCS = ROOT / "docs" / "colab_kaggle.md"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


def test_colab_notebook_exists_and_is_valid_json():
    assert NOTEBOOK.exists()
    nb = _notebook()
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"


def test_colab_notebook_code_cells_compile():
    nb = _notebook()
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cells"
    for index, cell in enumerate(code_cells):
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        compile(source, f"notebook-cell-{index}", "exec")


def test_colab_notebook_installs_project_requirements():
    sources = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )
    assert "requirements-colab.txt" in sources
    assert "src.pipeline" in sources
    # The key must never be written into a cell.
    assert "FINNHUB_API_KEY =" in sources
    assert "getpass" in sources


def test_colab_requirements_match_local_pins():
    def pins(path: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, version = line.partition("==")
            out[name.strip().lower()] = version.strip()
        return out

    local = pins(ROOT / "requirements.txt")
    cloud = pins(COLAB_REQUIREMENTS)
    assert cloud, "requirements-colab.txt is empty"
    assert set(cloud) == set(local), "cloud and local dependency sets differ"
    assert cloud == local, "cloud pins differ from local pins"


@pytest.mark.parametrize("name", ["config.yaml", "config.finnhub.yaml"])
def test_notebook_referenced_configs_exist(name):
    assert (ROOT / name).exists()


def test_colab_docs_exist_and_link_the_notebook():
    assert DOCS.exists()
    text = DOCS.read_text()
    assert "04_colab_kaggle_training.ipynb" in text
    assert "requirements-colab.txt" in text
