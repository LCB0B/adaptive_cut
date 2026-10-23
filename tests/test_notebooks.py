"""The example notebooks run from top to bottom (when their packages are installed)."""

import pathlib

import pytest

NOTEBOOK_DIR = pathlib.Path(__file__).parent.parent / "examples" / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
NEEDS = {
    "05_hdbscan": ["hdbscan"],
    "06_link_communities": ["networkx"],
    "07_balancedness": ["networkx"],
    "08_working_with_results": ["networkx"],
}


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs(path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    for module in ["matplotlib", "sklearn", *NEEDS.get(path.stem, [])]:
        pytest.importorskip(module)
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}
    )
    client.execute()
