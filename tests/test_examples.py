"""The README examples run and show what the README says they show."""

import importlib.util
import pathlib

import numpy as np
import pytest

EXAMPLES = pathlib.Path(__file__).parent.parent / "examples"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("seed", [0, 1])
def test_sbm_example(seed):
    out = _load("sbm").run(seed=seed)
    vd, bal = out["varying density"], out["balanced"]
    assert vd.adaptive.gain > 0.02
    assert bal.adaptive.gain < 0.005
    assert vd.balancedness < 0.4 < bal.balancedness


def test_any_dendrogram_example():
    level, result = _load("any_dendrogram").run(seed=0)
    assert result.value >= level.value


def _readme_blocks():
    text = (EXAMPLES.parent / "README.md").read_text()
    blocks, lang, buf = [], None, []
    for line in text.splitlines():
        if line.startswith("```"):
            if lang is None:
                lang, buf = line[3:].strip(), []
            else:
                blocks.append((lang, "\n".join(buf)))
                lang = None
        elif lang is not None:
            buf.append(line)
    return blocks


def test_readme_sbm_block_prints_the_readme_output(capsys):
    blocks = _readme_blocks()
    i = next(t for t, (lang, code) in enumerate(blocks) if lang == "python" and "def sbm" in code)
    exec(compile(blocks[i][1], "README.md", "exec"), {})
    printed = capsys.readouterr().out.strip()
    assert printed == blocks[i + 1][1].strip()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_points_example(seed):
    out = _load("points").run(seed=seed)
    ari = _load("points").ari
    assert out["adaptive"].value > out["level"].value
    assert ari(out["y"], out["level"].labels) < 0.75
    assert ari(out["y"], out["adaptive"].labels) > 0.9


def test_ari_matches_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 4, 300), rng.integers(0, 6, 300)
    b[:200] = a[:200]
    assert _load("points").ari(a, b) == pytest.approx(metrics.adjusted_rand_score(a, b))


def test_readme_points_block_matches_its_comments(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    code = next(c for lang, c in _readme_blocks() if lang == "python" and "gaussian_scores" in c)
    ns = {}
    exec(compile(code, "README.md", "exec"), ns)
    ari = _load("points").ari
    assert (ns["level"].n_clusters, round(ari(ns["y"], ns["level"].labels), 2)) == (3, 0.69)
    assert (ns["best"].n_clusters, round(ari(ns["y"], ns["best"].labels), 2)) == (5, 0.96)


def test_readme_classic_table():
    pytest.importorskip("sklearn")
    text = (EXAMPLES.parent / "README.md").read_text()
    for name, n, B, level, best, ari_level, ari_best in _load("points").classic():
        row = (
            f"| {name} | {n:,} | {B:.2f} | {level.n_clusters} clusters, ARI {ari_level:.2f} "
            f"| {best.n_clusters} clusters, ARI {ari_best:.2f} |"
        )
        assert row in text


def test_network_example_matches_the_readme():
    pytest.importorskip("networkx")
    _, r, best = _load("network").run()
    assert round(r.level_cut.value, 3) == 0.285
    assert round(best.value, 3) == 0.403
    text = (EXAMPLES.parent / "README.md").read_text()
    assert "partition density 0.285" in text and "partition density 0.403" in text


def _readme_block(marker):
    return next(c for lang, c in _readme_blocks() if lang == "python" and marker in c)


def test_readme_scikit_learn_blocks_run():
    pytest.importorskip("sklearn")
    X, _ = _load("points").blobs(seed=0)
    ns = {"X": X}
    exec(compile(_readme_block("make_pipeline"), "README.md", "exec"), ns)
    assert ns["model"].n_clusters_ >= 2 and len(set(ns["labels"])) >= 2
    exec(compile(_readme_block("linkage_from_sklearn"), "README.md", "exec"), ns)
    assert len(ns["labels"]) == len(X)


def test_readme_karate_block_matches_its_comments():
    pytest.importorskip("networkx")
    ns = {}
    exec(compile(_readme_block("karate_club_graph"), "README.md", "exec"), ns)
    assert round(ns["r"].level_cut.value, 3) == 0.285
    assert round(ns["r"].partition_density, 3) == 0.403


def test_readme_node_communities_block_runs():
    sknetwork = pytest.importorskip("sknetwork")
    nx = pytest.importorskip("networkx")
    A = nx.to_scipy_sparse_array(nx.karate_club_graph(), weight=None, format="csr")
    import scipy.sparse as sp

    A = sp.csr_matrix(A)
    A.indices, A.indptr = A.indices.astype(np.int32), A.indptr.astype(np.int32)
    ns = {"A": A}
    exec(compile(_readme_block("Paris()"), "README.md", "exec"), ns)
    from sknetwork.clustering import get_modularity

    assert sknetwork is not None
    assert ns["res"].value == pytest.approx(get_modularity(A, ns["res"].labels))
