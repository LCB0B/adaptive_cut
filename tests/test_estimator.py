import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, linkage

import adaptive_cut as ac


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [8, 0], [0, 8], [8, 8]])
    y = np.repeat(np.arange(4), [120, 80, 50, 30])
    return centers[y] + rng.standard_normal((len(y), 2)), y


@pytest.mark.parametrize(
    "method", ["ward", "average", "complete", "single", "weighted", "centroid", "median"]
)
def test_fit_matches_adaptive_cut_of_scipy_linkage(method):
    X, _ = _blobs()
    model = ac.AdaptiveClustering(linkage=method, random_state=0).fit(X)
    Z = linkage(X, method)
    init = "leaves" if method in ("centroid", "median") else "best_level"
    cut = ac.adaptive_cut(Z, ac.gaussian_scores(Z, X), "auto", init=init, rng=0).best
    np.testing.assert_array_equal(model.linkage_matrix_, Z)
    np.testing.assert_array_equal(model.labels_, cut.labels)
    assert model.n_clusters_ == cut.n_clusters
    np.testing.assert_array_equal(model.children_, Z[:, :2].astype(int))
    np.testing.assert_array_equal(model.distances_, Z[:, 2])


def test_fit_predict_finds_the_blobs():
    X, y = _blobs()
    labels = ac.AdaptiveClustering(random_state=0).fit_predict(X)
    assert len(set(zip(labels, y))) == len(set(y)) == len(set(labels))


def test_callable_objective_and_parameters():
    X, _ = _blobs()
    seen = {}

    def objective(Z, data):
        seen["shape"] = (Z.shape, data.shape)
        return ac.gaussian_scores(Z, data, covariance="spherical")

    model = ac.AdaptiveClustering(linkage="average", objective=objective).fit(X)
    assert seen["shape"] == ((len(X) - 1, 4), X.shape)
    assert model.get_params()["linkage"] == "average"
    model.set_params(linkage="ward")
    assert model.linkage == "ward"
    with pytest.raises(ValueError, match="objective"):
        ac.AdaptiveClustering(objective="bic").fit(X)


def test_scikit_learn_integration():
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _blobs()
    model = clone(ac.AdaptiveClustering(covariance="full"))
    assert model.covariance == "full"
    labels = make_pipeline(StandardScaler(), ac.AdaptiveClustering()).fit_predict(X)
    assert len(set(labels)) == 4


def test_passes_scikit_learn_estimator_checks():
    pytest.importorskip("sklearn")
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(ac.AdaptiveClustering())


@pytest.mark.parametrize("method", ["ward", "complete", "average", "single"])
def test_linkage_from_sklearn_matches_scipy(method):
    pytest.importorskip("sklearn")
    from sklearn.cluster import AgglomerativeClustering

    X = np.random.default_rng(0).standard_normal((200, 3))
    model = AgglomerativeClustering(linkage=method, n_clusters=None, distance_threshold=0).fit(X)
    Z = ac.linkage_from_sklearn(model)
    Zs = linkage(X, method)
    np.testing.assert_allclose(np.sort(Z[:, 2]), np.sort(Zs[:, 2]))
    for k in (2, 5, 20):
        a, b = fcluster(Z, k, "maxclust"), fcluster(Zs, k, "maxclust")
        assert len(set(zip(a, b))) == len(set(a)) == len(set(b))
    level = ac.best_level_cut(Z, ac.gaussian_scores(Z, X))
    np.testing.assert_array_equal(
        level.labels, ac.best_level_cut(Zs, ac.gaussian_scores(Zs, X)).labels
    )


def test_linkage_from_sklearn_needs_distances():
    pytest.importorskip("sklearn")
    from sklearn.cluster import AgglomerativeClustering

    X = np.random.default_rng(0).standard_normal((20, 2))
    with pytest.raises(ValueError, match="compute_distances"):
        ac.linkage_from_sklearn(AgglomerativeClustering(n_clusters=3).fit(X))


def test_works_without_scikit_learn():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        sys.modules["sklearn"] = None  # make "import sklearn" fail
        import numpy as np, scipy.sparse as sp
        import adaptive_cut as ac
        noise = np.random.default_rng(0).standard_normal((40, 2))
        X = np.r_[np.zeros((20, 2)), np.full((20, 2), 9.0)] + noise
        m = ac.AdaptiveClustering(linkage="average")
        assert m.fit_predict(X).max() == 1 and m.n_clusters_ == 2
        assert m.get_params()["linkage"] == "average" and "AdaptiveClustering(" in repr(m)
        assert m.set_params(covariance="full").covariance == "full"
        for bad, err in [(sp.csr_matrix(X), TypeError), (X[:1], ValueError), (X + 1j, ValueError)]:
            try:
                m.fit(bad)
            except err:
                pass
            else:
                raise AssertionError(bad)
        try:
            m.set_params(n_clusters=3)
        except ValueError:
            pass
        else:
            raise AssertionError("unknown parameter accepted")
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
