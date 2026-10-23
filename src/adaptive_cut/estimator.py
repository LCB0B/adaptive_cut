"""scikit-learn style estimator and conversion from scikit-learn trees.

``AdaptiveClustering`` behaves like ``sklearn.cluster.AgglomerativeClustering``
(same ``linkage`` and ``metric`` parameters, ``labels_``, ``children_`` and
``distances_`` attributes), but instead of cutting the tree at a fixed number
of clusters or a distance threshold it returns the best multi-level cut under
an objective. scikit-learn is optional: when it is installed the estimator
derives from its base classes, so ``clone``, ``get_params``, pipelines and
grid search work.
"""

from __future__ import annotations

import inspect

import numpy as np
from scipy.cluster.hierarchy import linkage as _scipy_linkage

from ._tree import is_monotone
from .cut import adaptive_cut
from .objectives import gaussian_scores

try:
    from sklearn.base import BaseEstimator as _Base
    from sklearn.base import ClusterMixin as _Mixin

    try:
        from sklearn.utils.validation import validate_data as _validate_data
    except ImportError:  # scikit-learn < 1.6

        def _validate_data(estimator, X, **kwargs):
            return estimator._validate_data(X, **kwargs)

except ImportError:  # scikit-learn is optional
    _validate_data = None

    class _Base:
        """Minimal stand-in for ``sklearn.base.BaseEstimator``."""

        @classmethod
        def _get_param_names(cls):
            sig = inspect.signature(cls.__init__)
            return sorted(p for p in sig.parameters if p != "self")

        def get_params(self, deep=True):
            return {k: getattr(self, k) for k in self._get_param_names()}

        def set_params(self, **params):
            for k, v in params.items():
                if k not in self._get_param_names():
                    raise ValueError(f"invalid parameter {k!r} for {type(self).__name__}")
                setattr(self, k, v)
            return self

        def __repr__(self):
            args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
            return f"{type(self).__name__}({args})"

    class _Mixin:
        def fit_predict(self, X, y=None, **kwargs):
            return self.fit(X, **kwargs).labels_


class AdaptiveClustering(_Mixin, _Base):
    """Hierarchical clustering cut at several levels (the adaptive cut).

    The data are clustered with ``scipy.cluster.hierarchy.linkage``; every
    node of the tree gets a score (by default the BIC of one Gaussian fitted
    to its points, see :func:`gaussian_scores`), and :func:`adaptive_cut`
    searches for the partition into whole subtrees with the highest total
    score. The number of clusters is chosen by the objective, and different
    branches of the tree can be cut at different heights.

    Parameters
    ----------
    linkage : {"ward", "complete", "average", "single", "weighted", "centroid", "median"}
        Linkage method, passed to ``scipy.cluster.hierarchy.linkage``.
    metric : str or callable
        Distance metric (``"euclidean"`` for ward, centroid and median).
    objective : "gaussian" or callable
        ``"gaussian"`` uses :func:`gaussian_scores`. A callable is called as
        ``objective(Z, X)`` and must return one additive score per node.
    covariance : {"diag", "spherical", "full"}
        Covariance model of the ``"gaussian"`` objective.
    var_floor : float
        Variance floor of the ``"gaussian"`` objective.
    T : float or "auto"
        Initial temperature of the sampler, in units of the objective (see
        :func:`adaptive_cut`). Raise it if the chain stays at its starting cut.
    n_steps : int, optional
        Number of sampler steps; default ``max(10_000, 10 n_samples)``.
    init : {"auto", "best_level", "leaves", "root"}
        Starting cut of the sampler. ``"auto"`` is the best single-level cut
        when the merge heights are monotone, else all singletons.
    random_state : None, int or numpy.random.Generator
        Seed of the sampler.
    engine : {"auto", "numba", "python"}
        Sampler engine (see :func:`adaptive_cut`).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster label of every sample.
    n_clusters_ : int
        Number of clusters found.
    linkage_matrix_ : ndarray of shape (n_samples - 1, 4)
        The scipy linkage matrix.
    children_ : ndarray of shape (n_samples - 1, 2)
        Children of every merge, numbered as in scikit-learn and scipy.
    distances_ : ndarray of shape (n_samples - 1,)
        Merge heights.
    scores_ : ndarray of shape (2 * n_samples - 1,)
        Score of every tree node.
    cut_ : Cut
        The best cut found (tree nodes, labels and objective value).
    result_ : AdaptiveCutResult or None
        The full sampler result (trace, acceptance rate, ...). ``None`` when
        the objective is the same for every cut and the sampler is not run.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, y = make_blobs(500, centers=4, random_state=0)
    >>> AdaptiveClustering(linkage="ward", random_state=0).fit(X).n_clusters_
    4
    """

    def __init__(
        self,
        linkage: str = "ward",
        metric="euclidean",
        objective="gaussian",
        covariance: str = "diag",
        var_floor: float = 0.01,
        T="auto",
        n_steps=None,
        init: str = "auto",
        random_state=None,
        engine: str = "auto",
    ):
        self.linkage = linkage
        self.metric = metric
        self.objective = objective
        self.covariance = covariance
        self.var_floor = var_floor
        self.T = T
        self.n_steps = n_steps
        self.init = init
        self.random_state = random_state
        self.engine = engine

    def fit(self, X, y=None):
        """Build the tree of ``X`` and search for the best cut. ``y`` is ignored."""
        X = self._check_X(X)
        Z = _scipy_linkage(X, method=self.linkage, metric=self.metric)
        if callable(self.objective):
            scores = np.asarray(self.objective(Z, X), dtype=np.float64)
        elif self.objective == "gaussian":
            scores = gaussian_scores(Z, X, covariance=self.covariance, var_floor=self.var_floor)
        else:
            raise ValueError(f'objective must be "gaussian" or a callable, got {self.objective!r}')
        init = self.init
        if init == "auto":
            init = "best_level" if is_monotone(Z) else "leaves"
        gains = scores[X.shape[0] :] - scores[Z[:, 0].astype(int)] - scores[Z[:, 1].astype(int)]
        if np.all(np.abs(gains) <= 1e-12 * max(1.0, np.max(np.abs(scores)))):
            # every cut has the same value: keep the root (one cluster)
            result = None
            cut = adaptive_cut(Z, scores, 1.0, n_steps=0, init="root").best
        else:
            result = adaptive_cut(
                Z,
                scores,
                self.T,
                n_steps=self.n_steps,
                init=init,
                rng=self.random_state,
                engine=self.engine,
            )
            cut = result.best
        self.result_ = result
        self.linkage_matrix_ = Z
        self.children_ = Z[:, :2].astype(np.intp)
        self.distances_ = Z[:, 2].copy()
        self.n_leaves_ = X.shape[0]
        self.scores_ = scores
        self.cut_ = cut
        self.labels_ = cut.labels
        self.n_clusters_ = cut.n_clusters
        return self

    def _check_X(self, X) -> np.ndarray:
        if _validate_data is not None:
            return _validate_data(self, X, dtype=np.float64, ensure_min_samples=2)
        if hasattr(X, "tocsr") or hasattr(X, "toarray"):
            raise TypeError("sparse input is not supported; pass a dense array")
        X = np.asarray(X)
        if np.iscomplexobj(X):
            raise ValueError("complex data is not supported")
        X = X.astype(np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-d array of samples, got shape {X.shape}")
        if X.shape[0] < 2:
            raise ValueError(f"need at least 2 samples, got n_samples = {X.shape[0]}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        self.n_features_in_ = X.shape[1]
        return X


def linkage_from_sklearn(model) -> np.ndarray:
    """scipy linkage matrix of a fitted ``sklearn.cluster.AgglomerativeClustering``.

    The model must have merge distances, i.e. be fitted with
    ``compute_distances=True`` or with ``distance_threshold`` set (and
    ``compute_full_tree`` left at its default, so the tree is complete).
    The node numbering of scikit-learn and scipy is the same: merge ``i``
    creates node ``n_samples + i``.
    """
    children = np.asarray(model.children_, dtype=np.float64)
    distances = getattr(model, "distances_", None)
    if distances is None:
        raise ValueError(
            "fit the model with compute_distances=True (or distance_threshold) to get merge heights"
        )
    n = children.shape[0] + 1
    if model.n_leaves_ != n:
        raise ValueError("the tree is not complete; fit with compute_full_tree=True")
    counts = np.ones(2 * n - 1)
    for i, (a, b) in enumerate(children.astype(np.int64)):
        counts[n + i] = counts[a] + counts[b]
    return np.column_stack([children, np.asarray(distances, dtype=np.float64), counts[n:]])
