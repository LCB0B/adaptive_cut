"""Balancedness of a dendrogram."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._tree import as_linkage, level_ends


class Levels(NamedTuple):
    """Per-level quantities behind :func:`balancedness` (one entry per distinct height)."""

    height: np.ndarray
    n_clusters: np.ndarray
    entropy: np.ndarray
    entropy_min: np.ndarray
    entropy_max: np.ndarray
    ratio: np.ndarray  # (H - H_min) / (H_max - H_min); nan where H_max == H_min


def _xlog2x(x):
    x = np.asarray(x, dtype=np.float64)
    return x * np.log2(x)


def balancedness(Z, *, return_levels: bool = False):
    r"""Balancedness ``B`` of a dendrogram, in ``[0, 1]``.

    Each distinct merge height of ``Z`` defines a level: the partition of the
    ``n`` leaves obtained by applying every merge at or below that height (the
    same partition as ``fcluster(Z, height, "distance")``). For a level with
    ``k`` clusters of sizes ``s_i``, with ``H = -\sum (s_i/n) \log_2(s_i/n)``,

    * ``H_max`` is the entropy of the most even split of ``n`` into ``k``
      clusters (``n mod k`` clusters of size ``ceil(n/k)``, the rest
      ``floor(n/k)``),
    * ``H_min`` is the entropy of the most skewed one (``k - 1`` singletons
      and one cluster of size ``n - k + 1``),

    and ``B`` is the mean of ``(H - H_min) / (H_max - H_min)`` over all levels
    where ``H_max > H_min`` (that is ``2 <= k <= n - 2``).

    ``B = 1`` when every level is as even as possible and ``B = 0`` for a
    caterpillar (each merge adds one leaf). ``nan`` is returned when no level
    is informative, e.g. ``n <= 3`` or all merges at a single height.

    Parameters
    ----------
    Z : array_like, shape (n-1, 4)
        scipy linkage matrix with non-decreasing merge heights.
    return_levels : bool
        Also return a :class:`Levels` tuple with the per-level values.

    Returns
    -------
    float, or (float, Levels)
    """
    Z, n = as_linkage(Z, monotone=True)
    left = Z[:, 0].astype(np.int64)
    right = Z[:, 1].astype(np.int64)
    size = np.ones(2 * n - 1)
    size[n:] = Z[:, 3]
    # running sum of s*log2(s) over the current clusters
    S = np.cumsum(_xlog2x(Z[:, 3]) - _xlog2x(size[left]) - _xlog2x(size[right]))
    ends = level_ends(Z)
    S = S[ends]
    k = n - 1 - ends
    a, q = n // k, n % k
    s_max = q * _xlog2x(a + 1) + (k - q) * _xlog2x(a)  # most even split
    s_min = _xlog2x(n - k + 1)  # most skewed split
    informative = (k >= 2) & (k <= n - 2)
    ratio = np.full(len(ends), np.nan)
    np.divide(s_min - S, s_min - s_max, out=ratio, where=informative)
    # (H - H_min)/(H_max - H_min) lies in [0, 1]; the clip only absorbs rounding
    ratio[informative] = np.clip(ratio[informative], 0.0, 1.0)
    B = float(ratio[informative].mean()) if informative.any() else float("nan")
    if not return_levels:
        return B
    log2n = np.log2(n)
    levels = Levels(
        height=Z[ends, 2],
        n_clusters=k,
        entropy=log2n - S / n,
        entropy_min=log2n - s_min / n,
        entropy_max=log2n - s_max / n,
        ratio=ratio,
    )
    return B, levels
