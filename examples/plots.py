"""Figures for the README and the tutorial notebook (needs matplotlib)."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

COLORS = [
    "#4269d0",
    "#efb118",
    "#ff725c",
    "#6cc5b0",
    "#3ca951",
    "#ff8ab7",
    "#a463f2",
    "#97bbf5",
    "#9c6b4e",
    "#9498a0",
]
GREY = "#b0b0b0"


def _color(k):
    return COLORS[k % len(COLORS)]


def _node_cluster(Z, labels):
    """Cluster of every node whose leaves all share one cluster, else -1."""
    n = len(labels)
    lab = np.r_[labels, np.full(n - 1, -1)]
    for i, (a, b) in enumerate(Z[:, :2].astype(int)):
        if lab[a] == lab[b]:
            lab[n + i] = lab[a]
    return lab


def plot_dendrogram_cut(ax, Z, level, adaptive):
    """Dendrogram coloured by the adaptive cut, with the single-level cut as a line."""
    lab = _node_cluster(Z, adaptive.labels)
    dendrogram(
        Z, ax=ax, no_labels=True, link_color_func=lambda v: _color(lab[v]) if lab[v] >= 0 else GREY
    )
    heights = np.unique(Z[:, 2])
    above = heights[heights > level.height]
    ax.axhline(
        (level.height + above[0]) / 2 if len(above) else level.height, color="k", ls="--", lw=1
    )
    ax.set_yscale("symlog", linthresh=np.quantile(Z[:, 2], 0.5))
    ax.set_ylabel("merge height")
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)


def plot_points(ax, X, labels, title):
    ax.scatter(X[:, 0], X[:, 1], s=4, c=[_color(k) for k in labels], linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def blobs_figure(out, ari):
    """Dendrogram on top; truth, single-level cut and adaptive cut below."""
    X, y, Z, level, adaptive = (out[k] for k in ("X", "y", "Z", "level", "adaptive"))
    fig = plt.figure(figsize=(9, 6.2), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.15])
    ax = top.subplots()
    plot_dendrogram_cut(ax, Z, level, adaptive)
    ax.set_title(
        "Ward dendrogram: colours are the adaptive cut, the dashed line is the "
        "best single-level cut",
        fontsize=10,
    )
    axes = bottom.subplots(1, 3)
    plot_points(axes[0], X, y, "true blobs")
    plot_points(
        axes[1],
        X,
        level.labels,
        f"single-level cut: {level.n_clusters} clusters, ARI {ari(y, level.labels):.2f}",
    )
    plot_points(
        axes[2],
        X,
        adaptive.labels,
        f"adaptive cut: {adaptive.n_clusters} clusters, ARI {ari(y, adaptive.labels):.2f}",
    )
    return fig


def digits_figure(images, y, cut, ncols=11):
    """Mean image of every cluster, with its size and most common digit."""
    k = cut.n_clusters
    order = np.argsort(-cut.sizes)
    nrows = -(-k // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 0.85, nrows * 1.05), layout="constrained"
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, c in zip(axes.ravel(), order):
        members = cut.labels == c
        ax.imshow(images[members].mean(0), cmap="gray_r")
        digit = np.bincount(y[members]).argmax()
        ax.set_title(f"'{digit}' n={members.sum()}", fontsize=7)
    return fig


def link_matrix_figure(r, cuts, titles):
    """Adjacency matrix of a network, each edge coloured by its link community."""
    fig, axes = plt.subplots(1, len(cuts), figsize=(4.2 * len(cuts), 4.4), layout="constrained")
    u, v = r.edges[:, 0], r.edges[:, 1]
    for ax, cut, title in zip(np.atleast_1d(axes), cuts, titles):
        # big communities get distinct colours, communities of <= 2 edges are grey
        rank = np.argsort(np.argsort(-cut.sizes))
        col = np.array([_color(k) if s > 2 else GREY for k, s in zip(rank, cut.sizes)])
        c = col[cut.labels]
        ax.scatter(np.r_[u, v], np.r_[v, u], s=1.5, c=np.r_[c, c], linewidths=0, marker="s")
        ax.set_xlim(-0.5, len(r.node_labels) - 0.5)
        ax.set_ylim(len(r.node_labels) - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    return fig
