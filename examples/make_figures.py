"""Regenerate the README figures in docs/images (needs matplotlib and scikit-learn)."""

import pathlib

import matplotlib

matplotlib.use("Agg")

import points  # noqa: E402
import sbm  # noqa: E402
from plots import blobs_figure, digits_figure, link_matrix_figure  # noqa: E402

OUT = pathlib.Path(__file__).parent.parent / "docs" / "images"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blobs_figure(points.run(seed=0), points.ari).savefig(OUT / "blobs.png", dpi=110)

    from sklearn.datasets import load_digits

    digits = load_digits()
    rows = points.classic()
    best = next(r for r in rows if r[0] == "digits")[4]
    digits_figure(digits.images, digits.target, best).savefig(OUT / "digits.png", dpi=110)

    vd = sbm.run()["varying density"]
    fig = link_matrix_figure(
        vd,
        [vd.level_cut, vd.cut],
        [
            f"single-level cut, D={vd.level_cut.value:.3f}",
            f"adaptive cut, D={vd.partition_density:.3f}",
        ],
    )
    fig.savefig(OUT / "sbm.png", dpi=110)


if __name__ == "__main__":
    main()
