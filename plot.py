import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
from matplotlib import colors as mcolors

log = logging.getLogger(__file__)

DPI = 300


def plot_intro_diagram(model, config, filename, n_cols=5):
    """Plot intro diagram"""
    WA = model.w

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.cm.viridis(np.squeeze(config.feature_importance))
    )

    n_rows = model.n_instance // n_cols
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(2 * n_cols, 2 * n_rows)
    )

    for idx, ax in enumerate(axes.flat):
        W = WA[idx]
        colors = [
            mcolors.to_rgba(c)
            for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ]
        ax.scatter(W[:, 0], W[:, 1], c=colors[0 : len(W[:, 0])])
        ax.set_aspect("equal")
        ax.add_collection(
            mc.LineCollection(np.stack((np.zeros_like(W), W), axis=1), colors=colors)
        )

        z = 1.5
        ax.set_facecolor("#FCFBF8")
        ax.set_xlim((-z, z))
        ax.set_ylim((-z, z))
        ax.tick_params(
            left=True, right=False, labelleft=False, labelbottom=False, bottom=True
        )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for spine in ["bottom", "left"]:
            ax.spines[spine].set_position("center")

    log.info(f"Writing {filename}")
    plt.savefig(filename, dpi=DPI)
