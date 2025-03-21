import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc

log = logging.getLogger(__file__)

DPI = 300


def plot_intro_diagram(model, config, filename, n_cols=5, z=1.5):
    """Plot intro diagram"""
    colors = plt.cm.viridis(config.feature_importance)

    n_rows = model.n_instance // n_cols

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(2 * n_cols, 2 * n_rows)
    )

    for idx, ax in enumerate(axes.flat):
        x, y = model.w[idx, :, 0], model.w[idx, :, 1]
        ax.scatter(x, y, c=colors)
        segments = np.stack((np.zeros_like(model.w[idx]), model.w[idx]), axis=1)
        ax.add_collection(mc.LineCollection(segments, colors=colors))

        ax.set_aspect("equal")
        ax.set_facecolor("#FCFBF8")
        ax.set_xlim((-z, z))
        ax.set_ylim((-z, z))
        ax.tick_params(
            left=True, right=False, labelleft=False, labelbottom=False, bottom=True
        )
        ax.set_title(f"Sparsity={1 - config.feature_probability[idx]:.2f}")

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for spine in ["bottom", "left"]:
            ax.spines[spine].set_position("center")

    log.info(f"Writing {filename}")
    fig.savefig(filename, dpi=DPI)
