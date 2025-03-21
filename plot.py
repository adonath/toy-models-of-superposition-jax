import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

log = logging.getLogger(__file__)

DPI = 300
FACECOLOR = "#FCFBF8"


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
        ax.set_facecolor(FACECOLOR)
        ax.set_xlim((-z, z))
        ax.set_ylim((-z, z))
        ax.tick_params(
            left=True, right=False, labelleft=False, labelbottom=False, bottom=True
        )
        ax.set_title(f"1 - S = {config.feature_probability[idx]:.2f}")

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for spine in ["bottom", "left"]:
            ax.spines[spine].set_position("center")

    log.info(f"Writing {filename}")
    fig.savefig(filename, dpi=DPI)


def plot_w_and_b(ax, w, b_final):
    """Plot weights"""
    ax.imshow(w @ w.mT, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.axis("off")
    ax.set_title("$W^TW$", loc="left")

    ax_divider = make_axes_locatable(ax)
    ax_b = ax_divider.append_axes("right", size="7%", pad="2%")
    ax_b.imshow(b_final[:, None], cmap="RdBu_r", vmin=-1, vmax=1)
    ax_b.axis("off")
    ax_b.set_title("b")


def plot_features(ax, w, b_final, config, show_y_label=False):
    """Plot features"""
    w_norm = w / (1e-5 + np.linalg.norm(w, 2, axis=-1, keepdims=True))

    interference = np.einsum("fh,gh->fg", w_norm, w)
    interference[np.arange(config.n_features), np.arange(config.n_features)] = 0

    polysemanticity = np.linalg.norm(interference, axis=-1)
    # net_interference = (interference**2 * config.feature_probability).sum(-1)
    norms = np.linalg.norm(w, 2, axis=-1)

    colors = plt.cm.viridis(polysemanticity)

    y_pos = np.arange(config.n_features, 0, -1)
    ax.barh(y_pos, norms, align="center", color=colors)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title("$\|W_i\|$", loc="left")
    ax.set_facecolor(3 * (0.9,))

    if show_y_label:
        ax.set_ylabel("$\\leftarrow $ Features", loc="top")


def plot_demonstrate_superposition(model, config, filename):
    """Plot demonstrate superposition"""

    idxs = [1, 3, 6, 10, 12, 15, 19]
    n_cols = len(idxs)

    fig = plt.figure(figsize=(12, 6))

    gs = GridSpec(nrows=3, ncols=n_cols, figure=fig)

    for idx in range(n_cols):
        ax_w = fig.add_subplot(gs[0, idx])
        plot_w_and_b(ax_w, model.w[idx], model.b_final[idx])

        ax_f = fig.add_subplot(gs[1:, idx])
        plot_features(
            ax_f, model.w[idx], model.b_final[idx], config=config, show_y_label=idx == 0
        )

    fig.suptitle(
        f"{config.activation.title()} Output Model",
        fontweight="bold",
        ha="left",
        x=0.03,
    )
    log.info(f"Writing {filename}")

    plt.tight_layout()
    fig.savefig(filename, dpi=DPI)
