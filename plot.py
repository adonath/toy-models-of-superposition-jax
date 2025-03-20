import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
from matplotlib import colors as mcolors


def plot_intro_diagram(config):
    """Plot intro diagram"""
    from .make import Model

    model = Model.read(config.result_filename)
    WA = model.W.detach()

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.cm.viridis(model.importance[0].cpu().numpy())
    )
    plt.rcParams["figure.dpi"] = 200

    n_rows = config.n_instances
    fig, axes = plt.subplots(1, rows=n_rows, figsize=(2 * n_rows, 2))

    for idx, ax in enumerate(axes):
        W = WA[idx].cpu().detach().numpy()
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
