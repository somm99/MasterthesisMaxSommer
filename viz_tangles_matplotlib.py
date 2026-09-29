"""Matplotlib rendering of the tangle search tree (the "Tree of Tangles" plot)."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from tangles.util.tree import BinTreeNetworkX

from core_tangles import compute_point_scores_for_tangle_node


def fig_tree_of_tangles(two_dimensional_array, tangles, agreement: int = 20):
    """
    Render the tangle tree, drawing a small score-colored scatter at each node.

    Every node of the maximal-tangle tree gets a miniature t-SNE scatter whose
    points are colored by that node's tangle score, so the tree shows how each
    split carves up the embedding.

    Parameters
    ----------
    two_dimensional_array : np.ndarray
        2D embedding (e.g. t-SNE), shape (n_points, 2).
    tangles : object
        Result returned by `tangles.convenience.search_tangles`.
    agreement : int
        Agreement threshold used to select the maximal tangles.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure (caller is responsible for display/closing).
    """
    # colormap: grey for low scores, ramping through blue to red at the top
    c = ["lightgrey", "lightgrey", "blue", "red"]
    v = [0, 0.6, 0.8, 1.0]
    cmap = LinearSegmentedColormap.from_list("mymap", list(zip(v, c)))

    def draw_tangle(G, node_id, ax):
        """Draw one node's mini scatter, colored by its per-point tangle score."""
        bin_tree_node = G.nodes[node_id][BinTreeNetworkX.node_attr_bin_tree_node]

        min_p = (
            two_dimensional_array[:, 0].min() - 1,
            two_dimensional_array[:, 1].min() - 1,
        )
        max_p = (
            two_dimensional_array[:, 0].max() + 1,
            two_dimensional_array[:, 1].max() + 1,
        )

        ax.set_xlim(xmin=min_p[0], xmax=max_p[0])
        ax.set_ylim(ymin=min_p[1], ymax=max_p[1])

        # white background rectangle behind the scatter
        ax.add_patch(
            patches.Rectangle(
                min_p,
                max_p[0] - min_p[0],
                max_p[1] - min_p[1],
                edgecolor="none",
                facecolor="w",
            )
        )

        if bin_tree_node.parent is None:
            ax.scatter(
                two_dimensional_array[:, 0],
                two_dimensional_array[:, 1],
                c=np.ones(two_dimensional_array.shape[0]),
                s=0.1,
                vmin=-1000,
                vmax=1,
                cmap=cmap,
            )
        else:
            scores = compute_point_scores_for_tangle_node(tangles, two_dimensional_array, bin_tree_node)

            ax.scatter(
                two_dimensional_array[:, 0],
                two_dimensional_array[:, 1],
                c=scores,
                s=0.05,
                cmap=cmap,
            )

    bintree = BinTreeNetworkX(
        tangles.tree.maximal_tangles(
            agreement=agreement,
            include_splitting="nodes",
        )
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    bintree.draw(
        draw_node_label_func=draw_tangle,
        draw_edge_label_func=None,
        ax=ax,
        node_label_size=0.03,
        draw_levels=True,
        level_label_func=lambda l: r"$\overrightarrow{{s_{{{0}}}}}$".format(l + 1),
    )

    ax2 = fig.add_axes([0.78, 0.85, 0.1, 0.02])
    ax2.set_ylabel(
        "Tangle score: ",
        rotation="horizontal",
        horizontalalignment="right",
        verticalalignment="center",
    )
    cb = mpl.colorbar.ColorbarBase(
        ax2,
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=0, vmax=1),
        orientation="horizontal",
    )
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])

    return fig