"""
Core tangle pipeline: data loading, separation building, tangle search,
scoring, and the derived tables consumed by the Streamlit frontend.

The public entry point is `build_tangle_model`, which runs the full workflow
end to end and returns a dictionary holding every artefact the UI needs.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler

from tangles.util.matrix_order import linear_similarity_from_dist_matrix
from tangles.convenience import search_tangles, DefaultProgressCallback, TangleSweepFeatureSys
from tangles.convenience import create_order_function
from tangles.analysis import tangle_score as _tangle_score

from dataclasses import dataclass

from tangles.util.tree import BinTreeNetworkX


@dataclass
class SeparationInfo:
    """Metadata for a single separation (one column of the matrix S)."""
    index: int          # column index in S
    feature: str        # source column name
    kind: str           # "numeric" or "categorical"
    threshold: float | None  # numeric separations only
    value: object | None     # categorical separations only
    direction: str      # base-side operator: ">" / "<=" / "=="

def build_numerical_separations(df_numeric: pd.DataFrame, k_per_feature: int = 5, ) -> tuple[np.ndarray, list[SeparationInfo]]:
    """
    Build separations for numeric features by thresholding each column.

    For every feature, `k_per_feature` equidistant thresholds are placed
    strictly between its min and max. Each threshold produces one separation
    with +1 where `value > threshold` and -1 otherwise.

    Returns
    -------
    tuple[np.ndarray, list[SeparationInfo]]
        The (n_points, n_separations) matrix S and one SeparationInfo per column.
    """

    n = len(df_numeric)
    S_cols: list[np.ndarray] = []
    sep_meta: list[SeparationInfo] = []
    col_idx = 0

    for feature in df_numeric.columns:
        values = df_numeric[feature].to_numpy()
        vmin, vmax = float(values.min()), float(values.max())

        # k equidistant thresholds (endpoints excluded)
        thresholds = np.linspace(vmin, vmax, k_per_feature + 2)[1:-1]

        for t in thresholds:
            # base side: feature > t
            sep = -np.ones(n, dtype=np.int8)
            sep[values > t] = 1
            S_cols.append(sep)

            sep_meta.append(
                SeparationInfo(
                    index=col_idx,
                    feature=feature,
                    kind="numeric",
                    threshold=float(t),
                    value=None,
                    direction=">",  # base side
                )
            )
            col_idx += 1

    S = np.stack(S_cols, axis=1)  # shape: (n, n_separations)
    return S, sep_meta

def build_categorical_separations(df_all: pd.DataFrame, index: pd.Index, max_categories_per_feature: int = 10, min_freq: int = 5, exclude_columns: list[str] | None = None):
    """
    Build one-vs-rest separations for categorical / boolean features.

    Only rows in `index` are used (to stay aligned with the numeric subset).
    For each feature, categories occurring at least `min_freq` times are kept,
    capped at `max_categories_per_feature`. Each kept category becomes a
    separation with +1 where `feature == value` and -1 otherwise.

    Returns
    -------
    tuple[np.ndarray | None, list[SeparationInfo]]
        The (n_points, n_separations) matrix and its metadata, or (None, [])
        when no usable categorical separation exists.
    """
    # restrict to the rows that survived numeric NaN-dropping / sampling
    df_cat = df_all.loc[index].select_dtypes(include=["object", "category", "bool"])

    if exclude_columns:
        to_drop = [c for c in exclude_columns if c in df_cat.columns]
        df_cat = df_cat.drop(columns=to_drop)

    if df_cat.empty:
        return None, []

    n = len(df_cat)
    S_cols = []
    sep_meta = []
    col_idx = 0

    for feature in df_cat.columns:
        series = df_cat[feature]

        vc = series.value_counts()
        # keep only frequent-enough categories, capped at the maximum
        vc = vc[vc >= min_freq].iloc[:max_categories_per_feature]
        if vc.empty:
            continue

        for val in vc.index:
            mask = (series == val).to_numpy()
            sep = np.where(mask, 1, -1).astype(np.int8)

            S_cols.append(sep)
            sep_meta.append(
                SeparationInfo(
                    index=col_idx,
                    feature=feature,
                    kind="categorical",
                    threshold=None,
                    value=val,
                    direction="==",  # base side
                )
            )
            col_idx += 1

    if not S_cols:
        return None, []

    S = np.stack(S_cols, axis=1)
    return S, sep_meta

# ------------------------
# Data preparation
# ------------------------

def prepare_data(csv_path: str, exclude_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a CSV and derive the numeric subset used for the embedding.

    Numeric columns are selected, excluded columns dropped, and rows with NaNs
    removed. Datasets larger than 1000 rows are randomly subsampled to 1000
    (fixed seed) to keep the t-SNE and tangle search tractable.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (numeric subset used downstream, full original dataframe).
    """
    df = pd.read_csv(csv_path)
    df_numeric = df.select_dtypes(include=[np.number])

    if exclude_columns:
        to_drop = [c for c in exclude_columns if c in df_numeric.columns]
        df_numeric = df_numeric.drop(columns=to_drop)

    df_numeric_not_null = df_numeric.dropna()
    if len(df.index) > 1000:
        df_numeric_not_null = df_numeric_not_null.sample(n=1000, random_state=32)
    return df_numeric_not_null, df

def apply_tsne(array: np.ndarray, n_components: int = 2, perplexity: float = 5.0) -> np.ndarray:
    """
    Embed `array` into `n_components` dimensions with t-SNE (PCA init, fixed seed).
    """
    tsne = TSNE(
        n_components=n_components,
        learning_rate="auto",
        init="pca",
        perplexity=perplexity,
        random_state=1,
    )
    return tsne.fit_transform(array)

def build_full_separation_system(df_numeric: pd.DataFrame, df_all_columns: pd.DataFrame, idx: pd.Index, max_same_parameter: int = 1, exclude_columns: list[str] | None = None):
    """
    Assemble the complete separation matrix from numeric and categorical features.

    Numeric and categorical separations are built independently and concatenated
    column-wise. The metadata indices of the categorical block are shifted so
    that every `SeparationInfo.index` matches its column position in the combined S.

    Returns
    -------
    tuple[np.ndarray, list[SeparationInfo]]
        The combined matrix S and its aligned metadata.
    """
    S_num, meta_num = build_numerical_separations(df_numeric, k_per_feature=max_same_parameter)

    S_cat, meta_cat = build_categorical_separations(
        df_all_columns,
        index=idx,
        max_categories_per_feature=10,
        min_freq=5,
        exclude_columns=exclude_columns,
    )

    if S_cat is not None:
        S = np.concatenate([S_num, S_cat], axis=1)
        # shift categorical indices to their position in the concatenated matrix
        offset = len(meta_num)
        for m in meta_cat:
            m.index += offset
        sep_meta = meta_num + meta_cat
    else:
        S = S_num
        sep_meta = meta_num

    return S, sep_meta


# ------------------------
# Distance / Similarity / Separations
# ------------------------

def compute_distance_matrix(embedding: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distance matrix of the embedded points.
    """
    return distance_matrix(embedding, embedding)


def compute_similarity_matrix(distances: np.ndarray, margin: float = 1.5, sparse_mat: bool = False) -> np.ndarray:
    """
    Convert a distance matrix into a linear similarity matrix.

    Used as input to the similarity-based order functions ("04", "cut", "radiocut").
    """
    return linear_similarity_from_dist_matrix(
        distances,
        margin=margin,
        sparse_mat=sparse_mat,
    )

# ------------------------
# Tangle-Search and Scores
# ------------------------

def run_tangle_search(S: np.ndarray, sep_meta: np.ndarray, min_agreement: int = 20, max_number_of_seps: int = 10, order_function: str = "01", sim_matrix: np.ndarray | None = None) -> TangleSweepFeatureSys:
    """
    Run the tangle search over the separation matrix S.

    If `order_function` is None, separations are considered in their given order
    and the search is bounded by `max_number_of_seps`. Otherwise a named order
    function drives the search: the similarity-based orders ("04", "cut",
    "radiocut") are built from `sim_matrix`, all others from S itself.

    Returns
    -------
    TangleSweepFeatureSys
        The search result, exposing the tangle tree and scoring helpers.
    """
    if order_function is None:
        tangles = search_tangles(
            S,
            min_agreement=min_agreement,
            sep_metadata=sep_meta,
            max_number_of_seps=max_number_of_seps,
            progress_callback=DefaultProgressCallback(
                show_info_while_running=True
            ),
        )
        return tangles

    if order_function in ("04", "cut", "radiocut"):
        order_function = create_order_function(order_function, sim_matrix)
    else:
        order_function = create_order_function(order_function, S)

    tangles = search_tangles(
        S,
        min_agreement=min_agreement,
        sep_metadata=sep_meta,
        order=order_function,
        progress_callback=DefaultProgressCallback(
            show_info_while_running=True
        ),
    )
    return tangles


def compute_point_scores_for_tangle_node(tangles, embedding: np.ndarray, bin_tree_node) -> np.ndarray:
    """
    Compute the per-point tangle score for a single node of the tangle tree.

    The node's root-to-node path gives the orientation of each separation; the
    library `tangle_score` then returns one score per data point (higher = the
    point agrees more with this tangle).
    """
    path = np.array(bin_tree_node.path_from_root_indicator())[np.newaxis, :]
    scores = _tangle_score(
        path,
        tangles.tree.sep_ids[: path.shape[1]],
        tangles.sep_sys,
    ).reshape(-1)
    return scores


def compute_tangle_scores(tangles, min_agreement: int, ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-point scores for every maximal tangle.

    Returns both the raw (unnormalized) scores and a per-tangle normalized copy
    scaled to [0, 1] by dividing each column by its maximum (columns whose max
    is 0 are left unchanged to avoid division by zero).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (scores_norm, raw_scores), each of shape (n_points, n_tangles).
    """
    # raw scores (counts), not normalized
    raw_scores = tangles.tangle_score(
        min_agreement=min_agreement,
        normalize=None,
    )  # shape: (n_points, n_tangles)

    # normalize each tangle (column) to [0, 1]
    max_per_tangle = raw_scores.max(axis=0, keepdims=True)  # shape: (1, n_tangles)
    max_per_tangle[max_per_tangle == 0] = 1.0

    scores_norm = raw_scores / max_per_tangle
    return scores_norm, raw_scores

def compute_tangle_feature_effects(
    df_numeric: pd.DataFrame,
    scores_all: np.ndarray,
) -> pd.DataFrame:
    """
    Score-weighted, standardized feature effects per tangle.

    For each tangle and numeric feature, the point scores are used as weights to
    compute a weighted mean, then expressed as a z-score against the global
    distribution:

        z = (weighted_mean_in_tangle - global_mean) / global_std

    Positive values mean the tangle skews high on that feature, negative low.

    Returns
    -------
    pd.DataFrame
        (n_tangles x n_features) z-scores, indexed "T0", "T1", ...
    """
    X = df_numeric.to_numpy().astype(float)
    features = list(df_numeric.columns)

    n_points, n_tangles = scores_all.shape
    if n_points == 0 or n_tangles == 0:
        return pd.DataFrame(columns=features)

    global_mean = X.mean(axis=0)
    global_std = X.std(axis=0, ddof=0)
    global_std[global_std == 0] = 1.0

    effects = np.zeros((n_tangles, len(features)), dtype=float)

    for t in range(n_tangles):
        w = scores_all[:, t].astype(float)
        w_sum = w.sum()

        if w_sum <= 1e-12:
            continue

        mean_t = (X * w[:, None]).sum(axis=0) / w_sum
        effects[t, :] = (mean_t - global_mean) / global_std

    index = [f"T{t}" for t in range(n_tangles)]
    return pd.DataFrame(effects, index=index, columns=features)

# ------------------------
# Build custom data for tooltips in visualizations
# ------------------------
def build_customdata(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """
    Build a Plotly `customdata` array (n_points, len(columns)) for hover tooltips.

    Row order is preserved and must match the embedding/scores. Returns None if
    no columns are requested.
    """
    if not columns:
        return None
    return np.stack([df[col].to_numpy() for col in columns], axis=-1)

# ------------------------
# Features of tangles
# ------------------------
def describe_tangle_conditions(
    tangles,
    sep_meta,
    agreement: int = 20,
):
    """
    Turn each maximal tangle into a human-readable list of conditions.

    Walks every leaf of the maximal-tangle tree and, for each separation on its
    root-to-leaf path, renders the oriented condition as text (e.g. "age > 30",
    "city != 'Berlin'"). Numeric orientations flip ">" to "<="; categorical
    orientations flip "==" to "!=".

    Returns
    -------
    list[dict]
        One entry per tangle with keys: node_id, name ("Tangle i"), conditions
        (list of strings) and rules (list of {sep_id, orientation, text}).
    """
    bintree = BinTreeNetworkX(
        tangles.tree.maximal_tangles(
            agreement=agreement,
            include_splitting="nodes",
        )
    )

    descriptions = []

    def collect_tangle(G, node_id, ax):
        # only leaves are maximal tangles (so no outgoing edges)
        if G.out_degree(node_id) != 0:
            return

        bin_tree_node = G.nodes[node_id][BinTreeNetworkX.node_attr_bin_tree_node]

        # orientation of each separation along the path: +1 / -1
        path = np.array(bin_tree_node.path_from_root_indicator())
        feature_ids = tangles.tree.sep_ids[: path.shape[0]]

        cond_strings = []
        cond_rules = []

        for ori, feature_id in zip(path, feature_ids):
            head_meta = tangles.sep_sys.separation_metadata(int(feature_id))
            sep_id = int(head_meta.info.index)  # column index in S / index into sep_meta
            meta = sep_meta[sep_id]
            effective_ori = int(ori) * int(head_meta.orientation)

            if meta.kind == "numeric":
                base_dir = meta.direction  # ">" on the base side
                direction = base_dir if effective_ori >= 0 else "<="
                cond_text = f"{meta.feature} {direction} {meta.threshold:.3f}"
            else:  # categorical
                if effective_ori >= 0:
                    cond_text = f"{meta.feature} == {meta.value!r}"
                else:
                    cond_text = f"{meta.feature} != {meta.value!r}"

            cond_strings.append(cond_text)
            cond_rules.append(
                {
                    "sep_id": sep_id,
                    "orientation": int(1 if effective_ori >= 0 else -1),
                    "text": cond_text,
                }
            )

        descriptions.append(
            {
                "node_id": node_id,
                "name": f"Tangle {len(descriptions)}",
                "conditions": cond_strings,
                "rules": cond_rules,
            }
        )

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    bintree.draw(
        draw_node_label_func=collect_tangle,
        draw_edge_label_func=None,
        ax=ax,
        node_label_size=0.05,
        draw_levels=True,
        level_label_func=lambda l: "",
    )

    plt.close(fig)
    return descriptions

def scan_agreement_range_from_tangle_dict(
    tangle_dict: dict,
    agreements: list[int],
    max_number_of_seps: int = 10,
    order_function: str = "01",
) -> pd.DataFrame:
    """
    Reuses the already computed separation system from tangle_dict and
    runs the tangle search for multiple agreement values.
    """
    S = tangle_dict.get("S")
    sep_meta = tangle_dict.get("sep_meta")
    sim_matrix = tangle_dict.get("similarity")

    if S is None or sep_meta is None:
        return pd.DataFrame(columns=["agreement", "n_tangles"])

    rows = []
    for agreement in agreements:
        tangles = run_tangle_search(
            S,
            sep_meta=sep_meta,
            min_agreement=int(agreement),
            max_number_of_seps=max_number_of_seps,
            order_function=order_function,
            sim_matrix=sim_matrix,
        )

        maximal_tangles = tangles.tree.maximal_tangles(agreement=int(agreement))
        rows.append(
            {
                "agreement": int(agreement),
                "n_tangles": int(len(maximal_tangles)),
            }
        )

    return pd.DataFrame(rows)


# ------------------------
# Membership of objects to tangles
# ------------------------
def membership_from_scores(scores_all: np.ndarray, tangle_index: int, threshold: float = 0.0,) -> np.ndarray:
    """
    Boolean membership mask for one tangle: True where score >= threshold.
    """
    col = scores_all[:, tangle_index]
    return col >= threshold

# ------------------------
# End-to-End Pipeline
# ------------------------

def build_tangle_model(csv_path: str, min_agreement: int = 20, tsne_perplexity: float = 5.0, max_same_parameter: int = 1, max_number_of_seps: int = 10, order_function: str = "01", normalized_values_for_tsne: bool = True, exclude_columns: list[str] | None = None) -> dict:
    """
    Run the full pipeline and return every artefact the UI needs.

    Steps: load data -> (optionally standardize) t-SNE embedding -> distance and
    similarity matrices -> separation system -> tangle search -> per-tangle
    condition descriptions, point scores, and feature-effect table.

    Returns
    -------
    dict
        Keys: df, df_all_columns, array, embedding, distances, similarity, S,
        sep_meta, tangle_descriptions, tangles, scores_all (normalized),
        scores_raw, feature_effects.
    """
    df, df_all_columns = prepare_data(csv_path, exclude_columns)
    array = df.to_numpy()
    scaler = StandardScaler()

    if normalized_values_for_tsne:
        array_tsne = scaler.fit_transform(array)

        embedding = apply_tsne(array_tsne, perplexity=tsne_perplexity)
    else:
        embedding = apply_tsne(array, perplexity=tsne_perplexity)

    array_for_similarity = scaler.fit_transform(array)
    dist = compute_distance_matrix(array_for_similarity)
    sim = compute_similarity_matrix(dist)

    S, sep_meta = build_full_separation_system(df, df_all_columns, df.index, max_same_parameter=max_same_parameter, exclude_columns=exclude_columns)

    tangles = run_tangle_search(S, min_agreement=min_agreement, max_number_of_seps=max_number_of_seps, order_function=order_function, sep_meta=sep_meta, sim_matrix=sim)
    tangle_descriptions = describe_tangle_conditions(tangles, sep_meta, agreement=min_agreement)

    scores, scores_raw = compute_tangle_scores(
        tangles,
        min_agreement=min_agreement,
    )

    feature_effects = compute_tangle_feature_effects(
        df_numeric=df,      # numeric subset
        scores_all=scores,  # normalized scores in [0, 1]
    )

    return {
        "df": df,
        "df_all_columns": df_all_columns,
        "array": array,
        "embedding": embedding,
        "distances": dist,
        "similarity": sim,
        "S": S,
        "sep_meta": sep_meta,
        "tangle_descriptions": tangle_descriptions,
        "tangles": tangles,
        "scores_all": scores,
        "scores_raw": scores_raw,
        "feature_effects": feature_effects,
    }