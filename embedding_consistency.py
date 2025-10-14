from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt


def _validate_embeddings(embeddings_by_split: Dict[str, pd.DataFrame]) -> None:
    for split, df in embeddings_by_split.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Embeddings for split '{split}' must be a pandas DataFrame.")
        if df.shape[1] != 2:
            raise ValueError(f"Embeddings for split '{split}' must have exactly 2 columns (got {df.shape[1]}).")
        if df.index.has_duplicates:
            raise ValueError(f"Embeddings index (cell-line ids) has duplicates in split '{split}'.")
        if not np.issubdtype(df.dtypes[0], np.number) or not np.issubdtype(df.dtypes[1], np.number):
            raise TypeError(f"Embedding columns must be numeric for split '{split}'.")


def compute_knn(
    embeddings: pd.DataFrame,
    n_neighbors: int,
    metric: str = "euclidean"
) -> Tuple[Dict[str, List[str]], Dict[str, List[float]]]:
    """
    Compute the KNN neighbor lists (and corresponding distances) for every index item.
    Returns two dicts: neighbors[query] = [nbr1, nbr2, ...], distances[query] = [d1, d2, ...]
    Excludes the query point itself from its neighbors.
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")

    X = embeddings.values.astype(float)
    labels = embeddings.index.to_list()
    D = cdist(X, X, metric=metric)
    # mask out the diagonal (inf distance) to ignore self-self distances
    np.fill_diagonal(D, np.inf)

    neighbors = {}
    distances = {}
    k = min(n_neighbors, max(0, X.shape[0] - 1))
    for i, lab in enumerate(labels):
        # # first get the closest k elements (unsorted)
        # order = np.argpartition(D[i], kth=k-1)[:k] if k > 0 else np.array([], dtype=int)
        # # then actually sort them by increasing distance
        # order = order[np.argsort(D[i, order])]
        # directly sort and get closest k cell-lines
        order = np.argsort(D[i])[:k]
        # get cell-line labels and distance values
        neighbors[lab] = [labels[j] for j in order]
        distances[lab] = [float(D[i, j]) for j in order]
    return neighbors, distances


def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    # if both sets are empty, consider them perfectly similar
    if not A and not B:
        return 1.0
    # if one set is empty and the other is not, return 0
    if not A and B:
        return 0.0
    if A and not B:
        return 0.0
    # regular Jaccard similarity index (intersection over union)
    intersection = len(A & B)
    union = len(A | B)
    return intersection / union, intersection, union


def rank_spearman(a: List[str], b: List[str]) -> Optional[float]:
    """
    Spearman correlation over *ranks*, considering only the intersection of items.
    Returns None if intersection is empty or has size 1.
    """
    if not a or not b:
        return None
    # convert in dictionary mappings
    pos_a = {lab: i for i, lab in enumerate(a)}
    pos_b = {lab: i for i, lab in enumerate(b)}
    common = sorted(set(pos_a) & set(pos_b))
    # minimum 2 common neighbours
    if len(common) < 2:
        return None
    ranks_a = [pos_a[c] for c in common]
    ranks_b = [pos_b[c] for c in common]
    rho, _ = spearmanr(ranks_a, ranks_b)
    return float(rho)


def distance_correlation_from_anchor(
    anchor: str,
    embeddings1: pd.DataFrame,
    embeddings2: pd.DataFrame,
    metric: str = "euclidean",
) -> Optional[float]:
    """
    Correlate the vector of distances from the anchor to all *common other* points
    across the two splits. Returns Pearson r. None if <2 common comparators.
    """
    if anchor not in embeddings1.index or anchor not in embeddings2.index:
        return None
    others1 = embeddings1.index.drop(anchor)
    others2 = embeddings2.index.drop(anchor)
    common_others = list(sorted(set(others1) & set(others2)))
    if len(common_others) < 2:
        return None

    a1 = embeddings1.loc[[anchor]].values.astype(float)
    V1 = embeddings1.loc[common_others].values.astype(float)
    d1 = cdist(a1, V1, metric=metric).ravel()

    a2 = embeddings2.loc[[anchor]].values.astype(float)
    V2 = embeddings2.loc[common_others].values.astype(float)
    d2 = cdist(a2, V2, metric=metric).ravel()

    r, _ = pearsonr(d1, d2)
    return float(r)


def compute_validation_neighborhood_consistency(
    embeddings_by_split: Dict[str, pd.DataFrame],
    validation_cell_by_split: Dict[str, str],
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """
    For each split's validation cell line, compute how consistent its N-nearest neighbors are
    with the neighborhoods obtained in all *other* splits where that cell appears.

    Returns a tidy DataFrame with columns:
        ['cell', 'split', 'other_split', 'k', 'jaccard', 'spearman', 'distance_corr',
         'overlap_count', 'union_count']
    """
    _validate_embeddings(embeddings_by_split)

    # Precompute KNN neighbours and corresponding distances per split
    knn_by_split = {}
    dist_by_split = {}
    for split, df in embeddings_by_split.items():
        knn_by_split[split], dist_by_split[split] = compute_knn(df, n_neighbors=n_neighbors, metric=metric)

    records = []
    for split, val_cell in validation_cell_by_split.items():
        if split not in embeddings_by_split:
            continue
        df_ref = embeddings_by_split[split]
        if val_cell not in df_ref.index:
            # if the validation cell isn't present in this split's embedding, skip
            continue

        # Get embeddings for validation cell-line when it's in val
        neigh_ref = knn_by_split[split].get(val_cell, [])
        for other_split, df_other in embeddings_by_split.items():
            if other_split == split:
                continue
            if val_cell not in df_other.index:
                continue
            # Get embeddings for same validation cell-line when it's in train
            neigh_other = knn_by_split[other_split].get(val_cell, [])

            # Check whether neighbours are consistent (which cell-lines are adjacent to the validation cell-line?)
            J, intersection, union = jaccard(neigh_ref, neigh_other)
            # Check whether distance ordering/ranking between common neighbours is consistent
            rho = rank_spearman(neigh_ref, neigh_other)
            # Check whether distances between val cell-line and all others are consistent across splits (independent of
            # the number of neighbours - checks all)
            rdist = distance_correlation_from_anchor(val_cell, df_ref, df_other, metric=metric)

            records.append({
                "cell": val_cell,
                "split": split,
                "other_split": other_split,
                "k": len(neigh_ref),
                "jaccard": J,
                "spearman": rho,
                "distance_corr": rdist,
                "overlap_count": intersection,
                "union_count": union,
            })

    return pd.DataFrame.from_records(records)


def summarize_consistency(per_pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per (cell, split vs other_split) results into per-cell and global summaries.
    Returns a DataFrame grouped by 'cell' with mean/median stats.
    """
    if per_pair_df.empty:
        return per_pair_df

    agg = (per_pair_df
           .groupby("cell", as_index=False)
           .agg(jaccard_mean=("jaccard", "mean"),
                jaccard_median=("jaccard", "median"),
                spearman_mean=("spearman", "mean"),
                distance_corr_mean=("distance_corr", "mean"),
                overlap_mean=("overlap_count", "mean")))
    return agg


# Minimal plotting utilities
def plot_distribution(per_pair_df: pd.DataFrame):
    if per_pair_df.empty:
        return

    metrics = ["jaccard", "spearman", "distance_corr"]
    df = per_pair_df.melt(
        value_vars=metrics,
        var_name="metric",
        value_name="value"
    )

    sns.histplot(
        data=df,
        x="value",
        hue="metric",  # metrics are now the hue
        bins=20,
        alpha=0.5,  # semi-transparent to see overlaps
        element="step",  # nicer for overlapping histograms
        common_norm=False,  # so each metric's histogram is scaled independently
    )
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Metric Distributions")
    plt.show()


def plot_per_cell_summary(summary_df: pd.DataFrame):
    if summary_df.empty:
        return

        # Select relevant metric columns
    metrics = ["jaccard_mean", "spearman_mean", "distance_corr_mean"]
    df = summary_df.melt(
        id_vars="cell",
        value_vars=metrics,
        var_name="metric",
        value_name="value"
    )

    # Sort cells by one metric (e.g. jaccard_mean) for consistent ordering
    order = sorted(summary_df.sort_values("jaccard_mean")["cell"].tolist())

    g = sns.catplot(
        data=df,
        x="cell", y="value", hue="metric",
        kind="bar",
        order=order, sharey=False,
        height=4, aspect=1.5
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Cell", "Mean Value")

    # Rotate x-axis labels
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)

    # Move legend outside
    g._legend.set_title("Metric")
    g.fig.subplots_adjust(right=0.8)  # make space on the right for legend
    g._legend.set_bbox_to_anchor((1.25, 0.5))  # x, y position (outside right, centered)

    plt.tight_layout()
    plt.show()
