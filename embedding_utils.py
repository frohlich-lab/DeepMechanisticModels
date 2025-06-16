import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC


def load_embedding_data_for_context(
        config_df: pd.DataFrame,
        evaluation_template
) -> pd.DataFrame:
    """
    Loads embedding data from files for a single context.

    Parameters
    ----------
    config_df : pd.DataFrame
        DataFrame of configuration dicts for a single context.
    evaluation_template : str
        Path template with placeholders for config values plus {model}, {data}, and {dataset}.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of embeddings from all matching files.
        Returns an empty DataFrame if none are found.
    """
    df_list = []

    for _, config_row in config_df.iterrows():
        config_dict = config_row.to_dict()
        for dataset in ["train", "test"]:
            filepath = evaluation_template.format(
                **config_dict,
                dataset=dataset
            )
            if Path(filepath).exists():
                df_list.append(pd.read_csv(filepath))
            else:
                print(f"[SKIP] File not found: {filepath}")

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        print(f"No valid files found for context: {config_df['context'].iloc[0]}")
        return pd.DataFrame()


def perform_pca_on_embeddings(
    embeddings_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    """
    Performs PCA on latent embeddings grouped by context and sample.

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        A single DataFrame containing all contexts, with at least:
        - 'context', 'samples', 'cell_line', 'L1', 'L2', 'job'

    Returns
    -------
    whole_pca_le : pd.DataFrame
        Concatenated PCA-reduced dataframes with added metadata.
    explained_variance_ratios : dict[tuple[str, str], float]
        Mapping from (context, samples) to total variance explained by first 2 PCs.
    """
    results_dfs = []
    explained_variance_ratios = {}

    for (context, samples), group_df in embeddings_df.groupby(["context", "samples"]):
        les = group_df[["cell_line", "L1", "L2", "job"]].set_index("cell_line")
        les_pivot = les.set_index(['job'], append=True).unstack(['job'])
        les_pivot.columns = [f"{col[0]}_{col[1]}" for col in les_pivot.columns]

        pca = PCA(n_components=2)
        les_pca = pca.fit_transform(les_pivot.values)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"Explained variance for {context} {samples}: {explained_var:.4f}")
        explained_variance_ratios[(context, samples)] = explained_var

        results_dfs.append(pd.DataFrame(
            index=les_pivot.index,
            data=les_pca,
            columns=["L1", "L2"]
        ).assign(
            variance_explained=explained_var,
            context=context,
            samples=samples
        ))

    whole_pca_le = pd.concat(results_dfs)
    return whole_pca_le, explained_variance_ratios


def evaluate_svm_classifier_per_context(
    df: pd.DataFrame,
    metric: str,
    strategy: str = "binary",  # "binary", "quartiles_multi", or "categorical"
    n_splits: int = 5,
):
    results = {}

    for context, group in df.groupby("context"):
        valid = group[group[metric].notna()]
        if len(valid) < n_splits:
            results[context] = np.nan
            continue

        X = valid[["L1", "L2"]].values
        y = valid[metric]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Label transformation based on strategy
        if strategy == "binary":
            y_transformed = (y > np.median(y)).astype(int)
        elif strategy == "quartiles_multi":
            y_transformed = pd.qcut(y, 4, labels=False)
        elif strategy == "categorical":
            le = LabelEncoder()
            y_transformed = le.fit_transform(y)
        else:
            raise ValueError("Strategy must be 'binary', 'quartiles_multi', or 'categorical'.")

        if len(np.unique(y_transformed)) < 2:
            results[context] = np.nan
            continue

        model = SVC(kernel='linear')
        model.fit(X_scaled, y_transformed)
        scores = cross_val_score(model, X_scaled, y_transformed, cv=n_splits, scoring='accuracy')
        results[context] = scores.mean()

    return pd.Series(results, name="svm_accuracy")


def compute_local_marker_smoothness(df, n_neighbors=5, marker="ERBB2", is_categorical=False):
    results = {}

    for context, group in df.groupby("context"):
        valid = group[["L1", "L2", marker]].copy()
        values = valid[marker].values

        # If too few usable entries, skip
        if is_categorical:
            if pd.Series(values).notna().sum() <= n_neighbors:
                results[context] = np.nan
                continue
        else:
            if np.count_nonzero(~np.isnan(values)) <= n_neighbors:
                results[context] = np.nan
                continue

        coords = valid[["L1", "L2"]].values

        # Fit neighbors on all coordinates
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        smoothness_scores = []

        for i, neighbors in enumerate(indices):
            self_value = values[i]

            # Skip if self is NaN
            if pd.isna(self_value):
                continue

            neighbor_vals = values[neighbors[1:]]  # Exclude self

            if is_categorical:
                # Only compare with non-NaN neighbors
                neighbor_vals = [val for val in neighbor_vals if pd.notna(val)]
                if not neighbor_vals:
                    continue
                # Proportion of neighbors different from self
                disagreement_rate = np.mean([val != self_value for val in neighbor_vals])
                smoothness_scores.append(disagreement_rate)

            else:
                # Continuous: compute RMSD
                neighbor_vals = np.array(neighbor_vals)
                neighbor_vals = neighbor_vals[~np.isnan(neighbor_vals)]
                if neighbor_vals.size == 0:
                    continue
                diffs = neighbor_vals - self_value
                rmsd = np.sqrt(np.mean(diffs ** 2))
                smoothness_scores.append(rmsd)

        if not smoothness_scores:
            results[context] = np.nan
        else:
            score = np.mean(smoothness_scores)

            if is_categorical:
                results[context] = score  # [0,1] disagreement
            else:
                val_range = np.nanmax(values) - np.nanmin(values)
                results[context] = score / val_range if val_range > 0 else np.nan

    label = "mean_disagreement" if is_categorical else "normalized_rmsd"
    return pd.Series(results, name=label)
