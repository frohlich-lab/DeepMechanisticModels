import jax
import jax.numpy as jnp
import optimistix as optx
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.integrate import trapezoid
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
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


def find_embedding_rotation(
        embeddings,
        pher2_values
):
    # Only use valid (non-missing) pHER2 values
    valid_mask = ~jnp.isnan(pher2_values)
    X_valid = embeddings[valid_mask]
    y_valid = pher2_values[valid_mask]
    # Fit a linear model to predict pHER2 from embeddings
    linreg = LinearRegression()
    linreg.fit(X_valid, y_valid)
    # Extract gradient direction, normalise and find second orthogonal vector
    w = linreg.coef_
    v1 = w / jnp.linalg.norm(w)
    v2 = jnp.array([-v1[1], v1[0]])
    # Build and return rotation matrix
    return jnp.stack([v1, v2], axis=1)


def find_embedding_origin(
        model,
        input_embeddings
):
    mask = jnp.array(model.output_sparsity_binary_mask)

    def masked_inflater_optx(x, _):  # Accepts second arg even if unused
        return model.deep_inflater(x) * mask

    solver = optx.Newton(rtol=1e-8, atol=1e-8)
    x0_init = jnp.mean(input_embeddings, axis=0)
    sol = optx.root_find(
        fn=masked_inflater_optx,
        y0=x0_init,
        solver=solver
    )
    return sol.value


def rotate_embeddings(
        embeddings,
        x0,
        R,
        check_inverse=True
):
    rotated = (embeddings - x0) @ R
    if check_inverse:
        reconstructed = rotated @ R.T + x0
        diff = jnp.sum(jnp.abs(reconstructed - embeddings))
        assert diff < 1e-6, f"Inverse rotation mismatch: {diff}"
    return rotated


def find_input_features_baseline(
        encoder,
        input_features,
        embeddings,
        baseline_mode: str = "inflater_origin",
        model = None,
):
    if baseline_mode == "zero_embedding":
        target_embedding = jnp.zeros_like(embeddings[0])
    elif baseline_mode == "inflater_origin":
        assert model is not None, "Model required for 'inflater_origin' baseline."
        target_embedding = find_embedding_origin(model, embeddings)
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    def encoder_target_diff(x, _):
        return encoder(x) - target_embedding

    solver = optx.Newton(rtol=1e-8, atol=1e-8)
    init_guess = jnp.mean(input_features, axis=0)
    sol = optx.root_find(fn=encoder_target_diff, y0=init_guess, solver=solver)
    input_features_baseline = sol.value
    emb_origin = encoder(input_features_baseline)

    print(
        f"Baseline input features found: {input_features_baseline}, "
        f"with corresponding baseline/origin embedding: {emb_origin}, "
        f"whereas the target embedding was: {target_embedding}"
    )

    assert jnp.sum(jnp.abs(emb_origin - target_embedding)) < 1e-6, \
        "Baseline embedding does not match target embedding!"
    return input_features_baseline


def integrated_gradients(
        input_x,
        model_fn,
        baseline_x=None,
        num_steps=100,
        method="riemann"
):
    if baseline_x is None:
        baseline_x = jnp.zeros_like(input_x)

    # Form straight path along which to integrate gradients
    alphas = jnp.linspace(0., 1., num_steps).reshape(-1, 1)
    interpolated = baseline_x + alphas * (input_x - baseline_x)

    jac_fn = jax.jacrev(model_fn)
    jacobians = jax.vmap(jac_fn)(interpolated)

    # Discrete integration (default: Riemann sum as in original paper)
    if method == "riemann":
        path_grad = jacobians.mean(axis=0)
    elif method == "trapezoid":
        path_grad = trapezoid(jacobians, dx=1.0/num_steps, axis=0)  # (M, 2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return (input_x - baseline_x) * path_grad


def run_inflater_ig_attributions(
        model,
        input_embeddings,
        pher2_values,
        num_steps=100
):
    mask = jnp.array(model.output_sparsity_binary_mask)
    R = find_embedding_rotation(input_embeddings, pher2_values)
    x0 = find_embedding_origin(model, input_embeddings)
    rotated_embeddings = rotate_embeddings(input_embeddings, x0, R)

    def inverse_rotate_then_mask_inflate(x_rot):
        # Reverse rotation and translation to apply inflater to original embeddings,
        # while computing gradients in rotated space
        return model.deep_inflater(x_rot @ R.T + x0) * mask

    def ig_inflater_per_sample(x):
        return integrated_gradients(
            input_x=x,
            model_fn=inverse_rotate_then_mask_inflate,
            baseline_x=jnp.zeros_like(x),
            num_steps=num_steps,
            method="riemann"
        )

    return rotated_embeddings, jax.vmap(ig_inflater_per_sample)(rotated_embeddings)


def run_encoder_ig_attributions(
        encoder,
        input_features,
        embeddings,
        pher2_values,
        model=None,
        baseline_mode="inflater_origin",
        num_steps: int = 100,
):
    R = find_embedding_rotation(embeddings, pher2_values)
    input_features_baseline = find_input_features_baseline(
        encoder=encoder,
        input_features=input_features,
        embeddings=embeddings,
        baseline_mode=baseline_mode,
        model=model
    )

    def encode_then_rotate(x):
        emb = encoder(x)
        return (emb - encoder(input_features_baseline)) @ R

    def ig_encoder_per_sample(x):
        return integrated_gradients(
            input_x=x,
            model_fn=encode_then_rotate,
            baseline_x=input_features_baseline,
            num_steps=num_steps,
            method="riemann"
        )

    return jax.vmap(ig_encoder_per_sample)(input_features)  # (N, 2, D_in)
