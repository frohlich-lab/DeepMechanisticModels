import jax
import jax.numpy as jnp
import optimistix as optx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import squareform
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
    embeddings_df: pd.DataFrame, n_components: int = 2
) -> tuple[pd.DataFrame, dict[tuple[str, str], float]]:
    """
    Performs PCA on latent embeddings grouped by context and sample.

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        A single DataFrame containing all contexts, with at least:
        - 'context', 'samples', 'cell_line', 'L1', 'L2', 'job'

    n_components : int
        Number of PCA components to use, defaults to 2.

    Returns
    -------
    whole_pca_le : pd.DataFrame
        Concatenated PCA-reduced dataframes with added metadata.
    explained_variance_ratios : dict[tuple[str, str], float]
        Mapping from (context, samples) to total variance explained by first 2 PCs.
    """
    results_dfs = []
    explained_variance_ratios = {}

    for (context, samples, n_hidden), group_df in embeddings_df.groupby(["context", "samples", "n_hidden"]):
        les = group_df[
            ["cell_line", "job"] + [f"L{i}" for i in range(1, n_hidden+1)]
        ].set_index("cell_line")
        les_pivot = les.set_index(['job'], append=True).unstack(['job'])
        les_pivot.columns = [f"{col[0]}_{col[1]}" for col in les_pivot.columns]

        # Transform to mean 0 and variance 1
        vals = les_pivot.values
        vals -= vals.mean()
        vals /= vals.std()

        # PCA transform
        pca = PCA(n_components=n_components)
        les_pca = pca.fit_transform(vals)
        explained_var = pca.explained_variance_ratio_.sum()
        # Most variance (often by vast margin, e.g. 80/20%) is captured by first component
        print(f"Explained variance for {context} {samples}: {pca.explained_variance_ratio_}; Total: {explained_var:.4f}")
        explained_variance_ratios[(context, samples)] = explained_var

        results_dfs.append(pd.DataFrame(
            index=les_pivot.index,
            data=les_pca,
            columns=[f"L{i}" for i in range(1, n_components+1)],
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
        model=None,
):
    """
    Finds baseline input features corresponding to a target embedding or zero inflater output.

    baseline_mode options:
        - "zero_embedding": baseline input features map to null embeddings
        - "inflater_origin": baseline input features map to null inflated parameter deviations (through embeddings)
        - "zero_parameter_deviations": baseline input features map to null inflated parameter deviations (directly)

    Returns:
        input_features_baseline
    """

    if baseline_mode == "zero_embedding":
        # Target: zero vector in embedding space
        target = jnp.zeros_like(embeddings[0])

        def root_find_fn(x, _):
            return encoder(x) - target
    elif baseline_mode == "inflater_origin":
        # Target: specific embedding that gives zero sparse deviations
        assert model is not None, "Model required for 'inflater_origin' baseline."
        target = find_embedding_origin(model, embeddings)

        def root_find_fn(x, _):
            return encoder(x) - target
    elif baseline_mode == "zero_parameter_deviations":
        assert model is not None, "Model required for 'zero_parameter_deviations' baseline."
        mask = jnp.array(model.output_sparsity_binary_mask)
        # Target: zero vector in parameter deviation space
        target = jnp.zeros_like(mask)

        def root_find_fn(x, _):
            emb = encoder(x)
            inflated_param_devs = model.deep_inflater(emb) * mask
            return inflated_param_devs - target
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    # Solver setup
    solver = optx.Newton(rtol=1e-8, atol=1e-8)
    init_guess = jnp.mean(input_features, axis=0)

    # Solve for baseline input
    if baseline_mode in ["zero_embedding", "inflater_origin"]:
        sol = optx.root_find(fn=root_find_fn, y0=init_guess, solver=solver)
        input_features_baseline = sol.value
        emb_origin = encoder(input_features_baseline)

        print(
            f"[{baseline_mode}] Baseline input: {input_features_baseline}, "
            f"Embedding: {emb_origin}, "
            f"Target: {target}"
        )

        assert jnp.sum(jnp.abs(emb_origin - target)) < 1e-6, \
            f"{baseline_mode} embedding does not match target!"

    elif baseline_mode == "zero_parameter_deviations":
        sol = optx.root_find(fn=root_find_fn, y0=init_guess, solver=solver)
        input_features_baseline = sol.value
        deviations = model.deep_inflater(encoder(input_features_baseline)) * mask

        print(
            f"[zero_parameter_deviations] Baseline input: {input_features_baseline}, "
            f"Parameter deviations: {deviations}"
        )

        assert jnp.linalg.norm(deviations) < 1e-6, \
            "zero_parameter_deviations baseline does not yield zero deviations!"
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    return input_features_baseline


def integrated_gradients(
        input_x,
        model_fn,
        baseline_x=None,
        num_steps=100,
):
    if baseline_x is None:
        baseline_x = jnp.zeros_like(input_x)

    # Form straight path along which to integrate gradients
    alphas = jnp.linspace(0., 1., num_steps).reshape(-1, 1)
    interpolated = baseline_x + alphas * (input_x - baseline_x)

    jac_fn = jax.jacrev(model_fn)
    jacobians = jax.vmap(jac_fn)(interpolated)

    # Discrete integration (Riemann sum as in original paper)
    path_grad = jacobians.mean(axis=0)
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
        )

    # Apply IG to each rotated embedding
    rotated_embeddings_output = jax.vmap(ig_inflater_per_sample)(rotated_embeddings)

    # ======= Begin completeness and baseline checks =======

    # Compute the model outputs at each rotated embedding
    outputs_at_input = jax.vmap(inverse_rotate_then_mask_inflate)(rotated_embeddings)  # shape (N, D)

    # Compute outputs at baseline (zero embedding) — should be close to zero
    baseline_output = inverse_rotate_then_mask_inflate(jnp.zeros_like(rotated_embeddings[0]))  # shape (D,)
    baseline_norm = jnp.linalg.norm(baseline_output)

    # Check that baseline output is near zero (strict check)
    assert baseline_norm < 1e-5, f"Inflater baseline output is not near zero: norm = {baseline_norm}"

    # Compute sum of attributions over embedding dimensions (shape: N, P, D -> N, P)
    attribution_sums = rotated_embeddings_output.sum(axis=2)

    # Compute difference between attribution sum and output sum
    diff = jnp.abs(attribution_sums - outputs_at_input)

    # Assert that each difference is small (allow tolerance for numerical integration)
    assert jnp.all(diff < 1e-5), f"Inflater attribution completeness check failed! Max difference: {diff.max()}"

    # ======= End checks =======

    return rotated_embeddings, rotated_embeddings_output


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
        )

    # Step 4: Apply IG to all inputs
    ig_attributions = jax.vmap(ig_encoder_per_sample)(input_features)  # shape: (N, 2, D_in)

    # ======= Begin completeness and baseline checks =======

    # Step 5: Compute true rotated embeddings relative to baseline
    rotated_embeddings = jax.vmap(encode_then_rotate)(input_features)  # shape: (N, 2)

    # Step 6: Compute baseline rotated embedding → must be zero by construction
    baseline_embedding = encode_then_rotate(input_features_baseline)  # shape: (2,)
    baseline_norm = jnp.linalg.norm(baseline_embedding)

    assert baseline_norm < 1e-5, f"Encoder baseline embedding not near zero: norm = {baseline_norm}"

    # Step 7: Compute sum of attributions over input dimensions (axis=-1)
    attribution_sums = ig_attributions.sum(axis=2)  # shape: (N, 2)

    # Step 8: Compute difference between IG sums and actual rotated embeddings
    diff = jnp.abs(attribution_sums - rotated_embeddings)  # shape: (N, 2)

    max_diff = diff.max()
    assert jnp.all(diff < 1e-5), f"Encoder attribution completeness failed! Max diff: {max_diff}"

    # ======= End checks =======

    return jax.vmap(ig_encoder_per_sample)(input_features)  # (N, 2, D_in)


def run_full_model_ig_attributions(
        encoder,
        model,
        input_features,
        embeddings,
        baseline_mode: str = "zero_parameter_deviations",
        num_steps: int = 100,
):
    """
    Computes Integrated Gradients (IG) attributions from input_features all the way to the
    sparse parameter deviations (inflater outputs), applying the encoder + inflater sequentially.

    Checks:
    - Baseline output is near zero.
    - IG attribution sums match actual output deviations.

    Returns:
        ig_attributions: per-sample, per-output, per-input attributions
    """
    mask = jnp.array(model.output_sparsity_binary_mask)  # shape: (P,)

    # Step 1: Baseline in input space → mapped through encoder → embedding → inflater
    input_features_baseline = find_input_features_baseline(
        encoder=encoder,
        input_features=input_features,
        embeddings=embeddings,
        baseline_mode=baseline_mode,
        model=model
    )

    # Step 2: Define full model function: input_features → embedding → inflated deviation
    def full_model_fn(x):
        embedding = encoder(x)
        inflated = model.deep_inflater(embedding) * mask
        return inflated

    # Step 3: Define per-sample IG function (returns (P, D_in))
    def ig_full_per_sample(x):
        return integrated_gradients(
            input_x=x,
            model_fn=full_model_fn,
            baseline_x=input_features_baseline,
            num_steps=num_steps,
        )

    # Step 4: Apply IG across all inputs
    ig_attributions = jax.vmap(ig_full_per_sample)(input_features)  # shape: (N, P, D_in)

    # ======= Begin completeness and baseline checks =======

    # Step 6: Compute true model outputs at inputs and baseline
    outputs_at_input = jax.vmap(full_model_fn)(input_features)  # shape: (N, P)
    baseline_output = full_model_fn(input_features_baseline)    # shape: (P,)
    baseline_norm = jnp.linalg.norm(baseline_output)

    assert baseline_norm < 1e-5, f"Full model baseline output not near zero: norm = {baseline_norm}"

    # Step 7: Compute sum of attributions over input dimensions (axis=-1)
    attribution_sums = ig_attributions.sum(axis=2)  # shape: (N, P)

    # Step 8: Compute difference between attribution sums and model outputs
    diff = jnp.abs(attribution_sums - outputs_at_input)  # shape: (N, P)

    max_diff = diff.max()
    assert jnp.all(diff < 1e-5), f"Full model attribution completeness failed! Max diff: {max_diff}"

    # ======= End checks =======

    return ig_attributions  # shape: (N, P, D_in)


def build_canonical_latents_from_params_and_pca(
    mean_par_dev_df: pd.DataFrame,
    pca_embedding_df: pd.DataFrame,
    model: str = "EGFR_MAPK__logobs",
    context: str = "cytof_init",
    samples: str = "all",
    meta_cols=None,
    min_modules: int = 2,
    max_modules: int = 15,
    n_accepted_singletons: int = 1,
    plot: bool = True,
):
    """
    End-to-end pipeline to derive 'canonical' latent directions from parameter
    deviations and align PCA-ed latent embeddings to them.

    Parameters
    ----------
    mean_par_dev_df : pd.DataFrame
        DataFrame with mean parameter deviations per cell_line (and metadata).
    pca_embedding_df : pd.DataFrame
        DataFrame with PCA-ed latent embeddings (columns L1, L2, ...) per cell_line.
    model : str
        Model name to filter on (column 'model').
    context : str
        Context name to filter on (column 'context').
    samples : str
        Samples label to filter on (column 'samples').
    meta_cols : list or None
        List of metadata columns that should NOT be treated as parameters.
        If None, defaults to ["model", "context", "samples", "cell_line", "n_hidden"].
    min_modules : int
        Minimum number of modules (clusters) to consider.
    max_modules : int
        Maximum number of modules (clusters) to consider.
    n_accepted_singletons : int
        Maximum number of singleton clusters (clusters of size 1) allowed
        when choosing K. The function scans K downward from max_modules until
        it finds a K with at most this many singletons.
    plot : bool
        If True, produce diagnostic plots.

    Logic
    -----
    1. Filter `mean_par_dev_df` and `pca_embedding_df` to the given
       (model, context, samples).

    2. Identify parameter modules:
       - Use only non-metadata, non-zero-variance parameter columns.
       - Compute the parameter–parameter correlation matrix.
       - Build a distance matrix as D = 1 - |corr|.
       - Perform hierarchical clustering (average linkage) on D.

    3. Select the number of modules K (denoted K_opt):
       - Let K_start = min(max_modules, n_params - 1),
         K_min = max(min_modules, 2).
       - For K = K_start, K_start-1, ..., K_min:
         * Cut the dendrogram into K clusters (fcluster).
         * Count the number of singleton clusters (size 1).
         * Compute a silhouette score on the precomputed distance matrix
           (stored for diagnostics only).
         * Stop at the first K with <= n_accepted_singletons singletons.
       - If no K satisfies the singleton constraint, fall back to K_min.
       - K_opt is the chosen K.

    4. For K_opt:
       - Assign each parameter to a module (cluster id 1...K_opt).
       - For each module:
         * Standardize its parameters across cell lines.
         * Take PC1 across cell lines (module eigenscore).
       - Collect these into `module_latents_df`
         (index = cell_line, columns = mod1...modK).

    5. Align PCA-ed latent embeddings to module latents:
       - Extract latent columns L1, L2, ... from `pca_embedding_df`.
       - Restrict to cell_lines present in both embeddings and module latents.
       - Fit a linear regression M ≈ Z @ A + b, where:
         * Z = PCA latents (cell_line × n_L)
         * M = module_latents_df (cell_line × n_modules)
       - Apply this map to Z to obtain `real_latents_df`
         (cell_line × n_modules), i.e. module-aligned "real" latents.

    6. Optional plotting (if `plot=True`):
       - Heatmap of |corr| ordered by dendrogram leaves.
       - Silhouette score vs K (diagnostic; K_opt is NOT chosen by silhouette).
       - Heatmap of correlations between canonical modules and latent PCs.

    Returns
    -------
    dict
        {
            "corr": corr,                         # parameter correlation matrix
            "dist": dist,                         # distance matrix D = 1 - |corr|
            "Z_linkage": Z_linkage,               # linkage from hierarchical clustering
            "param_to_module": param_to_module,   # Series: param -> module id (1...K_opt)
            "K_opt": K_opt,                       # chosen K after singleton constraint
            "sil_scores": sil_scores,             # dict K -> silhouette score
            "module_latents_df": module_latents_df,   # cell_line × modules (param-derived)
            "real_latents_df": real_latents_df,       # cell_line × modules (from latent PCs)
            "A": A,                               # weights in M ≈ Z @ A + b
            "b": b,                               # intercept in M ≈ Z @ A + b
        }
    """

    # ------------------------------------------------------------------
    # 0. Default metadata columns (not treated as parameters)
    # ------------------------------------------------------------------
    if meta_cols is None:
        meta_cols = ["model", "context", "samples", "cell_line", "n_hidden"]

    # ------------------------------------------------------------------
    # 1. Filter dataframes by model/context/samples
    # ------------------------------------------------------------------
    par_mean_sub = mean_par_dev_df[
        (mean_par_dev_df.model == model) &
        (mean_par_dev_df.context == context) &
        (mean_par_dev_df.samples == samples)
    ].copy()

    lat_sub = pca_embedding_df[
        (pca_embedding_df.model == model) &
        (pca_embedding_df.context == context) &
        (pca_embedding_df.samples == samples)
    ].copy()

    if par_mean_sub.empty:
        raise ValueError("Filtered mean_par_dev_df is empty for given model/context/samples")
    if lat_sub.empty:
        raise ValueError("Filtered pca_embedding_df is empty for given model/context/samples")

    # ------------------------------------------------------------------
    # 2. Parameter correlation & hierarchical clustering
    # ------------------------------------------------------------------
    # Parameter columns: non-metadata, with non-zero variance
    param_cols = [c for c in par_mean_sub.columns if c not in meta_cols]
    std_nonzero = par_mean_sub[param_cols].std()
    params_nonzero = std_nonzero[std_nonzero > 0].index.tolist()

    if len(params_nonzero) < 2:
        raise ValueError("Not enough non-zero-variance parameters to build modules.")

    corr = par_mean_sub[params_nonzero].corr()

    # Distance = 1 - |corr|
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist.values, 0)

    # Linkage on condensed distance matrix
    dist_condensed = squareform(dist.values)
    Z_linkage = linkage(dist_condensed, method="average")

    # ------------------------------------------------------------------
    # 3. Choose K by scanning downward until singleton constraint is met
    # ------------------------------------------------------------------
    sil_scores = {}

    n_params = len(params_nonzero)
    K_start = min(max_modules, n_params - 1)
    K_min = max(min_modules, 2)

    chosen_K = None
    labels_series = None
    n_singletons = None

    # Scan K downward: K_start, K_start-1, ..., K_min
    for K in range(K_start, K_min - 1, -1):
        labels = fcluster(Z_linkage, t=K, criterion="maxclust")
        labels_series = pd.Series(labels, index=corr.index)

        # Cluster sizes + singleton count
        cluster_sizes = labels_series.value_counts()
        n_singletons = (cluster_sizes == 1).sum()

        # Store silhouette (for diagnostics / plotting)
        score = silhouette_score(dist.values, labels, metric="precomputed")
        sil_scores[K] = score

        # Stop as soon as we have at most n_accepted_singletons singletons
        if n_singletons <= n_accepted_singletons:
            chosen_K = K
            break

    # If we never satisfied the singleton constraint, just use K_min
    if chosen_K is None:
        chosen_K = K_min
        labels = fcluster(Z_linkage, t=chosen_K, criterion="maxclust")
        labels_series = pd.Series(labels, index=corr.index)
        cluster_sizes = labels_series.value_counts()
        n_singletons = (cluster_sizes == 1).sum()

    K_opt = chosen_K
    print(f"Chosen K_opt={K_opt}, #singletons={n_singletons}")

    # Final module assignment
    param_to_module = labels_series.rename("module")

    # ------------------------------------------------------------------
    # 3b. Order parameters according to dendrogram leaves (for plotting)
    # ------------------------------------------------------------------
    order = leaves_list(Z_linkage)
    ordered_params = corr.index[order]
    corr_ordered = corr.loc[ordered_params, ordered_params]

    # ------------------------------------------------------------------
    # 4. Compute module eigenscores (PC1 per module) per cell_line
    # ------------------------------------------------------------------
    par_mean_sub = par_mean_sub.set_index("cell_line").sort_index()

    module_scores = {}
    for m in sorted(param_to_module.unique()):
        params_m = param_to_module.index[param_to_module == m]
        X = par_mean_sub[params_m]
        X_std = (X - X.mean()) / X.std()

        pca_m = PCA(n_components=1)
        score_m = pca_m.fit_transform(X_std)[:, 0]
        module_scores[f"mod{m}"] = score_m

    module_latents_df = pd.DataFrame(
        module_scores,
        index=par_mean_sub.index
    ).sort_index()

    # ------------------------------------------------------------------
    # 5. Align PCA-ed latent embeddings to module latents
    # ------------------------------------------------------------------
    lat_sub = lat_sub.set_index("cell_line").sort_index()
    lat_cols = [c for c in lat_sub.columns if c.startswith("L")]

    if not lat_cols:
        raise ValueError("No latent PCA columns starting with 'L' found in pca_embedding_df")

    # Common cell lines between latents and module latents
    common_cells = lat_sub.index.intersection(module_latents_df.index)
    if len(common_cells) == 0:
        raise ValueError("No overlapping cell_line entries between embeddings and parameter means.")

    Z = lat_sub.loc[common_cells, lat_cols].values
    M = module_latents_df.loc[common_cells].values

    reg = LinearRegression(fit_intercept=True)
    reg.fit(Z, M)
    A = reg.coef_.T         # shape: (n_latent, n_modules)
    b = reg.intercept_      # shape: (n_modules,)

    real_latents = Z @ A + b
    real_latents_df = pd.DataFrame(
        real_latents,
        index=common_cells,
        columns=module_latents_df.columns
    )

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    if plot:
        sns.set(style="white")

        # 6.1 correlation heatmap ordered by modules
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            np.abs(corr_ordered),
            cmap="coolwarm",
            center=0,
            xticklabels=False,
            yticklabels=False
        )
        plt.title(f"Parameter |corr| (K_opt={K_opt} modules, {model}, {context})")
        plt.tight_layout()

        # 6.2 silhouette vs K (diagnostic)
        if len(sil_scores) > 0:
            plt.figure(figsize=(6, 4))
            Ks = sorted(sil_scores.keys())
            scores = [sil_scores[K] for K in Ks]
            plt.plot(Ks, scores, marker="o")
            plt.xlabel("Number of modules (K)")
            plt.ylabel("Silhouette score (precomputed distance)")
            plt.title("Module number diagnostics")
            plt.grid(True)
            plt.tight_layout()

        # 6.3 correlation heatmap: real_latents vs latent PCs
        real_latents_aligned = real_latents_df.loc[common_cells]
        lat_aligned = lat_sub.loc[common_cells, lat_cols]

        corr_mod_lat = pd.DataFrame(
            np.corrcoef(
                real_latents_aligned.values.T,
                lat_aligned.values.T
            ),
        )

        # Build a nicer labeled matrix: rows = modules, cols = Ls
        n_mod = real_latents_aligned.shape[1]
        n_lat = lat_aligned.shape[1]
        corr_block = corr_mod_lat.iloc[:n_mod, n_mod:n_mod + n_lat].copy()
        corr_block.index = real_latents_aligned.columns
        corr_block.columns = lat_cols

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_block,
            cmap="vlag",
            center=0,
            annot=True,
            fmt=".2f"
        )
        plt.title("Correlation between canonical modules and latent PCs")
        plt.tight_layout()

    # ------------------------------------------------------------------
    # 7. Return everything useful
    # ------------------------------------------------------------------
    return {
        "corr": corr,
        "dist": dist,
        "Z_linkage": Z_linkage,
        "param_to_module": param_to_module,
        "K_opt": K_opt,
        "sil_scores": sil_scores,
        "module_latents_df": module_latents_df,
        "real_latents_df": real_latents_df,
        "A": A,
        "b": b,
    }
