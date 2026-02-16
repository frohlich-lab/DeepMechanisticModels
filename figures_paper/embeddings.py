import hashlib
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to sys.path so that modules in the root directory can be imported
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

warnings.filterwarnings("ignore")

from figure_config import (
    get_context_label,
    get_cytof_marker_label,
    get_model_label,
)

from annotation_utils import load_marcotte_subtypes
from common import basedir, evaluations_dir
from embedding_utils import perform_pca_on_embeddings
from training_configuration import (
    LATENT_DIMS,
    NETWORK_DEPTH,
    PATHWAYS_BY_FIGURE,
)

# Cache directory for correlation results
CACHE_DIR = basedir / "figures_paper" / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

sensitive_pars = [
    "ERBB2_p_Y1248_ERBB2_kw",  # L1/2
    "ERBB2_p_Y1248_EGFR__Y1173_p_kw",  # L1/2
    "ERK_p_Y204_MEK__S222_p_kw",  # L2/3
    "ERK_p_Y204_TOPK__Y74_p_kw",  # L2/3
    "RPS6KA1_p_S380_ERK__Y204_p_kw",  # L1/4
]

sensitive_dirs = {
    "D1": ("ERBB2_p_Y1248_ERBB2_kw", "ERBB2_p_Y1248_EGFR__Y1173_p_kw"),
    "D2": ("ERK_p_Y204_MEK__S222_p_kw", "ERK_p_Y204_TOPK__Y74_p_kw"),
    "D3": ("RPS6KA1_p_S380_ERK__Y204_p_kw",),
}


def _get_cache_key(model, context, figure, data, data_type, n_hvg=None):
    """Generate a unique cache key for correlation results."""
    key_str = f"{model}_{context}_{figure}_{data}_{data_type}_hvg{n_hvg}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_from_cache(cache_key):
    """Load cached correlation results if available."""
    cache_file = CACHE_DIR / f"correlations_{cache_key}.tsv"
    if cache_file.exists():
        return pd.read_csv(cache_file, sep="\t", index_col=0)
    return None


def _save_to_cache(cache_key, df):
    """Save correlation results to cache as TSV file."""
    if not df.empty:
        cache_file = CACHE_DIR / f"correlations_{cache_key}.tsv"
        df.to_csv(cache_file, sep="\t")


def load_all_marker_data(data_type: str = "cytof", markers: list = None):
    """Load all marker data from cytof/proteomics/transcriptomics.

    Args:
        data_type: Type of data - "cytof", "proteomics", or "transcriptomics"
        markers: Optional list of markers to load (for performance). If None, loads all.

    Returns:
        DataFrame with cell_line as index and markers as columns
    """
    data_dir = basedir / "data"

    if data_type == "cytof":
        df = pd.read_csv(data_dir / "cytof.csv", index_col=0)
        # Filter to baseline (time=0) and EGF condition, aggregate across replicates
        marker_df = df[
            (df["time"] == 0.0)
            & (df["simulationConditionId"].str.endswith("__EGF"))
        ]
        if markers is not None:
            marker_df = marker_df[marker_df["observableId"].isin(markers)]
        # Pivot to get cell lines as rows and markers as columns
        marker_values = (
            marker_df.groupby(["preequilibrationConditionId", "observableId"])[
                "measurement"
            ]
            .mean()
            .unstack()
        )
    elif data_type == "proteomics":
        df = pd.read_csv(data_dir / "proteomics.csv", index_col=0)
        if markers is not None:
            df = df[df["observableId"].isin(markers)]
        marker_values = df.pivot_table(
            index="preequilibrationConditionId",
            columns="observableId",
            values="measurement",
        )
    elif data_type == "transcriptomics":
        df = pd.read_csv(data_dir / "transcriptomics.csv", index_col=0)
        if markers is not None:
            df = df[df["observableId"].isin(markers)]
        marker_values = df.pivot_table(
            index="preequilibrationConditionId",
            columns="observableId",
            values="measurement",
        )
    else:
        raise ValueError(
            f"Unknown data_type: {data_type}. Use 'cytof', 'proteomics', or 'transcriptomics'."
        )

    return marker_values


def load_marker_data(marker: str, data_type: str = "cytof"):
    """Load marker data from cytof/proteomics/transcriptomics.

    Args:
        marker: Marker/gene name (e.g., "pEGFR", "EGFR", "TP53")
        data_type: Type of data - "cytof", "proteomics", or "transcriptomics"

    Returns:
        Series with cell_line as index and marker values
    """
    # Load only the specific marker for performance
    all_markers = load_all_marker_data(data_type, markers=[marker])

    if marker not in all_markers.columns:
        raise ValueError(f"Marker '{marker}' not found in {data_type} data")

    marker_values = all_markers[marker]
    marker_values.name = marker

    return marker_values


def get_available_parameters(
    model, figure: str = "figure3", data: str = "dream_cytof"
):
    """Get list of available parameters for a model.

    Args:
        model: Model name
        figure: Figure name (default: "figure3")
        data: Dataset name (default: "dream_cytof")

    Returns:
        List of parameter names
    """
    filepath = evaluations_dir / model / data / f"param_devs_{figure}.csv"

    if not filepath.exists():
        raise ValueError(f"Parameter file not found: {filepath}")

    df = pd.read_csv(filepath, index_col=0, nrows=1)

    # Exclude metadata columns
    metadata_cols = {
        "cell_line",
        "context",
        "samples",
        "dataset",
        "job",
        "n_hidden",
        "depth",
        "dropout_rate",
        "nn_init_scale",
        "l1reg_inflate",
        "oreg_inflate",
        "l1reg_encode",
        "oreg_encode",
        "l1reg_inflater_output",
        "l2reg_inflater_output",
        "recon_loss",
        "symm_reg",
        "inflater_output_reg_epoch",
        "n_epoch",
        "inflater_bound",
        "features",
        "multiheaded",
    }

    return [c for c in df.columns if c not in metadata_cols]


def load_parameter_data(
    model,
    parameter: str,
    figure: str = "figure3",
    data: str = "dream_cytof",
    context: str = "cytof_init",
):
    """Load parameter deviation data for a specific parameter.

    Args:
        model: Model name
        parameter: Parameter name (e.g., "ERBB2_p_Y1248_bact")
        figure: Figure name (default: "figure3")
        data: Dataset name (default: "dream_cytof")
        context: Context to filter by (default: "cytof_init")

    Returns:
        Series with cell_line as index and parameter values
    """
    # Validate parameter exists
    available_params = get_available_parameters(model, figure, data)
    if parameter not in available_params:
        raise ValueError(
            f"Parameter '{parameter}' not found. Available parameters: {available_params[:10]}..."
        )

    filepath = evaluations_dir / model / data / f"param_devs_{figure}.csv"
    df = pd.read_csv(filepath, index_col=0)

    # Filter by context if present
    if "context" in df.columns:
        df = df[df["context"] == context]

    # Filter to test set (samples starting with "all")
    if "samples" in df.columns:
        df = df[df["samples"].str.startswith("all")]

    # Average across jobs if multiple
    result = df.groupby("cell_line")[parameter].mean()
    result.name = parameter

    return result


def load_embedding_data(figure: str, data: str = "dream_cytof"):
    """Load embedding data for a specific figure.

    Args:
        figure: Figure name (e.g., 'figure1a', 'figure3')
        data: Dataset name

    Returns:
        DataFrame with all embeddings for the figure
    """
    pathways = PATHWAYS_BY_FIGURE.get(figure, [])

    dfs = []
    for model in pathways:
        filepath = evaluations_dir / model / data / f"embeddings_{figure}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0)
            df["model"] = model
            dfs.append(df)

    if not dfs:
        raise ValueError(f"No embedding data found for figure '{figure}'")

    embedding_df = pd.concat(dfs)

    # Only keep central values & test set
    embedding_df = embedding_df[
        (embedding_df.depth == NETWORK_DEPTH["central_value"])
        & (embedding_df.n_hidden == LATENT_DIMS["central_value"])
        & (embedding_df.samples.str.startswith("all"))
    ]

    return embedding_df


def prepare_pca_embeddings(embedding_df, subtypes_df=None):
    """Perform PCA on embeddings and optionally add subtype annotations.

    Args:
        embedding_df: DataFrame with raw embeddings
        subtypes_df: DataFrame with subtype annotations (optional)

    Returns:
        DataFrame with PCA embeddings and annotations
    """
    models = embedding_df["model"].unique()

    pca_embeddings_dict = {}
    for model in models:
        pca_embeddings_dict[model], _ = perform_pca_on_embeddings(
            embedding_df[embedding_df.model == model],
            n_components=LATENT_DIMS["central_value"],
        )

    # Add subtype annotations if provided
    if subtypes_df is not None:
        for model in models:
            for col in subtypes_df.columns:
                pca_embeddings_dict[model][col] = subtypes_df[col]

            # Create luminalbasal column if subtype_intrinsic exists
            if "subtype_intrinsic" in subtypes_df.columns:
                pca_embeddings_dict[model][
                    "luminalbasal"
                ] = pca_embeddings_dict[model]["subtype_intrinsic"].copy()
                pca_embeddings_dict[model]["luminalbasal"].replace(
                    ["LuminalA", "LuminalB", "HER2"], "Luminal", inplace=True
                )
                pca_embeddings_dict[model]["luminalbasal"].replace(
                    ["CL"], "Basal", inplace=True
                )

    pca_embedding_df = pd.concat(
        [pca_embeddings_dict[model].assign(model=model) for model in models]
    )

    return pca_embedding_df


def plot_embeddings(
    model,
    context,
    figure="figure3",
    data="dream_cytof",
    color_by="luminalbasal",
    marker_data_type=None,
    palette="tab10",
    cmap="RdBu_r",
    x_col="L1",
    y_col="L2",
    figsize=(20, 6),
    column_ratios=None,
    save_path=None,
):
    """Plot PCA embeddings colored by subtype information, marker expression, or parameters.

    Args:
        model: Model name to plot
        context: Context to plot
        figure: Figure name for loading data (default: "figure3")
        data: Dataset name (default: "dream_cytof")
        color_by: Column name for coloring (e.g., "luminalbasal", "subtype_intrinsic")
                  or marker/parameter name if marker_data_type is specified
        marker_data_type: If specified, load data from "cytof", "proteomics",
                          "transcriptomics", or "parameter" and use continuous coloring
        palette: Color palette for discrete coloring (default: "tab10")
        cmap: Colormap for continuous marker coloring (default: "RdBu_r", centered on mean)
        x_col: Column to plot on x-axis (default: "L1")
        y_col: Column to plot on y-axis (default: "L2")
        figsize: Figure size tuple (width, height)
        column_ratios: Custom width ratios for columns [left, plot, right] (default: [1, 1, 1])
        save_path: Path to save the figure (None = don't save)

    Returns:
        tuple: (fig, ax)
    """
    # Load and prepare data
    embedding_df = load_embedding_data(figure, data)
    subtypes_df = load_marcotte_subtypes(embedding_df.cell_line.unique())
    pca_embedding_df = prepare_pca_embeddings(embedding_df, subtypes_df)

    # Filter data
    plot_df = pca_embedding_df[
        (pca_embedding_df["model"] == model)
        & (pca_embedding_df["context"] == context)
    ].copy()

    # Handle continuous marker/parameter coloring
    continuous_color = False
    if marker_data_type is not None:
        if marker_data_type == "parameter":
            color_values = load_parameter_data(
                model, color_by, figure, data, context
            )
        else:
            color_values = load_marker_data(color_by, marker_data_type)
        plot_df[color_by] = plot_df.index.map(color_values)
        continuous_color = True

    # Create figure with 3 columns
    fig = plt.figure(figsize=figsize)
    ratios = column_ratios if column_ratios is not None else [1, 1, 1]
    gs = fig.add_gridspec(1, 3, width_ratios=ratios)
    ax_left = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    # Hide left and right axes
    ax_left.axis("off")
    ax_right.axis("off")

    if continuous_color:
        # Continuous coloring for markers - center on mean
        values = plot_df[color_by].dropna()
        mean_val = values.mean()
        max_dev = max(
            abs(values.min() - mean_val), abs(values.max() - mean_val)
        )
        vmin = mean_val - max_dev
        vmax = mean_val + max_dev

        scatter = ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            c=plot_df[color_by],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="black",
            linewidths=0.5,
            s=50,
        )

        # Compute gradient direction using linear regression (for drawing line later)
        from sklearn.linear_model import LinearRegression

        valid_mask = plot_df[color_by].notna()
        X = plot_df.loc[valid_mask, [x_col, y_col]].values
        y_vals = plot_df.loc[valid_mask, color_by].values
        reg = LinearRegression().fit(X, y_vals)
        gradient = reg.coef_  # [dx, dy] direction of steepest increase

        # Orthogonal direction (perpendicular to gradient)
        ortho = [-gradient[1], gradient[0]]
        ortho_norm = (ortho[0] ** 2 + ortho[1] ** 2) ** 0.5
        ortho = [ortho[0] / ortho_norm, ortho[1] / ortho_norm]

        # Add colorbar in the right subplot
        cbar = fig.colorbar(scatter, ax=ax_right, fraction=0.5, pad=0.1)
        cbar.set_label(f"{color_by} ({marker_data_type})")
    else:
        # Discrete coloring for subtypes
        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue=color_by,
            palette=palette,
            edgecolor="black",
            ax=ax,
        )
        # Place legend in the right subplot
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()
        ax_right.legend(
            handles,
            labels,
            title=color_by.replace("_", " ").title(),
            loc="center left",
            frameon=False,
        )

    # Axis labels
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{get_model_label(model)} - {get_context_label(context)}")

    # Center plot at 0 with symmetric limits
    x_max = max(abs(plot_df[x_col].min()), abs(plot_df[x_col].max()))
    y_max = max(abs(plot_df[y_col].min()), abs(plot_df[y_col].max()))
    ax.set_xlim(-x_max * 1.1, x_max * 1.1)
    ax.set_ylim(-y_max * 1.1, y_max * 1.1)

    # Draw line orthogonal to gradient for continuous markers
    if continuous_color:
        line_length = max(x_max, y_max) * 1.5
        ax.plot(
            [-ortho[0] * line_length, ortho[0] * line_length],
            [-ortho[1] * line_length, ortho[1] * line_length],
            color="black",
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove default seaborn grid/spines
    sns.despine(left=True, bottom=True)

    # Draw clean axes at (0,0) - thicker, black, and behind data points
    ax.axhline(0, color="black", linewidth=2, zorder=0)
    ax.axvline(0, color="black", linewidth=2, zorder=0)
    ax.grid(False)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def compute_marker_embedding_correlations(
    model,
    context,
    data_type,
    figure="figure3",
    data="dream_cytof",
    n_latents=4,
    use_cache=True,
    n_hvg=5000,
):
    """Compute correlations between all markers and embedding dimensions L1-L4.

    Args:
        model: Model name
        context: Context name
        data_type: Type of marker data - "cytof", "proteomics", or "transcriptomics"
        figure: Figure name for loading data
        data: Dataset name
        n_latents: Number of latent dimensions to compute correlations for (default: 4)
        use_cache: Whether to use cached results (default: True)
        n_hvg: Number of highly variable genes to use for proteomics/transcriptomics
               (default: 5000). Set to None to use all genes. Ignored for cytof.

    Returns:
        DataFrame with markers as rows and L1-L4 correlations as columns
    """
    # Check cache first
    cache_key = _get_cache_key(model, context, figure, data, data_type, n_hvg)
    if use_cache:
        cached = _load_from_cache(cache_key)
        if cached is not None:
            return cached

    # Load and prepare embedding data
    embedding_df = load_embedding_data(figure, data)
    subtypes_df = load_marcotte_subtypes(embedding_df.cell_line.unique())
    pca_embedding_df = prepare_pca_embeddings(embedding_df, subtypes_df)

    # Filter to specific model and context
    emb_df = pca_embedding_df[
        (pca_embedding_df["model"] == model)
        & (pca_embedding_df["context"] == context)
    ].copy()

    latent_cols = [f"L{i}" for i in range(1, n_latents + 1)]

    try:
        marker_data = load_all_marker_data(data_type)

        # Subset to highly variable genes for proteomics/transcriptomics
        if (
            data_type in ["proteomics", "transcriptomics"]
            and n_hvg is not None
        ):
            n_total = len(marker_data.columns)
            if n_total > n_hvg:
                hvg = get_highly_variable_genes(marker_data, n_top=n_hvg)
                marker_data = marker_data[hvg]
                print(
                    f"Subset {data_type} to {len(hvg)} highly variable genes (from {n_total})"
                )

        # Find common cell lines
        common_cells = list(set(emb_df.index) & set(marker_data.index))

        if len(common_cells) < 5:
            print(
                f"Warning: Only {len(common_cells)} common cell lines for {data_type}"
            )
            return pd.DataFrame()

        # Subset to common cell lines
        emb_subset = emb_df.loc[common_cells, latent_cols]
        marker_subset = marker_data.loc[common_cells]

        # Compute correlations for each marker
        correlations = {}
        pvalues = {}
        for marker in marker_subset.columns:
            marker_vals = marker_subset[marker]

            # Skip markers with too many missing values
            valid_mask = marker_vals.notna()
            if valid_mask.sum() < 5:
                continue

            marker_corrs = {}
            marker_pvals = {}
            for latent in latent_cols:
                latent_vals = emb_subset[latent]

                # Use only cells with valid marker values
                valid_idx = valid_mask[valid_mask].index
                x = marker_vals.loc[valid_idx].values
                y = latent_vals.loc[valid_idx].values

                # Compute Pearson correlation with p-value
                if len(x) >= 5 and np.std(x) > 0 and np.std(y) > 0:
                    from scipy.stats import spearmanr

                    corr, pval = spearmanr(x, y)
                    marker_corrs[latent] = corr
                    marker_pvals[latent] = pval
                else:
                    marker_corrs[latent] = np.nan
                    marker_pvals[latent] = np.nan

            correlations[marker] = marker_corrs
            pvalues[marker] = marker_pvals

        result = pd.DataFrame(correlations).T
        result.index.name = "marker"

        # Add p-values as separate columns
        pval_df = pd.DataFrame(pvalues).T
        for col in pval_df.columns:
            result[f"{col}_pval"] = pval_df[col]

    except Exception as e:
        print(f"Error processing {data_type}: {e}")
        result = pd.DataFrame()

    # Cache results
    if use_cache:
        _save_to_cache(cache_key, result)

    return result


def get_top_correlated_markers(
    correlations_df,
    latent="L1",
    alpha=0.05,
    method="fdr_bh",
    n_top=None,
):
    """Get statistically significant correlated markers for a specific latent dimension.

    Uses multiple testing correction to identify markers with significant correlations.

    Args:
        correlations_df: Output from compute_marker_embedding_correlations (DataFrame with
                         correlation values and p-values)
        latent: Latent dimension to analyze (default: "L1")
        alpha: Significance level after correction (default: 0.05)
        method: Multiple testing correction method (default: "fdr_bh" for Benjamini-Hochberg).
                Options: "bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg",
                "hommel", "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky"
        n_top: If specified, return only top n markers by absolute correlation (after filtering
               for significance). If None, return all significant markers.

    Returns:
        DataFrame with columns: marker, correlation, pvalue, pvalue_corrected, significant
        Sorted by absolute correlation value (descending)
    """
    from statsmodels.stats.multitest import multipletests

    if correlations_df.empty:
        return pd.DataFrame()

    pval_col = f"{latent}_pval"

    # Check if p-value column exists
    if pval_col not in correlations_df.columns:
        raise ValueError(
            f"P-value column '{pval_col}' not found. "
            f"Re-run compute_marker_embedding_correlations to compute p-values."
        )

    # Extract correlation and p-value columns
    df = correlations_df[[latent, pval_col]].copy()
    df["marker"] = df.index
    df = df.dropna(subset=[latent, pval_col])

    if len(df) == 0:
        return pd.DataFrame()

    # Apply multiple testing correction
    rejected, pvals_corrected, _, _ = multipletests(
        df[pval_col].values, alpha=alpha, method=method
    )

    df["pvalue"] = df[pval_col]
    df["pvalue_corrected"] = pvals_corrected
    df["significant"] = rejected
    df["correlation"] = df[latent]
    df["abs_corr"] = df[latent].abs()

    # Select columns and rename
    result = df[
        [
            "marker",
            "correlation",
            "pvalue",
            "pvalue_corrected",
            "significant",
            "abs_corr",
        ]
    ].copy()

    # Filter to significant markers
    result_sig = result[result["significant"]].copy()

    # Sort by absolute correlation
    result_sig = result_sig.sort_values("abs_corr", ascending=False)

    # Optionally limit to top n
    if n_top is not None and len(result_sig) > n_top:
        result_sig = result_sig.head(n_top)

    # Drop helper column and set index
    result_sig = result_sig.drop(columns=["abs_corr"])
    result_sig = result_sig.set_index("marker")

    return result_sig


def get_all_significant_correlations(
    correlations_df,
    alpha=0.05,
    method="fdr_bh",
):
    """Get all statistically significant marker-latent correlations.

    Args:
        correlations_df: Output from compute_marker_embedding_correlations
        alpha: Significance level after correction (default: 0.05)
        method: Multiple testing correction method (default: "fdr_bh")

    Returns:
        DataFrame with all significant correlations across all latent dimensions
    """
    # Find all latent columns (those without _pval suffix)
    latent_cols = [
        c for c in correlations_df.columns if not c.endswith("_pval")
    ]

    all_results = []
    for latent in latent_cols:
        sig_markers = get_top_correlated_markers(
            correlations_df, latent=latent, alpha=alpha, method=method
        )
        if not sig_markers.empty:
            sig_markers = sig_markers.reset_index()
            sig_markers["latent"] = latent
            all_results.append(sig_markers)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True).sort_values(
        ["latent", "pvalue_corrected"], ascending=[True, True]
    )


def get_highly_variable_genes(marker_data, n_top=5000):
    """Select top n highly variable genes based on variance.

    Args:
        marker_data: DataFrame with genes as columns and samples as rows
        n_top: Number of top variable genes to return (default: 5000)

    Returns:
        List of top n highly variable gene names
    """
    # Compute variance for each gene (appropriate for log-fold changes)
    variances = marker_data.var()

    # Get top n by variance
    top_genes = variances.nlargest(min(n_top, len(variances))).index.tolist()

    return top_genes


def compute_marker_parameter_correlations(
    model,
    context,
    parameters,
    data_type="transcriptomics",
    figure="figure3",
    data="dream_cytof",
    alpha=0.05,
    mt_method="fdr_bh",
    method="spearman",
    n_hvg=None,
    min_samples_per_group=5,
):
    """Compute correlations or differential expression between markers and model parameters or parameter groups.

    Args:
        model: Model name
        context: Context name
        parameters: List of parameter names or parameter group keys (e.g., ["D1", "D2"]) to correlate with markers.
                   Can mix individual parameters and parameter groups.
        data_type: Type of marker data - "proteomics" or "transcriptomics"
        figure: Figure name for loading data (default: "figure3")
        data: Dataset name (default: "dream_cytof")
        alpha: Significance level for multiple testing correction (default: 0.05)
        mt_method: Multiple testing correction method (default: "fdr_bh")
        method: Analysis method - "spearman" or "pearson" for correlation analysis,
               or "differential_expression" for differential expression (split at median) (default: "spearman")
        n_hvg: Number of highly variable genes to use (default: 5000).
               Set to None to use all genes.
        min_samples_per_group: Minimum samples per group for differential expression (default: 10)

    Returns:
        DataFrame with results:
        - If method is "spearman" or "pearson": correlations between markers and parameters with p-values
        - If method is "differential_expression": log2 fold change, t-test p-values, and group means

    Examples:
        # Spearman correlation analysis with parameter groups
        corr_df = compute_marker_parameter_correlations(
            model="p38", context="cytof_init", parameters=["D1", "D2", "D3"],
            data_type="transcriptomics", method="spearman"
        )

        # Pearson correlation analysis
        corr_df = compute_marker_parameter_correlations(
            model="p38", context="cytof_init", parameters=["D1"],
            data_type="transcriptomics", method="pearson"
        )

        # Differential expression analysis (split at median)
        de_df = compute_marker_parameter_correlations(
            model="p38", context="cytof_init", parameters=["D1", "D2"],
            data_type="transcriptomics", method="differential_expression"
        )
    """
    from scipy.stats import pearsonr, spearmanr
    from statsmodels.stats.multitest import multipletests

    # Load marker data
    marker_data = load_all_marker_data(data_type)
    n_total_genes = len(marker_data.columns)

    # Subset to highly variable genes if requested
    if n_hvg is not None and n_total_genes > n_hvg:
        hvg = get_highly_variable_genes(marker_data, n_top=n_hvg)
        marker_data = marker_data[hvg]
        print(
            f"Subset to {len(hvg)} highly variable genes (from {n_total_genes})"
        )

    # Load embedding data (needed for parameter groups)
    embedding_df = load_embedding_data(figure, data)
    subtypes_df = load_marcotte_subtypes(embedding_df.cell_line.unique())
    pca_embedding_df = prepare_pca_embeddings(embedding_df, subtypes_df)

    # Filter to specific model and context
    emb_df = pca_embedding_df[
        (pca_embedding_df["model"] == model)
        & (pca_embedding_df["context"] == context)
    ].copy()

    # Load parameter data for all parameters (individual or groups)
    param_data = {}
    for param in parameters:
        # Check if this is a parameter group
        if param in sensitive_dirs:
            print(
                f"Processing parameter group '{param}': {sensitive_dirs[param]}"
            )
            try:
                # Compute averaged gradient projection for the parameter group
                _, projection_values = compute_averaged_parameter_gradient(
                    model=model,
                    context=context,
                    parameter_group=sensitive_dirs[param],
                    embedding_df=emb_df,
                    figure=figure,
                    data=data,
                )
                param_data[param] = projection_values
            except Exception as e:
                print(
                    f"Error computing gradient for parameter group {param}: {e}"
                )
        else:
            # Load individual parameter
            try:
                param_values = load_parameter_data(
                    model, param, figure, data, context
                )
                param_data[param] = param_values
            except Exception as e:
                print(f"Error loading parameter {param}: {e}")

    if not param_data:
        raise ValueError("No parameters could be loaded")

    param_df = pd.DataFrame(param_data)

    # Find common cell lines
    common_cells = list(set(marker_data.index) & set(param_df.index))

    if len(common_cells) < 5:
        raise ValueError(f"Not enough common cell lines: {len(common_cells)}")

    # Subset to common cell lines
    marker_subset = marker_data.loc[common_cells]
    param_subset = param_df.loc[common_cells]

    # Compute results based on analysis method
    results = []

    if method in ("spearman", "pearson"):
        # Correlation analysis
        for marker in marker_subset.columns:
            marker_vals = marker_subset[marker]

            # Skip markers with too many missing values
            valid_mask = marker_vals.notna()
            if valid_mask.sum() < min_samples_per_group * 2:
                continue

            for param in param_subset.columns:
                param_vals = param_subset[param]

                # Get valid indices for both marker and parameter
                combined_valid = valid_mask & param_vals.notna()
                if combined_valid.sum() < min_samples_per_group * 2:
                    continue

                x = marker_vals.loc[combined_valid].values
                y = param_vals.loc[combined_valid].values

                # Compute correlation
                if np.std(x) > 0 and np.std(y) > 0:
                    if method == "spearman":
                        corr, pval = spearmanr(x, y)
                    elif method == "pearson":
                        corr, pval = pearsonr(x, y)
                    results.append(
                        {
                            "marker": marker,
                            "parameter": param,
                            "correlation": corr,
                            "pvalue": pval,
                            "n_samples": combined_valid.sum(),
                        }
                    )

    elif method == "differential_expression":
        # Differential expression analysis using linear model
        # marker ~ param_group (binary: high vs low based on median split)
        import statsmodels.api as sm

        for param in param_subset.columns:
            param_vals = param_subset[param]

            # Remove NaN values
            valid_cells = param_vals.notna()
            if valid_cells.sum() < 2 * min_samples_per_group:
                print(
                    f"Warning: Not enough valid samples for parameter {param}, skipping..."
                )
                continue

            param_valid = param_vals[valid_cells]

            # Split at median to create binary grouping
            median_val = param_valid.median()
            group_high = (param_valid > median_val).astype(int)

            n_low = (group_high == 0).sum()
            n_high = (group_high == 1).sum()

            # Check minimum samples per group
            if n_low < min_samples_per_group or n_high < min_samples_per_group:
                print(
                    f"Warning: Not enough samples per group for parameter {param} "
                    + f"(low={n_low}, high={n_high}), skipping..."
                )
                continue

            # Compute differential expression for each marker using linear model
            for marker in marker_subset.columns:
                marker_vals = marker_subset.loc[valid_cells, marker]

                # Get valid samples (non-NaN marker values)
                valid_marker = marker_vals.notna()
                if valid_marker.sum() < 2 * min_samples_per_group:
                    continue

                y = marker_vals[valid_marker].values
                X = sm.add_constant(group_high[valid_marker].values)

                try:
                    # Fit OLS model: marker ~ intercept + group_high
                    model = sm.OLS(y, X).fit()

                    # coefficient for group_high is the effect size (difference high - low)
                    # For log-transformed data, this is approximately log2 fold change
                    coef = model.params[1]
                    pval = model.pvalues[1]

                    # Compute group means for reference
                    mean_low = marker_vals[valid_marker][
                        group_high[valid_marker] == 0
                    ].mean()
                    mean_high = marker_vals[valid_marker][
                        group_high[valid_marker] == 1
                    ].mean()

                    results.append(
                        {
                            "marker": marker,
                            "parameter": param,
                            "log2_fold_change": coef,
                            "pvalue": pval,
                            "mean_low": mean_low,
                            "mean_high": mean_high,
                            "n_low": (group_high[valid_marker] == 0).sum(),
                            "n_high": (group_high[valid_marker] == 1).sum(),
                        }
                    )
                except Exception as e:
                    print(
                        f"Warning: Error computing differential expression for {marker} vs {param}: {e}"
                    )
                    continue
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'spearman', 'pearson', or 'differential_expression'."
        )

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Apply multiple testing correction
    for par in parameters:
        idx = result_df.parameter == par
        if idx.sum() > 0:
            rejected, pvals_corrected, _, _ = multipletests(
                result_df.loc[idx, "pvalue"].values,
                alpha=alpha,
                method=mt_method,
            )
            result_df.loc[idx, "pvalue_corrected"] = pvals_corrected
            result_df.loc[idx, "significant"] = rejected

    # Add sorting column based on method
    if method in ("spearman", "pearson"):
        result_df["abs_corr"] = result_df["correlation"].abs()
        result_df = result_df.sort_values("abs_corr", ascending=False)
    elif method == "differential_expression":
        result_df["abs_log2fc"] = result_df["log2_fold_change"].abs()
        result_df = result_df.sort_values("abs_log2fc", ascending=False)

    return result_df


def get_significant_marker_parameter_correlations(
    correlations_df,
    parameter=None,
    n_top=None,
):
    """Get significant marker-parameter correlations, optionally filtered by parameter.

    Args:
        correlations_df: Output from compute_marker_parameter_correlations
        parameter: If specified, filter to this parameter only
        alpha: Significance level (uses pvalue_corrected column)
        n_top: If specified, return only top n markers by absolute correlation

    Returns:
        DataFrame with significant correlations
    """
    if correlations_df.empty:
        return pd.DataFrame()

    df = correlations_df.copy()

    # Filter by parameter if specified
    if parameter is not None:
        df = df[df["parameter"] == parameter]

    # Filter to significant
    df = df[df["significant"]]

    # Sort by absolute correlation
    df = df.sort_values("abs_corr", ascending=False)

    # Optionally limit
    if n_top is not None and len(df) > n_top:
        df = df.head(n_top)

    return df.drop(columns=["abs_corr"])


def plot_marker_parameter_heatmap(
    correlations_df,
    parameters=None,
    n_top_markers=20,
    figsize=None,
    cmap="RdBu_r",
    save_path=None,
):
    """Plot a clustered heatmap of marker-parameter correlations or differential expression.

    Args:
        correlations_df: Output from compute_marker_parameter_correlations (either correlation or differential expression)
        parameters: List of parameters to include (default: all)
        n_top_markers: Number of top markers to show per parameter (default: 20)
        figsize: Figure size (auto-calculated if None)
        cmap: Colormap (default: "RdBu_r")
        save_path: Path to save the figure

    Returns:
        ClusterGrid object
    """
    if correlations_df.empty:
        raise ValueError("Empty correlations DataFrame")

    df = correlations_df.copy()

    # Detect if this is correlation or differential expression output
    is_differential_expression = "log2_fold_change" in df.columns

    if is_differential_expression:
        value_col = "log2_fold_change"
        sort_col = "abs_log2fc"
        cbar_label = "Log2 Fold Change"
        title = "Marker-Parameter Differential Expression (clustered)"
    else:
        value_col = "correlation"
        sort_col = "abs_corr"
        cbar_label = "Correlation"
        title = "Marker-Parameter Correlations (clustered)"

    # Filter parameters if specified
    if parameters is not None:
        df = df[df["parameter"].isin(parameters)]

    # Get top markers by absolute value (across all parameters)
    top_markers = (
        df.groupby("marker")[sort_col]
        .max()
        .nlargest(n_top_markers)
        .index.tolist()
    )

    # Filter to top markers
    df = df[df["marker"].isin(top_markers)]

    # Pivot to matrix
    pivot_df = df.pivot(index="marker", columns="parameter", values=value_col)

    # Fill NaN with 0 for clustering
    pivot_df = pivot_df.fillna(0)

    # Create figure size
    if figsize is None:
        figsize = (
            max(8, len(pivot_df.columns) * 1.5),
            max(6, len(pivot_df) * 0.4),
        )

    # Plot clustermap
    vmax = pivot_df.abs().max().max()
    g = sns.clustermap(
        pivot_df,
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        figsize=figsize,
        cbar_kws={"label": cbar_label},
        dendrogram_ratio=(0.1, 0.1),
        linewidths=0.5,
    )

    g.figure.suptitle(title, y=1.02)

    if save_path is not None:
        g.figure.savefig(save_path, bbox_inches="tight", dpi=150)

    return g


def plot_marker_parameter_volcano(
    correlations_df,
    parameter=None,
    alpha=0.05,
    effect_threshold=None,
    n_top_labels=10,
    figsize=(10, 8),
    save_path=None,
):
    """Plot a volcano plot of marker-parameter correlations or differential expression.

    Args:
        correlations_df: Output from compute_marker_parameter_correlations (either correlation or differential expression)
        parameter: Parameter to plot (required if multiple parameters in df)
        alpha: Significance threshold for adjusted p-values (default: 0.05)
        effect_threshold: Threshold for effect size (correlation or log2FC).
                         If None, uses 0.3 for correlation or 0.5 for log2FC
        n_top_labels: Number of top significant markers to label (default: 10)
        figsize: Figure size (default: (10, 8))
        save_path: Path to save the figure

    Returns:
        tuple: (fig, ax)
    """
    if correlations_df.empty:
        raise ValueError("Empty correlations DataFrame")

    df = correlations_df.copy()

    # Detect if this is correlation or differential expression output
    is_differential_expression = "log2_fold_change" in df.columns

    if is_differential_expression:
        effect_col = "log2_fold_change"
        x_label = "Log2 Fold Change"
        default_threshold = 0.5
    else:
        effect_col = "correlation"
        x_label = "Correlation"
        default_threshold = 0.3

    if effect_threshold is None:
        effect_threshold = default_threshold

    # Filter to specific parameter if multiple present
    parameters = df["parameter"].unique()
    if len(parameters) > 1:
        if parameter is None:
            raise ValueError(
                f"Multiple parameters in DataFrame. Specify one of: {list(parameters)}"
            )
        df = df[df["parameter"] == parameter]
    else:
        parameter = parameters[0]

    # Compute -log10(p-value)
    df["neg_log10_pval"] = -np.log10(df["pvalue_corrected"].clip(lower=1e-300))

    # Classify points
    df["significant"] = df["pvalue_corrected"] < alpha
    df["large_effect"] = df[effect_col].abs() > effect_threshold
    df["category"] = "Not significant"
    df.loc[df["significant"] & ~df["large_effect"], "category"] = "Significant"
    df.loc[
        df["significant"] & df["large_effect"] & (df[effect_col] > 0),
        "category",
    ] = "Significant (Up)"
    df.loc[
        df["significant"] & df["large_effect"] & (df[effect_col] < 0),
        "category",
    ] = "Significant (Down)"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors
    colors = {
        "Not significant": "lightgray",
        "Significant": "gray",
        "Significant (Up)": "firebrick",
        "Significant (Down)": "steelblue",
    }

    # Plot each category
    for category, color in colors.items():
        mask = df["category"] == category
        if mask.sum() > 0:
            ax.scatter(
                df.loc[mask, effect_col],
                df.loc[mask, "neg_log10_pval"],
                c=color,
                label=f"{category} (n={mask.sum()})",
                alpha=0.7,
                s=20,
                edgecolors="none",
            )

    # Add threshold lines
    ax.axhline(
        -np.log10(alpha), color="black", linestyle="--", linewidth=1, alpha=0.5
    )
    ax.axvline(
        effect_threshold, color="black", linestyle="--", linewidth=1, alpha=0.5
    )
    ax.axvline(
        -effect_threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )

    # Label top significant markers
    top_markers = df[df["significant"] & df["large_effect"]].nlargest(
        n_top_labels, "neg_log10_pval"
    )

    texts = []
    for _, row in top_markers.iterrows():
        texts.append(
            ax.text(
                row[effect_col],
                row["neg_log10_pval"],
                row["marker"],
                fontsize=8,
                ha="center",
                va="bottom",
            )
        )

    if texts:
        try:
            from adjustText import adjust_text

            adjust_text(
                texts,
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
                ax=ax,
            )
        except ImportError:
            pass  # adjustText not installed, labels stay in original position
        except Exception:
            pass  # If adjustText fails, labels stay in original position

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title(f"Volcano Plot: {parameter}")
    ax.legend(loc="upper right", frameon=False)

    # Style
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, ax


def plot_marker_parameter_volcano_grid(
    correlations_df,
    parameters=None,
    n_cols=3,
    alpha=0.05,
    effect_threshold=None,
    n_top_labels=5,
    figsize_per_plot=(5, 4),
    save_path=None,
):
    """Plot multiple volcano plots in a grid for different parameters.

    Args:
        correlations_df: Output from compute_marker_parameter_correlations (either correlation or differential expression)
        parameters: List of parameters to plot (default: all in df)
        n_cols: Number of columns in the grid (default: 3)
        alpha: Significance threshold for adjusted p-values (default: 0.05)
        effect_threshold: Threshold for effect size (correlation or log2FC).
                         If None, uses 0.3 for correlation or 0.5 for log2FC
        n_top_labels: Number of top significant markers to label per plot (default: 5)
        figsize_per_plot: Figure size per subplot (default: (5, 4))
        save_path: Path to save the figure

    Returns:
        tuple: (fig, axes)
    """
    if correlations_df.empty:
        raise ValueError("Empty correlations DataFrame")

    df = correlations_df.copy()

    # Detect if this is correlation or differential expression output
    is_differential_expression = "log2_fold_change" in df.columns

    if is_differential_expression:
        effect_col = "log2_fold_change"
        x_label = "Log2 Fold Change"
        default_threshold = 0.5
    else:
        effect_col = "correlation"
        x_label = "Correlation"
        default_threshold = 0.3

    if effect_threshold is None:
        effect_threshold = default_threshold

    # Get parameters to plot
    if parameters is None:
        parameters = df["parameter"].unique().tolist()

    n_params = len(parameters)
    n_rows = (n_params + n_cols - 1) // n_cols

    # Create figure
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Define colors
    colors = {
        "Not significant": "lightgray",
        "Significant": "gray",
        "Significant (Up)": "firebrick",
        "Significant (Down)": "steelblue",
    }

    for idx, param in enumerate(parameters):
        ax = axes[idx]
        param_df = df[df["parameter"] == param].copy()

        if param_df.empty:
            ax.set_visible(False)
            continue

        # Compute -log10(p-value)
        param_df["neg_log10_pval"] = -np.log10(
            param_df["pvalue_corrected"].clip(lower=1e-300)
        )

        # Classify points
        param_df["significant"] = param_df["pvalue_corrected"] < alpha
        param_df["large_effect"] = (
            param_df[effect_col].abs() > effect_threshold
        )
        param_df["category"] = "Not significant"
        param_df.loc[
            param_df["significant"] & ~param_df["large_effect"], "category"
        ] = "Significant"
        param_df.loc[
            param_df["significant"]
            & param_df["large_effect"]
            & (param_df[effect_col] > 0),
            "category",
        ] = "Significant (Up)"
        param_df.loc[
            param_df["significant"]
            & param_df["large_effect"]
            & (param_df[effect_col] < 0),
            "category",
        ] = "Significant (Down)"

        # Plot each category
        for category, color in colors.items():
            mask = param_df["category"] == category
            if mask.sum() > 0:
                ax.scatter(
                    param_df.loc[mask, effect_col],
                    param_df.loc[mask, "neg_log10_pval"],
                    c=color,
                    alpha=0.7,
                    s=15,
                    edgecolors="none",
                )

        # Add threshold lines
        ax.axhline(
            -np.log10(alpha),
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )
        ax.axvline(
            effect_threshold,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )
        ax.axvline(
            -effect_threshold,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

        # Label top significant markers
        top_markers = param_df[
            param_df["significant"] & param_df["large_effect"]
        ].nlargest(n_top_labels, "neg_log10_pval")

        for _, row in top_markers.iterrows():
            ax.annotate(
                row["marker"],
                (row[effect_col], row["neg_log10_pval"]),
                fontsize=6,
                ha="center",
                va="bottom",
                alpha=0.8,
            )

        # Title and labels
        ax.set_title(param, fontsize=10)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(x_label, fontsize=9)
        if idx % n_cols == 0:
            ax.set_ylabel("-log10(adj. p-value)", fontsize=9)

        # Style
        sns.despine(ax=ax)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    # Add legend to figure
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=8,
            label=cat,
        )
        for cat, color in colors.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=False,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, axes


def plot_marker_parameter_barplots(
    correlations_df,
    parameters=None,
    n_top=10,
    n_cols=3,
    alpha=0.05,
    figsize_per_plot=(4, 3),
    save_path=None,
):
    """Plot gridded barplots of top correlated/differentially expressed features per parameter.

    Args:
        correlations_df: Output from compute_marker_parameter_correlations (either correlation or differential expression)
        parameters: List of parameters to plot (default: all in df)
        n_top: Number of top features to show per parameter (default: 10)
        n_cols: Number of columns in the grid (default: 3)
        alpha: Significance threshold for highlighting (default: 0.05)
        figsize_per_plot: Figure size per subplot (default: (4, 3))
        save_path: Path to save the figure

    Returns:
        tuple: (fig, axes)

    Examples:
        # Correlation barplots
        corr_df = compute_marker_parameter_correlations(
            model="p38", context="cytof_init", parameters=["D1", "D2", "D3"],
            method="spearman"
        )
        fig, axes = plot_marker_parameter_barplots(corr_df)

        # Differential expression barplots
        de_df = compute_marker_parameter_correlations(
            model="p38", context="cytof_init", parameters=["D1", "D2", "D3"],
            method="differential_expression"
        )
        fig, axes = plot_marker_parameter_barplots(de_df)
    """
    if correlations_df.empty:
        raise ValueError("Empty correlations DataFrame")

    df = correlations_df.copy()

    # Detect if this is correlation or differential expression output
    is_differential_expression = "log2_fold_change" in df.columns

    if is_differential_expression:
        effect_col = "log2_fold_change"
        sort_col = "abs_log2fc"
        x_label = "Log2 Fold Change"
    else:
        effect_col = "correlation"
        sort_col = "abs_corr"
        x_label = "Correlation"

    # Get parameters to plot
    if parameters is None:
        parameters = df["parameter"].unique().tolist()

    n_params = len(parameters)
    n_rows = (n_params + n_cols - 1) // n_cols

    # Create figure
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True
    )
    axes = axes.flatten()

    for idx, param in enumerate(parameters):
        ax = axes[idx]
        param_df = df[df["parameter"] == param].copy()

        if param_df.empty:
            ax.set_visible(False)
            continue

        # Get top N features by absolute effect size
        top_df = param_df.nlargest(n_top, sort_col).copy()

        # Sort by effect size for plotting (most negative to most positive)
        top_df = top_df.sort_values(effect_col, ascending=True)

        # Determine bar colors based on significance and direction
        colors = []
        for _, row in top_df.iterrows():
            is_sig = row.get("pvalue_corrected", row["pvalue"]) < alpha
            effect = row[effect_col]
            if is_sig:
                colors.append("firebrick" if effect > 0 else "steelblue")
            else:
                colors.append("lightcoral" if effect > 0 else "lightsteelblue")

        # Create horizontal bar plot
        y_pos = np.arange(len(top_df))
        ax.barh(
            y_pos,
            top_df[effect_col],
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add marker names as y-tick labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_df["marker"], fontsize=8)
        if not is_differential_expression:
            ax.set_xlim(-1, 1)

        # Add vertical line at 0
        ax.axvline(0, color="black", linewidth=0.8)

        # Title and labels
        ax.set_title(param, fontsize=10, fontweight="bold")
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(x_label, fontsize=9)

        # Style
        sns.despine(ax=ax, left=True)
        ax.grid(True, axis="x", alpha=0.3)

    # Hide unused axes
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="firebrick",
            edgecolor="black",
            label="Significant (positive)",
        ),
        Patch(
            facecolor="steelblue",
            edgecolor="black",
            label="Significant (negative)",
        ),
        Patch(
            facecolor="lightcoral",
            edgecolor="black",
            label="Not significant (positive)",
        ),
        Patch(
            facecolor="lightsteelblue",
            edgecolor="black",
            label="Not significant (negative)",
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=False,
        fontsize=8,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, axes


def plot_marker_vs_parameter(
    model,
    context,
    marker,
    parameter,
    data_type="transcriptomics",
    figure="figure3",
    data="dream_cytof",
    figsize=(8, 6),
    save_path=None,
):
    """Plot a scatter plot of a marker vs a parameter.

    Args:
        model: Model name
        context: Context name
        marker: Marker name
        parameter: Parameter name
        data_type: Type of marker data - "proteomics" or "transcriptomics"
        figure: Figure name for loading data
        data: Dataset name
        figsize: Figure size tuple
        save_path: Path to save the figure

    Returns:
        tuple: (fig, ax)
    """
    from scipy.stats import pearsonr

    # Load marker data
    marker_data = load_all_marker_data(data_type, markers=[marker])
    marker_vals = marker_data[marker]

    # Load parameter data
    param_vals = load_parameter_data(model, parameter, figure, data, context)

    # Find common cell lines
    common_cells = list(set(marker_vals.index) & set(param_vals.index))

    # Get valid data
    valid_mask = (
        marker_vals.loc[common_cells].notna()
        & param_vals.loc[common_cells].notna()
    )
    valid_cells = [c for c in common_cells if valid_mask.get(c, False)]

    x = marker_vals.loc[valid_cells].values
    y = param_vals.loc[valid_cells].values

    # Compute correlation
    corr, pval = pearsonr(x, y)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, edgecolor="black", linewidths=0.5, s=50, alpha=0.7)

    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line,
        p(x_line),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"r = {corr:.3f}, p = {pval:.2e}",
    )

    ax.set_xlabel(f"{marker} ({data_type})")
    ax.set_ylabel(parameter)
    ax.set_title(
        f"{get_model_label(model)} - {get_context_label(context)}\nn={len(valid_cells)}"
    )
    ax.legend(loc="best", frameon=False)

    sns.despine()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_marker_vs_latent(
    model,
    context,
    marker,
    marker_data_type,
    latent="L1",
    figure="figure3",
    data="dream_cytof",
    figsize=(8, 6),
    show_regression=True,
    save_path=None,
):
    """Plot a marker value against a single latent embedding dimension.

    Args:
        model: Model name to plot
        context: Context to plot
        marker: Marker/gene name (e.g., "pEGFR", "EGFR")
        marker_data_type: Type of marker data - "cytof", "proteomics", or "transcriptomics"
        latent: Latent dimension to plot (default: "L1")
        figure: Figure name for loading data (default: "figure3")
        data: Dataset name (default: "dream_cytof")
        figsize: Figure size tuple (width, height)
        show_regression: Whether to show regression line (default: True)
        save_path: Path to save the figure (None = don't save)

    Returns:
        tuple: (fig, ax)
    """
    # Load and prepare embedding data
    embedding_df = load_embedding_data(figure, data)
    subtypes_df = load_marcotte_subtypes(embedding_df.cell_line.unique())
    pca_embedding_df = prepare_pca_embeddings(embedding_df, subtypes_df)

    # Filter to specific model and context
    plot_df = pca_embedding_df[
        (pca_embedding_df["model"] == model)
        & (pca_embedding_df["context"] == context)
    ].copy()

    # Load marker data
    marker_values = load_marker_data(marker, marker_data_type)
    plot_df[marker] = plot_df.index.map(marker_values)

    # Drop rows with missing marker values
    plot_df = plot_df.dropna(subset=[marker])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(
        plot_df[latent],
        plot_df[marker],
        edgecolor="black",
        linewidths=0.5,
        s=50,
        alpha=0.7,
    )

    # Add regression line if requested
    if show_regression and len(plot_df) >= 3:
        from sklearn.linear_model import LinearRegression

        X = plot_df[[latent]].values
        y = plot_df[marker].values
        reg = LinearRegression().fit(X, y)

        x_range = np.linspace(
            plot_df[latent].min(), plot_df[latent].max(), 100
        )
        y_pred = reg.predict(x_range.reshape(-1, 1))

        # Compute correlation
        corr = np.corrcoef(plot_df[latent].values, plot_df[marker].values)[
            0, 1
        ]

        ax.plot(
            x_range,
            y_pred,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"r = {corr:.3f}",
        )
        ax.legend(loc="best", frameon=False)

    # Labels
    ax.set_xlabel(f"{latent}")
    ax.set_ylabel(f"{marker} ({marker_data_type})")
    ax.set_title(f"{get_model_label(model)} - {get_context_label(context)}")

    # Style
    sns.despine()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def load_dynamic_cytof_data(
    observable: str, condition: str = "EGF", min_samples: int = 60
):
    """Load dynamic cytof data for a specific observable and condition.

    Args:
        observable: Observable/marker name (e.g., "p.EGFR", "p.ERK")
        condition: Condition suffix (default: "EGF")
        min_samples: Minimum number of samples required per timepoint (default: 60)

    Returns:
        DataFrame with cell_line, time, and measurement columns
    """
    data_dir = basedir / "data"
    df = pd.read_csv(data_dir / "cytof.csv", index_col=0)

    # Filter by observable and condition
    mask = (df["observableId"] == observable) & (
        df["simulationConditionId"].str.endswith(f"__{condition}")
    )
    filtered = df[mask].copy()

    # Extract cell line from preequilibrationConditionId
    filtered["cell_line"] = filtered["preequilibrationConditionId"]

    # Average across replicates (time_course A/B)
    result = (
        filtered.groupby(["cell_line", "time"])["measurement"]
        .mean()
        .reset_index()
    )

    # Filter out timepoints with less than min_samples
    timepoint_counts = result.groupby("time")["cell_line"].count()
    valid_timepoints = timepoint_counts[timepoint_counts >= min_samples].index
    result = result[result["time"].isin(valid_timepoints)]

    return result


def compute_averaged_parameter_gradient(
    model,
    context,
    parameter_group,
    embedding_df,
    figure="figure3",
    data="dream_cytof",
    min_valid_samples=5,
):
    """Compute averaged gradient across a group of parameters in embedding space.

    This function computes the gradient direction in embedding space for each parameter
    in the group using linear regression, then averages and normalizes these gradients.

    Args:
        model: Model name
        context: Context name
        parameter_group: Tuple/list of parameter names to average
        embedding_df: DataFrame with embeddings (must have columns starting with 'L')
        figure: Figure name for loading data (default: "figure3")
        data: Dataset name (default: "dream_cytof")
        min_valid_samples: Minimum number of valid samples required per parameter (default: 5)

    Returns:
        tuple: (avg_gradient, projection_values)
            - avg_gradient: numpy array with averaged normalized gradient
            - projection_values: pandas Series with projection of embeddings onto gradient
    """
    from sklearn.linear_model import LinearRegression

    # Get embedding columns (L1, L2, L3, ...)
    embedding_cols = [
        col for col in embedding_df.columns if col.startswith("L")
    ]

    # Collect gradients for each parameter in the group
    gradients = []
    for param in parameter_group:
        # Load parameter data
        param_values = load_parameter_data(model, param, figure, data, context)

        # Get common cell lines between embeddings and parameter
        common_cells = embedding_df.index.intersection(param_values.index)
        emb_subset = embedding_df.loc[common_cells]
        param_subset = param_values.loc[common_cells]

        # Remove NaN values
        valid_mask = param_subset.notna()
        if valid_mask.sum() < min_valid_samples:
            print(
                f"Warning: Too few valid samples for parameter {param} ({valid_mask.sum()}), skipping..."
            )
            continue

        # Prepare data for regression
        X = emb_subset.loc[valid_mask, embedding_cols].values
        y = param_subset.loc[valid_mask].values

        # Fit linear regression to get gradient
        reg = LinearRegression().fit(X, y)
        gradients.append(reg.coef_)

    if not gradients:
        raise ValueError(
            f"No valid gradients computed for parameter group with {len(parameter_group)} parameters"
        )

    # Average gradients across the parameter group
    avg_gradient = np.mean(gradients, axis=0)

    # Normalize gradient to unit length
    avg_gradient = avg_gradient / np.linalg.norm(avg_gradient)

    # Project embeddings onto averaged gradient direction
    embeddings_array = embedding_df[embedding_cols].values
    projection_values = embeddings_array @ avg_gradient

    # Convert to Series with cell line index
    projection_values = pd.Series(projection_values, index=embedding_df.index)

    return avg_gradient, projection_values


def plot_binned_dynamic_cytof(
    model,
    context,
    observables,
    conditions="EGF",
    bin_by="L1",
    bin_by_type="embedding",
    n_bins=3,
    bin_labels=None,
    figure="figure3",
    data="dream_cytof",
    figsize=None,
    cmap="coolwarm",
    save_path=None,
):
    """Plot averaged dynamic cytof data binned by marker or embedding values.

    Args:
        model: Model name
        context: Context name
        observables: Cytof observable(s) to plot - single string or list (e.g., "p.EGFR" or ["p.EGFR", "p.ERK"])
        conditions: Experimental condition(s) - single string or list (default: "EGF")
        bin_by: Column name to bin by - either a latent dimension (e.g., "L1"),
                a marker name, a parameter name if bin_by_type is "parameter",
                or a parameter group key (e.g., "D1", "D2", "D3") if bin_by_type is "parameter_group"
        bin_by_type: Type of binning - "embedding" for latent dimensions,
                     "cytof"/"proteomics"/"transcriptomics" for markers,
                     "parameter" for model parameters, or "parameter_group" for parameter groups
        n_bins: Number of bins (default: 3 for low/mid/high)
        bin_labels: Custom labels for bins (default: ["Low", "Mid", "High"] for 3 bins)
        figure: Figure name for loading data (default: "figure3")
        data: Dataset name (default: "dream_cytof")
        figsize: Figure size tuple (width, height) - auto-calculated if None
        cmap: Colormap for bin colors (default: "coolwarm")
        save_path: Path to save the figure (None = don't save)

    Returns:
        tuple: (fig, axes, all_data)
    """
    # Normalize inputs to lists
    if isinstance(observables, str):
        observables = [observables]
    if isinstance(conditions, str):
        conditions = [conditions]

    # Load embedding data
    embedding_df = load_embedding_data(figure, data)
    subtypes_df = load_marcotte_subtypes(embedding_df.cell_line.unique())
    pca_embedding_df = prepare_pca_embeddings(embedding_df, subtypes_df)

    # Filter to specific model and context
    emb_df = pca_embedding_df[
        (pca_embedding_df["model"] == model)
        & (pca_embedding_df["context"] == context)
    ].copy()

    # Get binning values
    if bin_by_type == "embedding":
        bin_values = emb_df[bin_by]
    elif bin_by_type == "parameter":
        # Load parameter data for binning
        param_values = load_parameter_data(
            model, bin_by, figure, data, context
        )
        bin_values = emb_df.index.map(param_values)
    elif bin_by_type == "parameter_group":
        # Bin by parameter group: compute averaged gradients in embedding space
        if bin_by not in sensitive_dirs:
            raise ValueError(
                f"Parameter group '{bin_by}' not found in sensitive_dirs. Available: {list(sensitive_dirs.keys())}"
            )

        param_group = sensitive_dirs[bin_by]
        _, bin_values = compute_averaged_parameter_gradient(
            model=model,
            context=context,
            parameter_group=param_group,
            embedding_df=emb_df,
            figure=figure,
            data=data,
        )
    else:
        # Load marker data for binning (cytof, proteomics, transcriptomics)
        marker_values = load_marker_data(bin_by, bin_by_type)
        bin_values = emb_df.index.map(marker_values)

    # Create bins
    if bin_labels is None:
        if n_bins == 3:
            bin_labels = ["Low", "Mid", "High"]
        else:
            bin_labels = [f"Bin {i+1}" for i in range(n_bins)]

    # Bin the values using quantiles
    emb_df["bin"] = pd.qcut(
        bin_values, q=n_bins, labels=bin_labels, duplicates="drop"
    )

    # Prepare cell bins for merging
    cell_bins = emb_df[["bin"]].reset_index()
    cell_bins.columns = ["cell_line", "bin"]

    # Get colors from colormap
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.2, 0.8, n_bins))

    # Create facet grid
    n_rows = len(observables)
    n_cols = len(conditions)

    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    all_data = {}

    for i, observable in enumerate(observables):
        for j, condition in enumerate(conditions):
            ax = axes[i, j]

            # Load dynamic cytof data
            dynamic_data = load_dynamic_cytof_data(observable, condition)

            # Merge with bin assignments
            merged = dynamic_data.merge(cell_bins, on="cell_line", how="inner")

            # Group by bin and time, compute mean and std
            grouped = (
                merged.groupby(["bin", "time"])["measurement"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            all_data[(observable, condition)] = grouped

            # Plot each bin
            for k, bin_label in enumerate(bin_labels):
                bin_data = grouped[grouped["bin"] == bin_label]
                if len(bin_data) > 0:
                    ax.plot(
                        bin_data["time"],
                        bin_data["mean"],
                        marker="o",
                        color=colors[k],
                        label=f"{bin_label} (n={bin_data['count'].iloc[0]})"
                        if i == 0 and j == 0
                        else None,
                        linewidth=2,
                        markersize=4,
                    )
                    # Add error bands (standard error)
                    se = bin_data["std"] / np.sqrt(bin_data["count"])
                    ax.fill_between(
                        bin_data["time"],
                        bin_data["mean"] - se,
                        bin_data["mean"] + se,
                        color=colors[k],
                        alpha=0.2,
                    )

            # Labels
            if i == n_rows - 1:
                ax.set_xlabel("Time (min)")
            if j == 0:
                ax.set_ylabel(get_cytof_marker_label(observable))
            if i == 0:
                ax.set_title(condition)

            # Style
            sns.despine(ax=ax)
            ax.grid(True, alpha=0.3)

    # Add overall title
    bin_by_label = (
        bin_by if bin_by_type == "embedding" else f"{bin_by} ({bin_by_type})"
    )
    fig.suptitle(
        f"Dynamics binned by {bin_by_label}\n{get_model_label(model)} - {get_context_label(context)}",
        fontsize=12,
        y=1.02,
    )

    # Add legend
    handles = [
        plt.Line2D(
            [0], [0], color=colors[k], linewidth=2, marker="o", markersize=4
        )
        for k in range(n_bins)
    ]
    # Get sample counts from first plot
    first_grouped = list(all_data.values())[0]
    legend_labels = []
    for bin_label in bin_labels:
        bin_data = first_grouped[first_grouped["bin"] == bin_label]
        n = bin_data["count"].iloc[0] if len(bin_data) > 0 else 0
        legend_labels.append(f"{bin_label} (n={n})")

    fig.legend(
        handles,
        legend_labels,
        title=bin_by_label,
        loc="center right",
        bbox_to_anchor=(1.15, 0.5),
        frameon=False,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes, all_data


def plot_embeddings_all_parameters(
    model,
    context,
    figure="figure3",
    data="dream_cytof",
    parameters=None,
    n_cols=6,
    x_col="L1",
    y_col="L2",
    cmap="RdBu_r",
    figsize_per_plot=(3, 3),
    save_path=None,
):
    """Plot a facet grid of embedding scatterplots for all parameters.

    Args:
        model: Model name to plot
        context: Context to plot
        figure: Figure name for loading data (default: "figure3")
        data: Dataset name (default: "dream_cytof")
        parameters: List of parameters to plot (default: all available)
        n_cols: Number of columns in the grid (default: 6)
        x_col: Column to plot on x-axis (default: "L1")
        y_col: Column to plot on y-axis (default: "L2")
        cmap: Colormap for continuous coloring (default: "RdBu_r")
        figsize_per_plot: Figure size per subplot (default: (3, 3))
        save_path: Path to save the figure (None = don't save)

    Returns:
        tuple: (fig, axes)
    """
    # Get parameters
    if parameters is None:
        parameters = get_available_parameters(model, figure, data)

    n_params = len(parameters)
    n_rows = int(np.ceil(n_params / n_cols))

    # Load and prepare embedding data
    embedding_df = load_embedding_data(figure, data)
    subtypes_df = load_marcotte_subtypes(embedding_df.cell_line.unique())
    pca_embedding_df = prepare_pca_embeddings(embedding_df, subtypes_df)

    # Filter to specific model and context
    plot_df = pca_embedding_df[
        (pca_embedding_df["model"] == model)
        & (pca_embedding_df["context"] == context)
    ].copy()

    # Create figure
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Compute axis limits once (centered at 0)
    x_max = max(abs(plot_df[x_col].min()), abs(plot_df[x_col].max()))
    y_max = max(abs(plot_df[y_col].min()), abs(plot_df[y_col].max()))

    for idx, param in enumerate(parameters):
        ax = axes_flat[idx]

        # Load parameter data
        try:
            param_values = load_parameter_data(
                model, param, figure, data, context
            )
            plot_df[param] = plot_df.index.map(param_values)

            # Center colorscale on mean
            values = plot_df[param].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                max_dev = max(
                    abs(values.min() - mean_val), abs(values.max() - mean_val)
                )
                vmin = mean_val - max_dev
                vmax = mean_val + max_dev

                ax.scatter(
                    plot_df[x_col],
                    plot_df[y_col],
                    c=plot_df[param],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    edgecolor="black",
                    linewidths=0.3,
                    s=20,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error:\n{str(e)[:20]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )

        # Set title (shortened parameter name)
        param_short = param[:25] + "..." if len(param) > 25 else param
        ax.set_title(param_short, fontsize=8)

        # Set axis limits
        ax.set_xlim(-x_max * 1.1, x_max * 1.1)
        ax.set_ylim(-y_max * 1.1, y_max * 1.1)

        # Draw axes at origin
        ax.axhline(0, color="black", linewidth=0.5, zorder=0)
        ax.axvline(0, color="black", linewidth=0.5, zorder=0)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove spines
        sns.despine(ax=ax, left=True, bottom=True)

    # Hide unused axes
    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Add overall title
    fig.suptitle(
        f"Parameter deviations - {get_model_label(model)} - {get_context_label(context)}\n({x_col} vs {y_col})",
        fontsize=14,
        y=1.01,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, axes


def plot_parameter_histograms_by_latent(
    model,
    context,
    parameters=None,
    latents=None,
    n_bins=3,
    bin_labels=None,
    figure="figure3",
    data="dream_cytof",
    figsize_per_plot=(3, 2.5),
    cmap="coolwarm",
    save_path=None,
):
    """Plot KDE distributions of parameters binned by latent embedding values.

    Creates a facet grid where rows are parameters and columns are latent dimensions.
    Each subplot shows KDE curves of the parameter for Low/Mid/High bins of that latent.

    Args:
        model: Model name
        context: Context name
        parameters: List of parameters to plot (default: all available)
        latents: List of latent dimensions (default: ["L1", "L2", "L3", "L4"])
        n_bins: Number of bins (default: 3 for low/mid/high)
        bin_labels: Custom labels for bins (default: ["Low", "Mid", "High"] for 3 bins)
        figure: Figure name for loading data (default: "figure3")
        data: Dataset name (default: "dream_cytof")
        figsize_per_plot: Figure size per subplot (default: (3, 2.5))
        cmap: Colormap for bin colors (default: "coolwarm")
        save_path: Path to save the figure (None = don't save)

    Returns:
        tuple: (fig, axes)
    """
    # Get parameters
    if parameters is None:
        parameters = get_available_parameters(model, figure, data)

    if latents is None:
        latents = ["L1", "L2", "L3", "L4"]

    # Create bin labels
    if bin_labels is None:
        if n_bins == 3:
            bin_labels = ["Low", "Mid", "High"]
        else:
            bin_labels = [f"Bin {i+1}" for i in range(n_bins)]

    # Load and prepare embedding data
    embedding_df = load_embedding_data(figure, data)
    subtypes_df = load_marcotte_subtypes(embedding_df.cell_line.unique())
    pca_embedding_df = prepare_pca_embeddings(embedding_df, subtypes_df)

    # Filter to specific model and context
    emb_df = pca_embedding_df[
        (pca_embedding_df["model"] == model)
        & (pca_embedding_df["context"] == context)
    ].copy()

    # Load all parameter data
    for param in parameters:
        try:
            param_values = load_parameter_data(
                model, param, figure, data, context
            )
            emb_df[param] = emb_df.index.map(param_values)
        except Exception as e:
            print(f"Error loading parameter {param}: {e}")
            emb_df[param] = np.nan

    # Create figure
    n_rows = len(parameters)
    n_cols = len(latents)
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Get colors from colormap
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.2, 0.8, n_bins))

    for i, param in enumerate(parameters):
        for j, latent in enumerate(latents):
            ax = axes[i, j]

            # Bin by latent values
            try:
                latent_values = emb_df[latent]
                emb_df["bin"] = pd.qcut(
                    latent_values,
                    q=n_bins,
                    labels=bin_labels,
                    duplicates="drop",
                )

                # Plot KDE for each bin
                for k, bin_label in enumerate(bin_labels):
                    bin_data = emb_df[emb_df["bin"] == bin_label][
                        param
                    ].dropna()
                    if len(bin_data) > 2:  # Need at least 3 points for KDE
                        sns.kdeplot(
                            data=bin_data,
                            ax=ax,
                            color=colors[k],
                            label=bin_label if i == 0 and j == 0 else None,
                            fill=True,
                            alpha=0.3,
                            linewidth=1.5,
                        )

            except Exception:
                ax.text(
                    0.5,
                    0.5,
                    "Error",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            # Labels
            if i == n_rows - 1:
                ax.set_xlabel(
                    param[:15] + "..." if len(param) > 15 else param,
                    fontsize=8,
                )
            if j == 0:
                # Shorten parameter name for y-label
                param_short = param[:20] + "..." if len(param) > 20 else param
                ax.set_ylabel(param_short, fontsize=8)
            if i == 0:
                ax.set_title(latent)

            # Style
            sns.despine(ax=ax)
            ax.set_yticks([])

    # Add overall title
    fig.suptitle(
        f"Parameter distributions by latent bins\n{get_model_label(model)} - {get_context_label(context)}",
        fontsize=12,
        y=1.02,
    )

    # Add legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[k], alpha=0.3)
        for k in range(n_bins)
    ]
    fig.legend(
        handles,
        bin_labels,
        title="Latent Bin",
        loc="center right",
        bbox_to_anchor=(1.1, 0.5),
        frameon=False,
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, axes
