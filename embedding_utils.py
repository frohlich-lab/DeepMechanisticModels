import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


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

    for (context, samples, n_hidden), group_df in embeddings_df.groupby(
        ["context", "samples", "n_hidden"]
    ):
        les = group_df[
            ["cell_line", "job"] + [f"L{i}" for i in range(1, n_hidden + 1)]
        ].set_index("cell_line")
        les_pivot = les.set_index(["job"], append=True).unstack(["job"])
        les_pivot.columns = [f"{col[0]}_{col[1]}" for col in les_pivot.columns]

        # Transform to mean 0 and variance 1
        vals = les_pivot.values
        vals -= vals.mean()
        vals /= vals.std()

        # PCA transform
        pca = PCA(n_components=n_components)
        les_pca = pca.fit_transform(vals)
        explained_var = pca.explained_variance_ratio_.sum()
        explained_variance_ratios[(context, samples)] = explained_var

        results_dfs.append(
            pd.DataFrame(
                index=les_pivot.index,
                data=les_pca,
                columns=[f"L{i}" for i in range(1, n_components + 1)],
            ).assign(
                variance_explained=explained_var,
                context=context,
                samples=samples,
            )
        )

    whole_pca_le = pd.concat(results_dfs)
    return whole_pca_le, explained_variance_ratios


def _latent_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        col for col in df.columns if col.startswith("L") and col[1:].isdigit()
    ]
    return sorted(cols, key=lambda x: int(x[1:]))


def _make_classifier_pipeline(classifier: str):
    if classifier == "svm":
        estimator = SVC(kernel="linear", class_weight="balanced")
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    if classifier == "logreg":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    if classifier == "rf":
        estimator = RandomForestClassifier(
            n_estimators=200, random_state=0, class_weight="balanced"
        )
        return estimator
    raise ValueError("Classifier must be one of: 'svm', 'logreg', 'rf'.")


def _extract_feature_importances(
    fitted_pipeline, feature_cols: list[str], classifier: str
) -> pd.Series:
    """Extract feature importances from a fitted classifier pipeline.

    Returns a pd.Series indexed by feature column names with absolute
    importance values (normalised to sum to 1).
    """
    if classifier in ("svm", "logreg"):
        # Pipeline: scaler → model
        model = fitted_pipeline.named_steps["model"]
        coefs = np.abs(model.coef_)
        # For multiclass, average across one-vs-rest rows
        if coefs.ndim == 2:
            coefs = coefs.mean(axis=0)
        else:
            coefs = coefs.ravel()
    elif classifier == "rf":
        coefs = fitted_pipeline.feature_importances_
    else:
        return pd.Series(dtype=float)

    total = coefs.sum()
    if total > 0:
        coefs = coefs / total
    return pd.Series(coefs, index=feature_cols, name="importance")


def evaluate_luminal_basal_classifier(
    embeddings_df: pd.DataFrame,
    subtype_mapping: dict,
    label_key: str = "Luminal/Basal",
    group_cols: list[str] | None = None,
    n_splits: int = 5,
    classifier: str = "svm",
) -> pd.DataFrame:
    if embeddings_df.empty:
        return pd.DataFrame()

    if subtype_mapping and isinstance(
        next(iter(subtype_mapping.values())), dict
    ):
        subtype_lookup = {
            key: value.get(label_key) for key, value in subtype_mapping.items()
        }
    else:
        subtype_lookup = subtype_mapping

    df = embeddings_df.copy()
    df["subtype_lb"] = df["cell_line"].map(subtype_lookup)
    df = df[df["subtype_lb"].isin(["Luminal", "Basal"])].copy()
    if df.empty:
        return pd.DataFrame()

    feature_cols = _latent_feature_columns(df)
    if not feature_cols:
        raise ValueError(
            "No latent embedding columns found (expected L1, L2, ...)."
        )

    if group_cols is None:
        group_cols = [
            col
            for col in df.columns
            if col not in ["cell_line", "subtype_lb", *feature_cols]
        ]

    results = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        group = group.dropna(subset=feature_cols)
        if group.empty:
            continue

        labels, counts = np.unique(group["subtype_lb"], return_counts=True)
        if len(labels) < 2:
            splits = 0
        else:
            splits = min(n_splits, int(counts.min()))

        if splits < 2:
            accuracy = np.nan
            roc_auc = np.nan
        else:
            X = group[feature_cols].values
            y = LabelEncoder().fit_transform(group["subtype_lb"].values)
            cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)

            pipeline = _make_classifier_pipeline(classifier)
            accuracy = cross_val_score(
                pipeline, X, y, cv=cv, scoring="accuracy"
            ).mean()
            roc_auc = cross_val_score(
                pipeline, X, y, cv=cv, scoring="roc_auc"
            ).mean()

        group_info = (
            dict(zip(group_cols, group_key))
            if isinstance(group_key, tuple)
            else {group_cols[0]: group_key}
        )
        results.append(
            {
                **group_info,
                "classifier": classifier,
                "n_samples": int(len(group)),
                "n_splits": int(splits),
                "n_features": int(len(feature_cols)),
                "accuracy": accuracy,
                "roc_auc": roc_auc,
            }
        )

    return pd.DataFrame(results)


def evaluate_annotation_classifier(
    embeddings_df: pd.DataFrame,
    annotation_series: pd.Series,
    annotation_name: str,
    group_cols: list[str] | None = None,
    n_splits: int = 5,
    classifier: str = "svm",
    min_class_count: int = 3,
    return_feature_importances: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate how well embeddings separate an arbitrary categorical annotation.

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        PCA-transformed embeddings with cell_line column and L1, L2, … columns.
    annotation_series : pd.Series
        Series mapping cell_line (index) → category label. NaN / missing are dropped.
    annotation_name : str
        Human-readable name for the annotation (stored in the output).
    group_cols : list[str] | None
        Columns to group by (e.g. ["context", "samples"]).
    n_splits : int
        Number of cross-validation folds.
    classifier : str
        One of "svm", "logreg", "rf".
    min_class_count : int
        Minimum number of samples per class. Classes with fewer samples are dropped.
    return_feature_importances : bool
        If True, also return a DataFrame of feature importances per group.

    Returns
    -------
    pd.DataFrame (or tuple of two DataFrames if return_feature_importances=True)
        One row per group with columns: annotation, classifier, n_classes,
        n_samples, n_splits, accuracy, (macro_f1 for multiclass, roc_auc for binary).
        If return_feature_importances=True, second DataFrame has one row per
        (group, feature) with importance values.
    """
    if embeddings_df.empty:
        _empty = pd.DataFrame()
        return (_empty, _empty) if return_feature_importances else _empty

    df = embeddings_df.copy()

    # Map annotation onto cell_line
    if "cell_line" in df.columns:
        df["_annot"] = df["cell_line"].map(annotation_series)
    elif df.index.name == "cell_line":
        df["_annot"] = df.index.map(annotation_series)
    else:
        _empty = pd.DataFrame()
        return (_empty, _empty) if return_feature_importances else _empty

    df = df.dropna(subset=["_annot"])
    if df.empty:
        _empty = pd.DataFrame()
        return (_empty, _empty) if return_feature_importances else _empty

    feature_cols = _latent_feature_columns(df)
    if not feature_cols:
        _empty = pd.DataFrame()
        return (_empty, _empty) if return_feature_importances else _empty

    if group_cols is None:
        group_cols = [
            col
            for col in df.columns
            if col not in ["cell_line", "_annot", *feature_cols]
        ]

    results = []
    importance_records = []
    for group_key, group in df.groupby(group_cols, dropna=False):
        group = group.dropna(subset=feature_cols)
        if group.empty:
            continue

        # Filter classes by min_class_count
        counts = group["_annot"].value_counts()
        keep_classes = counts[counts >= min_class_count].index
        group = group[group["_annot"].isin(keep_classes)]

        labels, label_counts = np.unique(group["_annot"], return_counts=True)
        n_classes = len(labels)
        if n_classes < 2:
            continue

        splits = min(n_splits, int(label_counts.min()))
        extra_name = "roc_auc" if n_classes == 2 else "macro_f1"
        feat_imp = pd.Series(dtype=float)
        balanced_accuracy = np.nan
        if splits < 2:
            accuracy = np.nan
            extra_metric = np.nan
        else:
            X = group[feature_cols].values
            y = LabelEncoder().fit_transform(group["_annot"].values)
            cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)

            pipeline = _make_classifier_pipeline(classifier)
            accuracy = cross_val_score(
                pipeline, X, y, cv=cv, scoring="accuracy"
            ).mean()
            balanced_accuracy = cross_val_score(
                pipeline, X, y, cv=cv, scoring="balanced_accuracy"
            ).mean()

            # Use ROC AUC for binary, macro F1 for multiclass
            if n_classes == 2:
                extra_metric = cross_val_score(
                    pipeline, X, y, cv=cv, scoring="roc_auc"
                ).mean()
                extra_name = "roc_auc"
            else:
                extra_metric = cross_val_score(
                    pipeline, X, y, cv=cv, scoring="f1_macro"
                ).mean()
                extra_name = "macro_f1"

            # Fit on full data for feature importances
            if return_feature_importances:
                pipeline_full = _make_classifier_pipeline(classifier)
                pipeline_full.fit(X, y)
                feat_imp = _extract_feature_importances(
                    pipeline_full, feature_cols, classifier
                )

        # Chance levels
        majority_frac = float(label_counts.max()) / float(label_counts.sum())
        chance_balanced = 1.0 / n_classes

        group_info = (
            dict(zip(group_cols, group_key))
            if isinstance(group_key, tuple)
            else {group_cols[0]: group_key}
        )
        results.append(
            {
                **group_info,
                "annotation": annotation_name,
                "classifier": classifier,
                "n_classes": int(n_classes),
                "n_samples": int(len(group)),
                "n_splits": int(splits),
                "n_features": int(len(feature_cols)),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "chance_accuracy": majority_frac,
                "chance_balanced": chance_balanced,
                extra_name: extra_metric,
            }
        )

        # Record feature importances
        if return_feature_importances and not feat_imp.empty:
            for feat, imp in feat_imp.items():
                importance_records.append(
                    {
                        **group_info,
                        "annotation": annotation_name,
                        "classifier": classifier,
                        "feature": feat,
                        "importance": imp,
                    }
                )

    results_df = pd.DataFrame(results)
    if return_feature_importances:
        importance_df = pd.DataFrame(importance_records)
        return results_df, importance_df
    return results_df
