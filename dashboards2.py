"""
Diagnostic Dashboards, Fast EDA Reports, and Feature Engineering Value Audits.

Features:
- Single Source of Truth variable classification (classify_variables).
- Fast exploratory diagnostic reporting (fast_eda_report).
- Adaptive bounded sampling engine with automatic full-data bypass for N <= 10,000.
- Unified linear (Pearson), monotonic (Spearman), and non-linear (k-NN & Normalized MI [0, 1]) analytics.
- Feature Engineering ROI audit (audit_fe_information_gain).
- Interactive, bug-free Plotly correlation visualizations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score

import seaborn as sns
# =============================================================================
# 1. ADAPTIVE SAMPLING & STATISTICAL HELPER FUNCTIONS
# =============================================================================

def get_adaptive_sample_size(
    n_rows: int,
    task: str = "date",
    custom_size: Optional[Union[int, float]] = None,
    full_sample_threshold: int = 10000
) -> int:
    """
    Computes optimal sample size dynamically as a function of DataFrame length.

    Parameters:
    -----------
    n_rows : int
        Total number of rows in the dataset.
    task : str, default 'date'
        Task identifier for adaptive tuning bounds:
        - 'date': Fast parsing with regex optimization (bounds: 1,000 to 10,000).
        - 'plot': Representative sample for KDE and boxplots (bounds: 1,500 to 15,000).
        - 'mi'  : k-NN Mutual Information estimation (bounds: 2,500 to 25,000).
    custom_size : int, float, or None
        - If float in (0.0, 1.0]: Interpreted as an exact fraction of n_rows.
        - If int > 0: Explicit row count.
        - If None: Dynamic calculation based on dataset size and task.
    full_sample_threshold : int, default 10000
        Datasets with n_rows <= full_sample_threshold use 100% of rows (fraction = 1.0).

    Returns:
    --------
    int
        Final calculated row count.
    """
    # 1. If dataset is small or medium, use 100% of the data
    if n_rows <= full_sample_threshold and custom_size is None:
        return n_rows

    # 2. Handle explicit user overrides
    if custom_size is not None:
        if isinstance(custom_size, float) and 0.0 < custom_size <= 1.0:
            return max(100, min(int(n_rows * custom_size), n_rows))
        elif isinstance(custom_size, int) and custom_size > 0:
            return min(custom_size, n_rows)

    # 3. Adaptive configuration curves: (rate_fraction, min_bound, max_bound)
    task_configs: Dict[str, Tuple[float, int, int]] = {
        "date": (0.05, 1000, 10000),   # 5% of data, capped between 1k and 10k
        "plot": (0.10, 1500, 15000),   # 10% of data, capped between 1.5k and 15k
        "mi":   (0.20, 2500, 25000)    # 20% of data, capped between 2.5k and 25k
    }

    fraction, min_val, max_val = task_configs.get(task, (0.10, 1000, 10000))
    calculated = int(n_rows * fraction)
    return int(np.clip(calculated, min_val, min(max_val, n_rows)))


def adaptive_sample_df(
    df: pd.DataFrame,
    n_samples: int,
    stratify_col: Optional[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Extracts a representative sample combining boundaries (Head + Tail) with
    uniform random sampling to prevent temporal or ordering bias.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    n_samples : int
        Number of rows to extract.
    stratify_col : str, optional
        Target column for stratified sampling if categorical/discrete.
    random_state : int, default 42
        Reproducibility seed.

    Returns:
    --------
    pd.DataFrame
        Sampled DataFrame.
    """
    n_rows = len(df)
    if n_rows <= n_samples or n_samples <= 0:
        return df.copy()

    # Stratified sampling when target column is specified
    if stratify_col is not None and stratify_col in df.columns:
        try:
            from sklearn.model_selection import train_test_split
            sample_df, _ = train_test_split(
                df,
                train_size=n_samples,
                stratify=df[stratify_col],
                random_state=random_state
            )
            return sample_df.copy()
        except Exception:
            pass  # Fallback to multi-slice sampling if stratification fails

    # Multi-slice representation: Head (10%) + Tail (10%) + Random Body (80%)
    n_edge = max(20, int(n_samples * 0.10))
    n_random = n_samples - (2 * n_edge)

    if n_rows <= (2 * n_edge + n_random) or n_random <= 0:
        return df.sample(n_samples, random_state=random_state).copy()

    head_idx = df.index[:n_edge]
    tail_idx = df.index[-n_edge:]
    middle_pool = df.index[n_edge:-n_edge]

    np.random.seed(random_state)
    random_idx = np.random.choice(middle_pool, size=n_random, replace=False)
    combined_idx = head_idx.union(tail_idx).union(pd.Index(random_idx))

    return df.loc[combined_idx].copy()


def assign_feature_family(
    col: str,
    cat_low: Optional[List[str]] = None,
    cat_high: Optional[List[str]] = None,
    discrete: Optional[List[str]] = None,
    cont: Optional[List[str]] = None
) -> str:
    """
    Classifies engineered features into their source Feature Engineering families.
    """
    if col.endswith('_TE'): return 'High-Card (Target Encoded)'
    if col.endswith('_NA'): return 'Missing Indicator (_NA)'
    if col.endswith('_out'): return 'Outlier Flag (_out)'
    if col.endswith('_RANK'): return 'Percentile Rank (_RANK)'
    if col.endswith('_is_zero'): return 'Zero Indicator (_is_zero)'
    if col.endswith('_bin'): return 'Discretized Bin (_bin)'
    if col.startswith('poly_'): return 'Polynomial Interaction'
    if '_tfidf_' in col or '_hash_' in col: return 'NLP (TF-IDF/Hash)'
    if col.startswith('fourier_') or col.startswith('date_'): return 'Time Signature'

    if cat_low and any(col.startswith(v + '_') for v in cat_low): return 'Low-Card (One-Hot)'
    if cat_low and col in cat_low: return 'Low-Card (Base)'
    if cat_high and col in cat_high: return 'High-Card (Base)'
    if discrete and col in discrete: return 'Discrete (Base)'
    if cont and col in cont: return 'Continuous (Base)'
    return 'Other Transformations'


def extract_base_feature(col: str, known_categoricals: Optional[List[str]] = None) -> str:
    """
    Extracts the root base feature name from transformed or encoded variable strings.
    """
    if col.startswith('poly_'):
        return col.replace('poly_', '').split(' ')[0]

    for suffix in ['_TE', '_NA', '_out', '_RANK', '_is_zero', '_bin']:
        if col.endswith(suffix):
            return col[:-len(suffix)]

    if known_categoricals:
        for cat_col in known_categoricals:
            if col.startswith(f"{cat_col}_"):
                return cat_col

    return col


def quantize_series(s: pd.Series, max_bins: int = 50) -> pd.Series:
    """
    Quantizes a numeric or discrete series for robust discrete entropy calculation.
    """
    s_clean = pd.Series(s).dropna()
    if len(s_clean) == 0:
        return pd.Series(dtype=str)
    if s_clean.nunique() <= max_bins:
        return s_clean.astype(str)

    n_bins = min(max_bins, max(2, int(np.sqrt(len(s_clean)))))
    try:
        return pd.qcut(s_clean, q=n_bins, duplicates='drop').astype(str)
    except Exception:
        return pd.cut(s_clean, bins=n_bins, duplicates='drop').astype(str)


def compute_discrete_entropy(series: pd.Series) -> float:
    """
    Computes Shannon entropy (in nats) for a discrete series.
    """
    if len(series) == 0:
        return 1e-12
    p = series.value_counts(normalize=True).values
    return float(-np.sum(p * np.log(p + 1e-12)))


# =============================================================================
# 2. VARIABLE CLASSIFICATION & EXPLORATORY DIAGNOSTICS
# =============================================================================

def classify_variables(
    df: pd.DataFrame,
    target_col: str = "target",
    date_vars: Optional[List[str]] = None,
    n_dis: int = 30,
    max_cat_low: int = 15,
    rare_tol: float = 0.02,
    infer_dates: bool = True,
    skew_threshold: float = 3.0,
    cv_threshold: float = 3.0,
    zero_inflated_threshold: float = 0.30,
    sample_date_size: Optional[Union[int, float]] = None,
    full_sample_threshold: int = 10000
) -> Dict[str, Any]:
    """
    Single Source of Truth for variable classification with adaptive date inference.

    Categorizes columns into:
    - continuous, discrete, cat_low, cat_high, zero_inflated, extreme_skew, date.
    - Generates actionable data quality alerts.
    """
    n_rows = len(df)
    exclude_cols = [target_col, "ID", "id"] + (date_vars or [])
    analysis: Dict[str, Any] = {
        "cat_low": [], "cat_high": [], "categorical": [],
        "discrete": [], "continuous": [], "zero_inflated": [],
        "extreme_skew": [], "date": (date_vars or []).copy(),
        "alerts": [], "metadata": {}
    }

    df_temp = df.copy()

    # --- Adaptive Date Inference ---
    n_date_sample = n_rows
    if infer_dates:
        n_date_sample = get_adaptive_sample_size(
            n_rows, task="date", custom_size=sample_date_size, full_sample_threshold=full_sample_threshold
        )
        sample_df = adaptive_sample_df(df_temp, n_samples=n_date_sample)

        for col in df_temp.columns:
            if col in exclude_cols or pd.api.types.is_numeric_dtype(df_temp[col]):
                continue
            try:
                parsed = pd.to_datetime(sample_df[col], errors="coerce")
                if parsed.notna().mean() > 0.85:
                    analysis["date"].append(col)
                    analysis["alerts"].append(f"INFO: '{col}' inferred as date (tested on {n_date_sample:,} rows)")
            except Exception:
                pass

    # --- Feature Classification Loop ---
    for col in df_temp.columns:
        if col in exclude_cols or col in analysis["date"]:
            continue

        null_ratio = float(df_temp[col].isnull().mean())
        unique_count = int(df_temp[col].nunique(dropna=True))

        if null_ratio > 0.30:
            analysis["alerts"].append(f"⚠️ {col}: {null_ratio:.1%} missing values")
        if unique_count <= 1:
            analysis["alerts"].append(f"⚠️ {col}: zero variance (constant feature)")
            continue

        # Categorical variables
        if (
            df_temp[col].dtype == 'O'
            or isinstance(df_temp[col].dtype, pd.CategoricalDtype)
            or df_temp[col].dtype == 'bool'
        ):
            analysis["categorical"].append(col)
            if unique_count <= max_cat_low:
                analysis["cat_low"].append(col)
            else:
                analysis["cat_high"].append(col)
                analysis["alerts"].append(f"⚠️ {col}: high cardinality ({unique_count} distinct categories)")

            counts = df_temp[col].value_counts(normalize=True)
            if (counts < rare_tol).sum() > 0:
                analysis["alerts"].append(f"⚠️ {col}: has rare categories (<{rare_tol:.0%})")

        # Numeric variables
        elif pd.api.types.is_numeric_dtype(df_temp[col]):
            valid_data = df_temp[col].dropna()

            if len(valid_data) >= 20:
                zero_ratio = float((valid_data == 0).mean())
                skewness = float(stats.skew(valid_data))
                mean_val = float(valid_data.mean())
                cv_val = float(valid_data.std() / abs(mean_val)) if abs(mean_val) > 1e-8 else np.nan

                if zero_ratio >= zero_inflated_threshold:
                    analysis["zero_inflated"].append(col)
                    analysis["alerts"].append(f"⚠️ {col}: zero-inflated ({zero_ratio:.1%} zeros)")

                if abs(skewness) >= skew_threshold:
                    analysis["extreme_skew"].append(col)
                    analysis["alerts"].append(f"⚠️ {col}: extreme skew ({skewness:.1f})")

                if pd.notna(cv_val) and cv_val >= cv_threshold:
                    analysis["alerts"].append(f"⚠️ {col}: high coefficient of variation (CV = {cv_val:.1f})")

            if unique_count <= n_dis:
                analysis["discrete"].append(col)
            else:
                analysis["continuous"].append(col)

    analysis["metadata"] = {
        "total_cols": len(df_temp.columns),
        "nulls_total": int(df_temp.isnull().sum().sum()),
        "n_rows": n_rows,
        "date_inference_sample_size": n_date_sample if infer_dates else 0
    }
    return analysis


def analyze_numerical_distributions(
    df: pd.DataFrame,
    continuous_vars: List[str],
    skew_threshold: float = 0.75
) -> pd.DataFrame:
    """
    Computes statistical moments, coefficient of variation, normality tests,
    and transformation recommendations for continuous features.
    """
    rows = []
    for col in continuous_vars:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        if len(data) < 5:
            continue

        mean_val = float(data.mean())
        median_val = float(data.median())
        skew_val = float(stats.skew(data))
        cv_val = float(data.std() / abs(mean_val)) if abs(mean_val) > 1e-8 else np.nan

        is_normal = False
        if len(data) >= 20:
            try:
                _, p = stats.normaltest(data)
                is_normal = bool(p > 0.05)
            except Exception:
                pass

        suggest = bool(
            abs(skew_val) > skew_threshold
            or (median_val != 0 and abs(mean_val - median_val) / abs(median_val) > 0.10)
        )

        rows.append({
            "variable": col,
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "skew": round(skew_val, 3),
            "cv": round(cv_val, 3) if pd.notna(cv_val) else np.nan,
            "is_normal": is_normal,
            "suggest_transform": suggest
        })
    return pd.DataFrame(rows)


def fast_eda_report(
    df: pd.DataFrame,
    target: Optional[str] = None,
    date_vars: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 4),
    max_cat: int = 12,
    max_plot_samples: Optional[Union[int, float]] = None,
    full_sample_threshold: int = 10000
) -> Dict[str, Any]:
    """
    Renders a comprehensive EDA diagnostic report with adaptive sub-sampling
    for visualization to prevent memory leaks and UI lag.
    """
    print("=" * 75)
    print("⚡ FAST EDA DIAGNOSTIC REPORT")
    print("=" * 75)
    n_rows = len(df)
    print(f"Shape   : {n_rows:,} rows × {df.shape[1]} columns")
    print(f"Memory  : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    analysis = classify_variables(
        df, target_col=target if target else "target", date_vars=date_vars, full_sample_threshold=full_sample_threshold
    )

    print("\n" + "-" * 55)
    print("VARIABLE CLASSIFICATION SUMMARY")
    print("-" * 55)
    print(f"• Continuous : {len(analysis['continuous'])}")
    print(f"• Discrete   : {len(analysis['discrete'])}")
    print(f"• Categorical: {len(analysis['categorical'])} (Low-Card: {len(analysis['cat_low'])}, High-Card: {len(analysis['cat_high'])})")
    print(f"• Date       : {len(analysis['date'])}")

    if analysis["alerts"]:
        print("\nQUALITY ALERTS:")
        for alert in analysis["alerts"][:10]:
            print(f"  {alert}")
        if len(analysis["alerts"]) > 10:
            print(f"  ... and {len(analysis['alerts']) - 10} additional alerts.")

    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    print("\n" + "-" * 55)
    print("MISSING VALUES SUMMARY")
    print("-" * 55)
    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing.apply(lambda x: f"{x:.1%}").to_string())

    if analysis["continuous"]:
        num_report = analyze_numerical_distributions(df, analysis["continuous"])
        print("\n" + "-" * 55)
        print("NUMERICAL DISTRIBUTION INSIGHTS")
        print("-" * 55)
        print(num_report.to_string(index=False))

    # Adaptive plot subsampling
    plot_sample_n = get_adaptive_sample_size(
        n_rows, task="plot", custom_size=max_plot_samples, full_sample_threshold=full_sample_threshold
    )
    plot_df = adaptive_sample_df(df, n_samples=plot_sample_n)
    print(f"\n📊 Plotting with {plot_sample_n:,} rows ({plot_sample_n/n_rows:.1%} sample representation)...")

    # 1. Continuous feature distributions
    cont_cols = analysis["continuous"]
    if cont_cols:
        n = len(cont_cols)
        fig, axes = plt.subplots(n, 2, figsize=(figsize[0], 2.8 * n))
        if n == 1:
            axes = np.array([axes])
        for i, col in enumerate(cont_cols):
            sns.histplot(plot_df[col].dropna(), bins=30, kde=True, ax=axes[i, 0], color="steelblue")
            axes[i, 0].set_title(f"{col} - Distribution")
            sns.boxplot(x=plot_df[col].dropna(), ax=axes[i, 1], color="steelblue")
            axes[i, 1].set_title(f"{col} - Boxplot")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # 2. Discrete feature distributions
    disc_cols = analysis["discrete"]
    if disc_cols:
        n = len(disc_cols)
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], 2.5 * n))
        if n == 1:
            axes = [axes]
        for i, col in enumerate(disc_cols):
            order = plot_df[col].value_counts().sort_index().index
            sns.countplot(x=plot_df[col], order=order, ax=axes[i], palette="viridis")
            axes[i].set_title(f"{col} - Frequency")
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    # 3. Categorical feature distributions
    cat_cols = analysis["categorical"]
    if cat_cols:
        n = len(cat_cols)
        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], 2.5 * n))
        if n == 1:
            axes = [axes]
        for i, col in enumerate(cat_cols):
            order = plot_df[col].value_counts().index[:max_cat]
            sns.countplot(y=plot_df[col], order=order, ax=axes[i], palette="Blues_r")
            axes[i].set_title(f"{col} - Top Categories")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    print("\n" + "=" * 75)
    print("END OF FAST EDA REPORT")
    print("=" * 75)
    return analysis


# =============================================================================
# 3. FEATURE ENGINEERING VALUE & MUTUAL INFORMATION DIAGNOSTICS
# =============================================================================

def compute_feature_diagnostics(
    df_transformed: pd.DataFrame,
    y_train: pd.Series,
    task: str = "classification",
    target_name: str = "Target",
    cat_low: Optional[List[str]] = None,
    cat_high: Optional[List[str]] = None,
    discrete: Optional[List[str]] = None,
    cont: Optional[List[str]] = None,
    sample_size: Optional[Union[int, float]] = None,
    full_sample_threshold: int = 10000,
    random_state: int = 42,
    plot: bool = True
) -> pd.DataFrame:
    """
    Unified high-performance engine for Linear (Pearson), Monotonic (Spearman),
    Raw Non-Linear (k-NN), and Bounded Normalized Mutual Information ([0, 1]).

    Parameters:
    -----------
    df_transformed : pd.DataFrame
        Transformed / engineered feature matrix.
    y_train : pd.Series
        Target vector.
    task : str, default 'classification'
        'classification' or 'regression'.
    sample_size : int, float, or None
        Downsampling configuration for MI computation.
        If None, adapts automatically using full dataset for N <= full_sample_threshold.
    plot : bool, default True
        Renders the 3-panel feature diagnostic dashboard.

    Returns:
    --------
    pd.DataFrame
        Ranked metrics per feature with family attribution and normalized scores.
    """
    df_calc = df_transformed.copy()
    y_series = pd.Series(y_train, index=df_calc.index)
    n_rows = len(df_calc)
    cols = list(df_calc.columns)

    # 1. Full-dataset vectorized linear and monotonic correlations
    pearson_corrs = df_calc.corrwith(y_series.astype(float), method='pearson').to_dict()
    spearman_corrs = df_calc.corrwith(y_series.astype(float), method='spearman').to_dict()

    # 2. Adaptive sampling for k-NN Mutual Information
    mi_sample_n = get_adaptive_sample_size(
        n_rows, task="mi", custom_size=sample_size, full_sample_threshold=full_sample_threshold
    )

    if n_rows > mi_sample_n:
        df_mi_sample = adaptive_sample_df(
            df_calc, n_samples=mi_sample_n, stratify_col=target_name if target_name in df_calc.columns else None, random_state=random_state
        )
        y_mi_sample = y_series.loc[df_mi_sample.index]
    else:
        df_mi_sample = df_calc
        y_mi_sample = y_series

    # Fill NaNs with 0 in numeric matrix for k-NN estimators
    df_mi_numeric = df_mi_sample.select_dtypes(include=[np.number]).fillna(0.0)

    # 3. Continuous k-NN Mutual Information (Raw nats)
    if task == "classification":
        mi_raw = mutual_info_classif(
            df_mi_numeric, y_mi_sample.astype(int), random_state=random_state
        )
    else:
        mi_raw = mutual_info_regression(
            df_mi_numeric, y_mi_sample.astype(float), random_state=random_state
        )
    mi_raw_dict = dict(zip(df_mi_numeric.columns, mi_raw))

    # 4. Normalized Discrete Mutual Information (Bounded strictly in [0, 1])
    y_quant = quantize_series(y_mi_sample)
    H_y = compute_discrete_entropy(y_quant)

    mi_norm_dict: Dict[str, float] = {}
    for c in cols:
        if c not in df_mi_sample.columns:
            mi_norm_dict[c] = 0.0
            continue
        x_quant = quantize_series(df_mi_sample[c])
        common_idx = y_quant.index.intersection(x_quant.index)
        if len(common_idx) == 0:
            mi_norm_dict[c] = 0.0
            continue
        mi_val = mutual_info_score(x_quant.loc[common_idx], y_quant.loc[common_idx])
        H_x = compute_discrete_entropy(x_quant.loc[common_idx])
        denom = min(H_x, H_y)
        mi_norm_dict[c] = float(np.clip(mi_val / denom, 0.0, 1.0)) if denom > 1e-10 else 0.0

    familias = [assign_feature_family(c, cat_low, cat_high, discrete, cont) for c in cols]

    df_metrics = pd.DataFrame({
        'Feature': cols,
        'Pearson_Corr': [pearson_corrs.get(c, 0.0) for c in cols],
        'Spearman_Corr': [spearman_corrs.get(c, 0.0) for c in cols],
        'Mutual_Info_Raw': [mi_raw_dict.get(c, 0.0) for c in cols],
        'Mutual_Info_Norm': [mi_norm_dict.get(c, 0.0) for c in cols],
        'Familia_FE': familias
    })
    df_metrics['Abs_Pearson'] = df_metrics['Pearson_Corr'].abs()

    # 5. Three-Panel Visual Diagnostic Dashboard
    if plot:
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle(f"FEATURE DIAGNOSTIC DASHBOARD ({task.upper()}) | Target: '{target_name}'", fontsize=14, weight='bold')

        # Panel 1: Linear (|Pearson|) vs Raw Non-Linear (k-NN MI)
        sns.scatterplot(
            data=df_metrics, x='Abs_Pearson', y='Mutual_Info_Raw', hue='Familia_FE',
            palette='Set2', s=100, alpha=0.85, edgecolor='black', linewidth=0.5, ax=axes[0]
        )
        axes[0].set_title("Linear (|Pearson|) vs Raw Non-Linear MI (k-NN)", fontsize=11, weight='bold')
        axes[0].set_xlabel("|Pearson Correlation|")
        axes[0].set_ylabel("Mutual Information (Raw nats)")

        # Panel 2: Linear (|Pearson|) vs Normalized MI [0, 1]
        sns.scatterplot(
            data=df_metrics, x='Abs_Pearson', y='Mutual_Info_Norm', hue='Familia_FE',
            palette='Set2', s=100, alpha=0.85, edgecolor='black', linewidth=0.5, ax=axes[1], legend=False
        )
        axes[1].set_title("Linear (|Pearson|) vs Normalized NMI [0, 1]", fontsize=11, weight='bold')
        axes[1].set_xlabel("|Pearson Correlation|")
        axes[1].set_ylabel("Normalized MI (NMI)")
        axes[1].set_ylim(-0.02, 1.02)

        # Panel 3: Cumulative Information Gain by Feature Family
        df_contrib = (
            df_metrics.groupby('Familia_FE')['Mutual_Info_Norm']
            .sum().reset_index()
            .sort_values('Mutual_Info_Norm', ascending=False)
        )
        sns.barplot(
            data=df_contrib, x='Mutual_Info_Norm', y='Familia_FE',
            palette='Set2', ax=axes[2], edgecolor='black', linewidth=0.5
        )
        axes[2].set_title("Cumulative Information Gain by Feature Family", fontsize=11, weight='bold')
        axes[2].set_xlabel("Total Normalized MI")
        axes[2].set_ylabel("")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    return df_metrics


def get_classification_feature_dashboard(
    df_transformed: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any
) -> pd.DataFrame:
    """Wrapper for classification feature diagnostics."""
    return compute_feature_diagnostics(df_transformed, y_train, task="classification", **kwargs)


def get_regression_feature_dashboard(
    df_transformed: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any
) -> pd.DataFrame:
    """Wrapper for regression feature diagnostics."""
    return compute_feature_diagnostics(df_transformed, y_train, task="regression", **kwargs)


def audit_clas_norm(
    df_transformed: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any
) -> pd.DataFrame:
    """Alias for Normalized Classification Feature Diagnostics."""
    return compute_feature_diagnostics(df_transformed, y_train, task="classification", **kwargs)


def audit_regre_norm(
    df_transformed: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any
) -> pd.DataFrame:
    """Alias for Normalized Regression Feature Diagnostics."""
    return compute_feature_diagnostics(df_transformed, y_train, task="regression", **kwargs)


def audit_fe_information_gain(
    df_metrics: pd.DataFrame,
    known_categoricals: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Audits derived/engineered features against raw base features to highlight
    the most informative transformation per root variable.
    """
    df_audit = df_metrics.copy()
    df_audit['Base_Feature'] = df_audit['Feature'].apply(lambda c: extract_base_feature(c, known_categoricals))

    metric_col = 'Mutual_Info_Norm' if 'Mutual_Info_Norm' in df_audit.columns else 'Mutual_Info_Raw'

    audit_summary = df_audit.groupby('Base_Feature').agg(
        Max_MI=(metric_col, 'max'),
        Mean_MI=(metric_col, 'mean'),
        Best_Transformation=('Feature', lambda x: df_audit.loc[x.index].sort_values(metric_col, ascending=False)['Feature'].iloc[0]),
        Best_FE_Technique=('Familia_FE', lambda x: df_audit.loc[x.index].sort_values(metric_col, ascending=False)['Familia_FE'].iloc[0]),
        Derived_Features_Count=('Feature', 'count')
    ).reset_index().sort_values('Max_MI', ascending=False)

    print("\n" + "=" * 75)
    print("📊 FEATURE ENGINEERING ROI AUDIT (BEST TRANSFORMATION PER BASE VARIABLE)")
    print("=" * 75)
    return audit_summary


# =============================================================================
# 4. HIGH-PERFORMANCE INTERACTIVE PLOTLY VISUALIZATIONS
# =============================================================================

def plot_feature_correlations_plotly(
    xx: pd.DataFrame,
    y_train: pd.Series,
    target_name: str = "Target",
    clase: Optional[int] = None,
    top_n: int = 35
) -> None:
    """
    Vectorized Plotly scatter chart for linear feature-target correlations.
    Guarantees that all top_n categorical ticks on the y-axis are rendered without skips.
    """
    yy = y_train.copy()
    if clase is not None and (yy.nunique() > 2 or not pd.api.types.is_numeric_dtype(yy)):
        yy = (yy == clase).astype(int)
        class_label = f"{target_name}_class_{clase}"
    else:
        class_label = f"{target_name}"

    # Fast vectorized Pearson correlation
    corr_series = xx.corrwith(pd.Series(yy, index=xx.index).astype(float), method='pearson')
    ddcor = pd.DataFrame({'Feature': corr_series.index, 'Correlation': corr_series.values})
    ddcor['AbsCorrelation'] = ddcor['Correlation'].abs()
    ddcor = ddcor.dropna().sort_values(by='AbsCorrelation', ascending=False).head(top_n)

    if ddcor.empty:
        print("⚠️ No valid correlation scores found to plot.")
        return

    fig = px.scatter(
        ddcor, x='Correlation', y='Feature', color='Correlation',
        color_continuous_scale='RdBu_r', range_color=[-1, 1],
        title=f'Top {len(ddcor)} Features Most Correlated with "{class_label}"'
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis_title='Pearson Correlation',
        yaxis_title='Feature',
        xaxis=dict(range=[-1.05, 1.05], zeroline=True, zerolinewidth=1.5, zerolinecolor='Gray'),
        yaxis=dict(
            autorange='reversed',
            type='category',
            categoryorder='array',
            categoryarray=ddcor['Feature'].tolist(),
            dtick=1,             # Explicitly renders every category label
            tickmode='linear',   # Prevents automatic tick decimation
            automargin=True      # Adjusts left margin dynamically for long names
        ),
        height=max(500, len(ddcor) * 24),
        margin=dict(l=200, r=40, t=60, b=50)
    )
    fig.show()


def plot_classification_correlations_plotly(
    xx: pd.DataFrame,
    y_train: pd.Series,
    target_name: str,
    clase: int = 1,
    top_n: int = 35
) -> None:
    """Wrapper for classification correlation Plotly scatter chart."""
    plot_feature_correlations_plotly(xx, y_train, target_name=target_name, clase=clase, top_n=top_n)


def plot_classification_correlations_plotly_r(
    xx: pd.DataFrame,
    y_train: pd.Series,
    target_name: str,
    top_n: int = 35
) -> None:
    """Wrapper for regression correlation Plotly scatter chart."""
    plot_feature_correlations_plotly(xx, y_train, target_name=target_name, clase=None, top_n=top_n)
