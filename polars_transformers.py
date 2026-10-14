"""
Comprehensive Suite of High-Speed Custom Estimators for Polars DataFrames.
Maintains exact statistical parity with the Pandas suite while maximizing Rust-engine speed.
"""
import polars as pl
import pandas as pd
import numpy as np

try:

    import polars as pl
    import pytimetk as tk

    def add_time_features_polars(df_pl, date_col='date', 
                                 periods=[1, 7, 14, 28, 90], max_order=2):
        """
        Extrae variables estacionales de tiempo burlando las restricciones de tipo de PyTimeTK,
        pero manteniendo la velocidad del motor de Polars para el cálculo y el Join.
        """
        # 1. Extraemos fechas únicas con Polars y convertimos a Pandas de forma ultra-ligera (~214 filas)
        df_dates_pd =     df_pl.select(date_col).unique().to_pandas()
        
        # 2. Augmentar transformada de Fourier usando el motor de Polars (engine="polars")
        # pero alimentando la estructura Pandas para pasar las validaciones internas
        df_dates_pd = tk.augment_fourier(
            data=df_dates_pd,
            date_column=date_col,
            periods=periods,
            max_order=max_order,
            engine="polars"  # <- Sigue usando Polars para el cálculo matemático rápido
        )
        
        # 3. Augmentar firma de tiempo (Timeseries Signature)
        df_dates_pd = tk.augment_timeseries_signature(
            data=df_dates_pd,
            date_column=date_col
        )
        
        # 4. Convertimos el dataframe resultante de vuelta a Polars (operación instantánea)
        df_dates_pl = pl.from_pandas(df_dates_pd)
        
        # 5. Join izquierdo optimizado en Polars (Aquí es donde Polars demuestra su poder en 13M de filas)
        return df_pl.join(df_dates_pl, on=date_col, how='left')


except ImportError:
    pass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, KBinsDiscretizer, OneHotEncoder,
    PowerTransformer, PolynomialFeatures
)
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction import FeatureHasher
from scipy.interpolate import interp1d


# =====================================================================
# SAFE DROP COLUMNS (DropColumnsPolars)
# =====================================================================
class DropColumnsPolars(BaseEstimator, TransformerMixin):
    """
    Elimina de forma segura una lista de variables.

    - Compatible con Pandas DataFrame, Polars DataFrame y LazyFrame.
    - No falla si alguna columna no existe.
    - Mantiene el tipo de entrada.
    - Compatible con sklearn Pipeline.
    """

    def __init__(self, var):
        if isinstance(var, str):
            self.var = [var]
        else:
            self.var = list(var)

    def fit(self, X, y=None):
        # Guardamos solamente las columnas que realmente existen en fit.
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X

        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.columns_to_drop_ = [
            col for col in self.var
            if col in X_pl.columns
        ]

        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)

        X_pl = pl.from_pandas(X) if is_pandas else X

        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        # Seguridad adicional: también verificamos en transform.
        cols_to_drop = [
            col for col in self.columns_to_drop_
            if col in X_pl.columns
        ]

        if cols_to_drop:
            X_out = X_pl.drop(cols_to_drop)
        else:
            X_out = X_pl

        return X_out.to_pandas() if is_pandas else X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(
                [c for c in self.var if c not in self.columns_to_drop_],
                dtype=object
            )

        return np.array(
            [c for c in input_features if c not in self.columns_to_drop_],
            dtype=object
        )

class DropConstantFeaturesPolars(BaseEstimator, TransformerMixin):
    """
    Versión Polars de DropConstantFeatures.
    Compatible con Polars DataFrame, LazyFrame y Pandas DataFrame.
    """
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        n_rows = len(X_pl)
        if n_rows == 0:
            self.features_to_drop_ = []
            self.kept_features_ = X_pl.columns
            return self

        # 1. Consulta vectorizada de n_unique en todas las columnas de un solo golpe
        unique_counts = X_pl.select([pl.col(c).n_unique().alias(c) for c in X_pl.columns]).to_dicts()[0]

        self.features_to_drop_ = []
        for col, count in unique_counts.items():
            if count <= 1:
                self.features_to_drop_.append(col)
            elif self.threshold < 1.0:
                # Comprobación de cuasi-constante
                max_count = X_pl.select(pl.col(col).value_counts(sort=True)).to_series()[0]["count"]
                if (max_count / n_rows) >= self.threshold:
                    self.features_to_drop_.append(col)

        self.kept_features_ = [c for c in X_pl.columns if c not in self.features_to_drop_]
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        cols_to_drop = [c for c in self.features_to_drop_ if c in X_pl.columns]
        X_out = X_pl.drop(cols_to_drop) if cols_to_drop else X_pl

        return X_out.to_pandas() if is_pandas else X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.kept_features_, dtype=object)
        return np.array([c for c in input_features if c not in self.features_to_drop_], dtype=object)

# =====================================================================
# 0. VARIABLE CLASSIFIER (POLARS)
# =====================================================================
def categorizar_todo_polars(df_pl: pl.DataFrame, target_col="target_bin", max_cat=15, n_dis=30, excluir=None):
    if isinstance(df_pl, pl.LazyFrame):
        df_pl = df_pl.collect()

    if excluir is None:
        excluir = [target_col, "ID", "id", "user_id", "date"]

    cat_vars_low, cat_vars_high, num_vars, discrete_vars, cont_vars = [], [], [], [], []

    for col, dtype in df_pl.schema.items():
        if col in excluir:
            continue

        # Use Utf8 for string-like columns in Polars
        if dtype in [pl.Utf8, pl.Categorical, pl.Boolean]:
            n_unicos = df_pl.select(pl.col(col).n_unique()).item()
            if n_unicos <= max_cat:
                cat_vars_low.append(col)
            else:
                cat_vars_high.append(col)
        elif dtype.is_numeric():
            num_vars.append(col)
            n_unicos = df_pl.select(pl.col(col).n_unique()).item()
            if n_unicos <= n_dis:
                discrete_vars.append(col)
            else:
                cont_vars.append(col)

    return cat_vars_low, cat_vars_high, num_vars, discrete_vars, cont_vars




# =====================================================================
# 1. POLYNOMIAL FEATURES WRAPPER
# =====================================================================
class SklearnPolynomialWrapperPolars(BaseEstimator, TransformerMixin):
    def __init__(self, variables, degree=2, interaction_only=True, include_bias=False):
        self.variables = [variables] if isinstance(variables, str) else list(variables)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.poly_ = PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only, include_bias=self.include_bias
        )
        # Extract to numpy for sklearn
        self.poly_.fit(X_pl.select(self.variables).fill_null(0.0).to_numpy())
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        poly_matrix = self.poly_.transform(X_pl.select(self.variables).fill_null(0.0).to_numpy())
        feature_names = [f"poly_{f}" for f in self.poly_.get_feature_names_out(self.variables)]

        # Filter out original variables returned by PolyFeatures to avoid duplicates
        valid_cols = [i for i, name in enumerate(feature_names) if name.replace("poly_", "") not in self.variables]
        filtered_names = [feature_names[i] for i in valid_cols]
        filtered_matrix = poly_matrix[:, valid_cols]

        df_poly = pl.DataFrame(filtered_matrix, schema=filtered_names)
        X_out = pl.concat([X_pl, df_poly], how="horizontal")
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 2. OPTIMIZED PERCENTILE RANKER
# =====================================================================
class PercentileRankerPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        self.var = [var] if isinstance(var, str) else list(var)

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()
        self.dist_ = {
            col: X_pl[col].drop_nulls().sort().to_numpy()
            for col in self.var if col in X_pl.columns
        }
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        for col in self.var:
            if col in self.dist_ and len(self.dist_[col]) > 0:
                ref = self.dist_[col]
                vals = X_pl[col].to_numpy()
                ranks = np.searchsorted(ref, vals, side="right") / len(ref)
                exprs.append(pl.Series(f"{col}_RANK", ranks))

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.var, dtype=object)
        out = list(input_features)
        for col in self.var:
            if col in self.dist_:
                out.append(f"{col}_RANK")
        return np.array(out, dtype=object)

# =====================================================================
# 3. DISCRETIZATION (BiniPolars)
# =====================================================================
class BiniPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var, nbins):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var
        self.nbins = nbins

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.bins_ = {}
        for col in self.var:
            if col in X_pl.columns:
                series = X_pl[col].drop_nulls()
                if len(series) > 0:
                    # np.linspace provides identical logic to sklearn's uniform strategy
                    self.bins_[col] = np.linspace(series.min(), series.max(), self.nbins + 1)[1:-1].tolist()
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        for col in self.var:
            if col in self.bins_:
                exprs.append(
                    pl.col(col).fill_null(0.0).cut(breaks=self.bins_[col]).to_physical().cast(pl.Int32).alias(f"{col}_bin")
                )

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.var, dtype=object)
        out = list(input_features)
        for col in self.var:
            if col in self.bins_:
                out.append(f"{col}_bin")
        return np.array(out, dtype=object)

# =====================================================================
# 4. ONE-HOT ENCODING (OnehPolars)
# =====================================================================
class OnehPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.dummies_schema_ = {}
        for col in self.var:
            if col in X_pl.columns:
                cats = X_pl[col].drop_nulls().unique().sort().to_list()
                if len(cats) > 1:
                    cats = cats[1:]  # Drop first
                self.dummies_schema_[col] = cats
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        cols_drop = []
        for col, cats in self.dummies_schema_.items():
            if col in X_pl.columns:
                cols_drop.append(col)
                for cat in cats:
                    exprs.append((pl.col(col) == cat).cast(pl.Int32).alias(f"{col}_{cat}"))

        X_out = X_pl.with_columns(exprs).drop(cols_drop) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 5. ADVANCED IMPUTATION (Flagnan3Polars)
# =====================================================================
class Flagnan3Polars(BaseEstimator, TransformerMixin):
    def __init__(self, var, fill_zero_vars=None):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var
        self.fill_zero_vars = fill_zero_vars if fill_zero_vars else []

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.params_ = {}
        n_rows = len(X_pl)
        for col in self.var:
            if col not in X_pl.columns:
                continue
            null_count = X_pl.select(pl.col(col).null_count()).item()
            if null_count == 0:
                continue
            if null_count == n_rows:
                self.params_[col] = ("allnull", None)
                continue

            if X_pl[col].dtype.is_numeric():
                if col in self.fill_zero_vars:
                    self.params_[col] = ("value", 0)
                else:
                    skew_val = X_pl.select(pl.col(col).skew()).item() or 0.0
                    val = X_pl.select(pl.col(col).median()).item() if abs(skew_val) > 0.75 else X_pl.select(pl.col(col).mean()).item()
                    self.params_[col] = ("value", val)
            else:
                mode_series = X_pl.select(pl.col(col).mode()).to_series()
                mode_list = mode_series.to_list() if mode_series is not None else []
                self.params_[col] = ("value", mode_list[0] if len(mode_list) > 0 else "Missing")
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs, cols_drop = [], []
        for col, (st, val) in self.params_.items():
            if col in X_pl.columns:
                if st == "allnull":
                    cols_drop.append(col)
                else:
                    exprs.append(pl.col(col).is_null().cast(pl.Int32).alias(f"{col}_NA"))
                    exprs.append(pl.col(col).fill_null(val).alias(col))

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        if cols_drop:
            X_out = X_out.drop(cols_drop)
        return X_out.to_pandas() if is_pandas else X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.var, dtype=object)
        out = list(input_features)
        for col in self.var:
            if col in self.params_:
                if self.params_[col][0] == "allnull":
                    if col in out:
                        out.remove(col)
                else:
                    out.append(f"{col}_NA")
        return np.array(out, dtype=object)

# =====================================================================
# 6. OUTLIERS WITH CLIPPING (Flagout2Polars)
# =====================================================================
class Flagout2Polars(BaseEstimator, TransformerMixin):
    def __init__(self, var, distance=3.0):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var
        self.distance = distance

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.limits_ = {}
        for col in self.var:
            if col in X_pl.columns:
                mu = X_pl.select(pl.col(col).mean()).item()
                sigma = X_pl.select(pl.col(col).std()).item()
                if mu is not None and sigma is not None:
                    self.limits_[col] = (mu - self.distance * sigma, mu + self.distance * sigma)
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        for col, (lb, ub) in self.limits_.items():
            if col in X_pl.columns:
                col_float = pl.col(col).cast(pl.Float64)
                exprs.append(((col_float < lb) | (col_float > ub)).cast(pl.Int32).alias(f"{col}_out"))
                exprs.append(col_float.clip(lower_bound=lb, upper_bound=ub).alias(col))

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 7. SIMPLE IMPUTATION (FlagnanPolars)
# =====================================================================
class FlagnanPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.params_ = {}
        for col in self.var:
            if col not in X_pl.columns:
                continue
            null_count = X_pl.select(pl.col(col).null_count()).item()
            if null_count > 0:
                if X_pl[col].dtype.is_numeric():
                    self.params_[col] = X_pl.select(pl.col(col).median()).item()
                else:
                    mode_series = X_pl.select(pl.col(col).mode()).to_series()
                    mode_list = mode_series.to_list() if mode_series is not None else []
                    self.params_[col] = mode_list[0] if len(mode_list) > 0 else "Missing"
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        for col, val in self.params_.items():
            if col in X_pl.columns:
                exprs.append(pl.col(col).is_null().cast(pl.Int32).alias(f"{col}_NA"))
                exprs.append(pl.col(col).fill_null(val).alias(col))

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 8. OUTLIERS WITHOUT CLIPPING (FlagoutPolars)
# =====================================================================
class FlagoutPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var, distance=3.0):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var
        self.distance = distance

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.limits_ = {}
        for col in self.var:
            if col in X_pl.columns:
                mu = X_pl.select(pl.col(col).mean()).item()
                sigma = X_pl.select(pl.col(col).std()).item()
                if mu is not None and sigma is not None:
                    lb, ub = mu - self.distance * sigma, mu + self.distance * sigma
                    if X_pl.filter((pl.col(col) < lb) | (pl.col(col) > ub)).height > 0:
                        self.limits_[col] = (lb, ub)
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        for col, (lb, ub) in self.limits_.items():
            if col in X_pl.columns:
                exprs.append(((pl.col(col) < lb) | (pl.col(col) > ub)).cast(pl.Int32).alias(f"{col}_out"))

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 9. ZERO INFLATED TRANSFORMER (ZeroInflatedTransformerPolars)
# =====================================================================
class ZeroInflatedTransformerPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        self.var = [var] if isinstance(var, str) else list(var)

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.dists_ = {}
        for col in self.var:
            if col in X_pl.columns:
                non_zero = X_pl.filter(pl.col(col).is_not_null() & (pl.col(col) != 0))[col].sort().to_numpy()
                self.dists_[col] = non_zero
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        for col in self.var:
            if col not in X_pl.columns:
                continue

            exprs.append((pl.col(col) == 0).cast(pl.Int32).alias(f"{col}_is_zero"))

            ref = self.dists_.get(col, np.array([]))
            if len(ref) > 0:
                vals = X_pl[col].to_numpy()
                ranks = np.searchsorted(ref, vals, side="right") / len(ref)
                ranks = np.where(vals == 0, 0.0, ranks)
                exprs.append(pl.Series(f"{col}_RANK", ranks))
            else:
                exprs.append(pl.lit(0.0).alias(f"{col}_RANK"))

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.var, dtype=object)
        out = list(input_features)
        for col in self.var:
            out.append(f"{col}_is_zero")
            out.append(f"{col}_RANK")
        return np.array(out, dtype=object)

# =====================================================================
# 10. SEQUENTIAL TARGET ENCODING (CateEncodePolars)
# =====================================================================
class CateEncodePolars(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var

    def fit(self, X, y):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        y_pl = pl.Series("__target__", y).cast(pl.Float64)
        temp = X_pl.with_columns(y_pl)

        self.encoder_dict_ = {}
        for col in self.var:
            if col in X_pl.columns:
                ordered = temp.group_by(col).agg(pl.col("__target__").mean()).sort("__target__")[col].to_list()
                self.encoder_dict_[col] = dict(zip(ordered, range(len(ordered))))
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = [
            pl.col(col).replace(self.encoder_dict_[col]).cast(pl.Int64).alias(col)
            for col in self.var if col in self.encoder_dict_
        ]

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

class TargetEncoderFastPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        self.var = [var] if isinstance(var, str) else list(var)

    def fit(self, X, y):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame): X_pl = X_pl.collect()

        y_pl = pl.Series("__target__", y)
        temp = X_pl.with_columns(y_pl)
        self.global_mean_ = float(y_pl.mean())
        self.maps_ = {}
        smoothing = 10  # CORRECCIÓN: Smoothing añadido

        for col in self.var:
            if col in X_pl.columns:
                mapping = temp.group_by(col).agg([
                    pl.col('__target__').mean().alias('mean'),
                    pl.len().alias('count')
                ]).with_columns(
                    smoothed=((pl.col('count') * pl.col('mean')) + smoothing * self.global_mean_) / (pl.col('count') + smoothing)
                )
                self.maps_[col] = dict(zip(mapping[col].to_list(), mapping['smoothed'].to_list()))
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame): X_pl = X_pl.collect()

        exprs = [
            pl.col(col).replace(self.maps_[col], default=self.global_mean_).cast(pl.Float64).alias(f"{col}_TE")
            for col in self.var if col in self.maps_
        ]

        X_out = X_pl.with_columns(exprs).drop(self.var) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None: return np.array(self.var, dtype=object)
        out = [c for c in input_features if c not in self.var]
        out.extend([f"{col}_TE" for col in self.var])
        return np.array(out, dtype=object)
# =====================================================================
# 11. TEMPORAL DIFFERENCE (TemporalVariableTPolars)
# =====================================================================
class TemporalVariableTPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var, refvar):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var
        self.refvar = refvar

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        for col in self.var:
            if col in X_pl.columns and self.refvar in X_pl.columns:
                # If datetime dtype, get total days, else assume numerical difference
                if X_pl[col].dtype in [pl.Datetime, pl.Date]:
                    exprs.append(((pl.col(self.refvar) - pl.col(col)).dt.days()).alias(col))
                else:
                    exprs.append((pl.col(self.refvar) - pl.col(col)).alias(col))

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 12. MANUAL MAPPING (MapperPolars)
# =====================================================================
class MapperPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var, mapper):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var
        self.mapper = mapper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = [
            pl.col(col).replace(self.mapper).alias(col)
            for col in self.var if col in X_pl.columns
        ]

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 13. MEAN IMPUTATION (MeanImputerPolars)
# =====================================================================
class MeanImputerPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.imputer_dict_ = {col: X_pl.select(pl.col(col).mean()).item() for col in self.var if col in X_pl.columns}
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = [
            pl.col(col).fill_null(self.imputer_dict_[col]).alias(col)
            for col in self.var if col in self.imputer_dict_
        ]

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 14. QUANTILE NORMALIZATION (QnormPolars)
# =====================================================================
class QnormPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        # Needs numpy for scipy interpolation
        X_sub = X_pl.select(self.var).to_numpy()
        df_sorted = np.sort(X_sub, axis=0)
        reference_dist = df_sorted.mean(axis=1)
        ranks = np.arange(1, len(X_pl) + 1)
        self.finterp_ = interp1d(ranks, reference_dist, fill_value="extrapolate")
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        for col in self.var:
            if col in X_pl.columns:
                # Rank and map interpolation
                ranks = X_pl[col].rank().to_numpy()
                mapped = self.finterp_(ranks)
                exprs.append(pl.Series(col, mapped))

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 15. POWER TRANSFORM (PowertPolars)
# =====================================================================
class PowertPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var, method="yeo-johnson"):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var
        self.method = method

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.vari_ = []
        for col in self.var:
            if col in X_pl.columns:
                pt = PowerTransformer(method=self.method)
                pt.fit(X_pl.select(col).to_numpy())
                self.vari_.append((col, pt))
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        for col, pt in self.vari_:
            if col in X_pl.columns:
                transformed = pt.transform(X_pl.select(col).to_numpy()).flatten()
                exprs.append(pl.Series(col, transformed))

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 16. STANDARDIZATION (EscaleoPolars)
# =====================================================================
class EscaleoPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.params_ = {}
        for col in self.var:
            if col in X_pl.columns:
                mu = X_pl.select(pl.col(col).mean()).item()
                sigma = X_pl.select(pl.col(col).std()).item()
                self.params_[col] = (mu, sigma if sigma and sigma > 0 else 1.0)
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = [
            ((pl.col(c) - mu) / sigma).alias(c)
            for c, (mu, sigma) in self.params_.items() if c in X_pl.columns
        ]

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 17. RARE LABEL ENCODER (RareLabelPolars)
# =====================================================================
class RareLabelPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var, tol=0.02, rare_value="Rare"):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var
        self.tol = tol
        self.rare_value = rare_value

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.common_cats_ = {}
        total = len(X_pl)
        for col in self.var:
            if col in X_pl.columns:
                counts = X_pl[col].value_counts()
                common = counts.filter((pl.col("count") / total) >= self.tol)[col].to_list()
                self.common_cats_[col] = common
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = [
            pl.when(pl.col(c).is_in(com))
            .then(pl.col(c))
            .otherwise(pl.lit(self.rare_value))
            .alias(c)
            for c, com in self.common_cats_.items() if c in X_pl.columns
        ]

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

# =====================================================================
# 18. NLP FEATURE EXTRACTOR (NLPFeatureExtractorPolars)
# =====================================================================
class NLPFeatureExtractorPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var, method="tfidf", max_features=30, ngram_range=(1, 3), max_df=0.95):
        if not isinstance(var, list):
            raise ValueError("variables must be a list")
        self.var = var
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_df = max_df

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.vectorizers_ = {}
        for col in self.var:
            if col in X_pl.columns:
                text_data = X_pl[col].fill_null("").cast(pl.Utf8).to_list()
                if self.method == "tfidf":
                    vec = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range, max_df=self.max_df)
                else:
                    vec = HashingVectorizer(n_features=self.max_features, ngram_range=self.ngram_range, alternate_sign=False)
                vec.fit(text_data)
                self.vectorizers_[col] = vec
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        # Convert to pandas for vectorizer outputs (sklearn returns numpy / sparse)
        X_df = X_pl.to_pandas()

        feature_name_map = {}
        for col in self.var:
            if col not in self.vectorizers_ or col not in X_df.columns:
                continue

            vec = self.vectorizers_[col]
            text_data = X_df[col].fillna("").astype(str).tolist()
            transformed = vec.transform(text_data)

            # Convert to dense array if sparse
            if hasattr(transformed, "toarray"):
                arr = transformed.toarray()
            else:
                arr = np.asarray(transformed)

            # Determine feature names
            if hasattr(vec, "get_feature_names_out"):
                base_names = vec.get_feature_names_out()
                feat_names = [f"{col}__{fn}" for fn in base_names]
            else:
                feat_names = [f"{col}__hash_{i}" for i in range(arr.shape[1])]

            df_feats = pd.DataFrame(arr, columns=feat_names, index=X_df.index)

            # Drop original text column and append new features
            X_df = pd.concat([X_df.drop(columns=[col]), df_feats], axis=1)

            feature_name_map[col] = feat_names

        self._last_feature_names_ = [c for c in X_df.columns]
        if is_pandas:
            return X_df
        else:
            return pl.from_pandas(X_df)

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, "_last_feature_names_"):
            return np.array(self._last_feature_names_, dtype=object)
        if input_features is None:
            names = []
            for col, vec in getattr(self, "vectorizers_", {}).items():
                if hasattr(vec, "get_feature_names_out"):
                    names.extend([f"{col}__{fn}" for fn in vec.get_feature_names_out()])
                else:
                    names.extend([f"{col}__hash_{i}" for i in range(self.max_features)])
            return np.array(names, dtype=object)
        return np.array(input_features, dtype=object)


# =====================================================================
# HASHING ENCODER (HashingEncoderCustomPolars)
# =====================================================================
class HashingEncoderCustomPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var, n_features=8):
        self.var = [var] if isinstance(var, str) else list(var)
        self.n_features = n_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        hasher = FeatureHasher(n_features=self.n_features, input_type='string')
        for col in self.var:
            if col not in X_pl.columns:
                continue
            # Convert to list of tuples
            raw_tuples = [[(f"{col}_{val}", 1)] for val in X_pl[col].cast(pl.Utf8).fill_null("").to_list()]
            matrix = hasher.transform(raw_tuples).toarray()
            hash_cols = [f"{col}_hash_{i}" for i in range(self.n_features)]
            df_hash = pd.DataFrame(matrix, columns=hash_cols)
            X_pl = X_pl.with_columns([pl.Series(name, df_hash[name]) for name in hash_cols])
            X_pl = X_pl.drop(col)
        return X_pl.to_pandas() if is_pandas else X_pl

    def get_feature_names_out(self, input_features=None):
        features = self.var if input_features is None else list(input_features)
        out_features = []
        for f in features:
            if f in self.var:
                out_features.extend([f"{f}_hash_{i}" for i in range(self.n_features)])
            else:
                out_features.append(f)
        return np.array(out_features, dtype=object)

# =====================================================================
# ORDINAL ENCODER (OrdinalEncPolars)
# =====================================================================
class OrdinalEncPolars(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.maps_ = {}
        for col in self.var:
            if col in X_pl.columns:
                cats = X_pl[col].drop_nulls().unique().sort().to_list()
                self.maps_[col] = {cat: i for i, cat in enumerate(cats)}
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = [
            pl.col(col).replace(self.maps_[col]).cast(pl.Int64).alias(col)
            for col in self.var if col in self.maps_
        ]
        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out




# =====================================================================
# TARGET ENCODER FAST POLARS (IN-PLACE + TYPE-SAFE)
# =====================================================================
class TargetEncoderFastPolars2(BaseEstimator, TransformerMixin):
    def __init__(self, var, smoothing=10):
        self.var = [var] if isinstance(var, str) else list(var)
        self.smoothing = smoothing

    def fit(self, X, y):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame): 
            X_pl = X_pl.collect()

        # Asegurar que 'y' sea un Float64 Series
        if isinstance(y, (pd.Series, np.ndarray, list)):
            y_pl = pl.Series("__target__", y).cast(pl.Float64)
        elif isinstance(y, pl.Series):
            y_pl = y.alias("__target__").cast(pl.Float64)
        else:
            y_pl = pl.Series("__target__", y).cast(pl.Float64)

        temp = X_pl.with_columns(y_pl)
        self.global_mean_ = float(y_pl.mean())
        self.maps_ = {}

        for col in self.var:
            if col in X_pl.columns:
                mapping = (
                    temp.group_by(col)
                    .agg([
                        pl.col('__target__').mean().alias('mean'),
                        pl.len().alias('count')
                    ])
                    .with_columns(
                        smoothed=(
                            (pl.col('count') * pl.col('mean') + self.smoothing * self.global_mean_) 
                            / (pl.col('count') + self.smoothing)
                        )
                    )
                )
                # Guardamos como floats nativos de Python
                self.maps_[col] = dict(
                    zip(mapping[col].cast(pl.Utf8).to_list(), 
                        mapping['smoothed'].cast(pl.Float64).to_list())
                )
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame): 
            X_pl = X_pl.collect()

        exprs = []
        for col in self.var:
            if col in self.maps_:
                # Transforma string a float numérico y sobreescribe la columna
                exprs.append(
                    pl.col(col)
                    .cast(pl.Utf8)
                    .replace(self.maps_[col], default=self.global_mean_)
                    .cast(pl.Float64, strict=False)
                    .fill_null(self.global_mean_)
                    .alias(col)
                )

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features or self.var, dtype=object)


# =====================================================================
# BINI POLARS (IN-PLACE + BLINDADO CONTRA STRINGS)
# =====================================================================
class BiniPolars2(BaseEstimator, TransformerMixin):
    def __init__(self, var, nbins=5):
        self.var = [var] if isinstance(var, str) else list(var)
        self.nbins = nbins

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame): 
            X_pl = X_pl.collect()

        self.bins_ = {}
        for col in self.var:
            if col in X_pl.columns:
                # Forzar casteo a Float64 para evitar strings en np.linspace
                series = X_pl[col].cast(pl.Float64, strict=False).drop_nulls()
                if len(series) > 0:
                    min_val = float(series.min())
                    max_val = float(series.max())
                    if min_val < max_val:
                        self.bins_[col] = np.linspace(
                            min_val, max_val, self.nbins + 1
                        )[1:-1].tolist()
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame): 
            X_pl = X_pl.collect()

        exprs = []
        for col in self.var:
            if col in self.bins_:
                exprs.append(
                    pl.col(col)
                      .cast(pl.Float64, strict=False)
                      .fill_null(0.0)
                      .cut(breaks=self.bins_[col])
                      .to_physical()
                      .cast(pl.Int32)
                      .alias(col)
                )

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features or self.var, dtype=object)
# =====================================================================
# PERCENTILE RANKER POLARS (IN-PLACE)
# =====================================================================
class PercentileRankerPolars2(BaseEstimator, TransformerMixin):
    """
    Percentile Ranker in-place para Polars y Pandas.
    - Reemplaza la columna original con su rango percentil (0 a 1).
    - No genera columnas *_RANK.
    - Compatible con LazyFrame / DataFrame / Pandas.
    """
    def __init__(self, var):
        self.var = [var] if isinstance(var, str) else list(var)

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.dist_ = {}
        for col in self.var:
            if col in X_pl.columns:
                series = X_pl[col].cast(pl.Float64, strict=False).drop_nulls()
                if len(series) > 0:
                    self.dist_[col] = series.sort().to_numpy()
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        for col in self.var:
            if col in self.dist_ and len(self.dist_[col]) > 0:
                ref = self.dist_[col]
                vals = X_pl[col].cast(pl.Float64, strict=False).to_numpy()
                
                valid_mask = ~np.isnan(vals)
                ranks = np.full(len(vals), np.nan, dtype=np.float64)
                if valid_mask.any():
                    ranks[valid_mask] = np.searchsorted(ref, vals[valid_mask], side="right") / float(len(ref))

                exprs.append(pl.Series(col, ranks, dtype=pl.Float64))

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.var, dtype=object)
        return np.array(input_features, dtype=object)


# =====================================================================
# ZERO INFLATED TRANSFORMER POLARS (IN-PLACE)
# =====================================================================
class ZeroInflatedTransformerPolars2(BaseEstimator, TransformerMixin):
    """
    Transformación Zero-Inflated in-place para Polars y Pandas:
    - Ceros se mantienen como 0.0.
    - Valores no nulos != 0 se mapean a su percentil rank entre (0, 1].
    - Reemplaza la columna original sin crear *_is_zero ni *_RANK.
    """
    def __init__(self, var):
        self.var = [var] if isinstance(var, str) else list(var)

    def fit(self, X, y=None):
        X_pl = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        self.dists_ = {}
        for col in self.var:
            if col in X_pl.columns:
                non_zero = (
                    X_pl.filter(
                        (pl.col(col).is_not_null()) & 
                        (pl.col(col).cast(pl.Float64, strict=False) != 0.0)
                    )[col]
                    .cast(pl.Float64, strict=False)
                    .sort()
                    .to_numpy()
                )
                if len(non_zero) > 0:
                    self.dists_[col] = non_zero
        return self

    def transform(self, X):
        is_pandas = isinstance(X, pd.DataFrame)
        X_pl = pl.from_pandas(X) if is_pandas else X
        if isinstance(X_pl, pl.LazyFrame):
            X_pl = X_pl.collect()

        exprs = []
        for col in self.var:
            if col not in X_pl.columns:
                continue

            if col in self.dists_ and len(self.dists_[col]) > 0:
                ref = self.dists_[col]
                vals = X_pl[col].cast(pl.Float64, strict=False).to_numpy()
                
                out_vals = np.full(len(vals), np.nan, dtype=np.float64)
                
                # Máscara de ceros
                zero_mask = (vals == 0.0)
                out_vals[zero_mask] = 0.0
                
                # Máscara de no ceros válidos
                non_zero_mask = (~np.isnan(vals)) & (~zero_mask)
                if non_zero_mask.any():
                    out_vals[non_zero_mask] = (
                        np.searchsorted(ref, vals[non_zero_mask], side="right") / float(len(ref))
                    )

                exprs.append(pl.Series(col, out_vals, dtype=pl.Float64))
            else:
                exprs.append(
                    pl.when(pl.col(col).is_null())
                    .then(pl.lit(None, dtype=pl.Float64))
                    .otherwise(pl.lit(0.0, dtype=pl.Float64))
                    .alias(col)
                )

        X_out = X_pl.with_columns(exprs) if exprs else X_pl
        return X_out.to_pandas() if is_pandas else X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.var, dtype=object)
        return np.array(input_features, dtype=object)
