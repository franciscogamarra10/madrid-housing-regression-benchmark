"""
Comprehensive Suite of Custom Scikit-Learn Estimators for Pandas DataFrames.
Includes Time Series extraction, NLP, Target Encoding, and robust imputations.
"""
import pandas as pd
import numpy as np
import polars as pl
from typing import List, Union,Optional
try:

	import polars as pl
	import pytimetk as tk

	def add_time_features_pandas(df_pl, date_col='date', 
							  periods=[1, 7, 14, 28, 90], max_order=2):
		
		# 1. Extraemos fechas únicas con Polars y convertimos a Pandas de forma ultra-ligera (~214 filas) 
		df_dates_pd =pd.DataFrame(pd.to_datetime(np.unique(df_pl[date_col])))
		df_dates_pd.columns=[date_col]
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
		#df_dates_pl = pl.from_pandas(df_dates_pd)

		# 5. Join izquierdo optimizado en Polars (Aquí es donde Polars demuestra su poder en 13M de filas)
		return df_pl.merge(df_dates_pd, on=date_col, how='left')


except ImportError:
    pass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, KBinsDiscretizer, OneHotEncoder,
    PowerTransformer, PolynomialFeatures
)
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction import FeatureHasher

# =====================================================================
# 0. VARIABLE CLASSIFIER (PANDAS)
# =====================================================================
def categorizar_todo_pandas(df: pd.DataFrame, target_col="target_bin", max_cat=15, n_dis=30, excluir=None):
    if excluir is None:
        excluir = [target_col, "ID", "id", "user_id", "date"]

    cat_vars_low, cat_vars_high, num_vars, discrete_vars, cont_vars = [], [], [], [], []

    for col in df.columns:
        if col in excluir:
            continue

        if df[col].dtype == 'O' or isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == 'bool':
            n_unicos = df[col].nunique(dropna=True)
            if n_unicos <= max_cat: cat_vars_low.append(col)
            else: cat_vars_high.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            num_vars.append(col)
            n_unicos = df[col].nunique(dropna=True)
            if n_unicos <= n_dis: discrete_vars.append(col)
            else: cont_vars.append(col)

    return cat_vars_low, cat_vars_high, num_vars, discrete_vars, cont_vars

# =====================================================================
# TIME SERIES FEATURES (PYTIMETK)
# =====================================================================
def add_time_features_polars(df_pl, date_col='date', periods=[1, 7, 14, 28, 90], max_order=2):
    """
    Extract seasonal time features bypassing PyTimeTK type restrictions,
    maintaining Polars engine speed for math and joins.
    """
    # 1. Extract unique dates ultra-fast via Polars and convert to Pandas
    df_dates_pd = df_pl.select(date_col).unique().to_pandas()

    # 2. Augment fourier using Polars engine internally via PyTimeTK
    df_dates_pd = tk.augment_fourier(
        data=df_dates_pd,
        date_column=date_col,
        periods=periods,
        max_order=max_order,
        engine="polars"
    )

    # 3. Augment timeseries signature
    df_dates_pd = tk.augment_timeseries_signature(
        data=df_dates_pd,
        date_column=date_col
    )

    # 4. Convert back to Polars
    df_dates_pl = pl.from_pandas(df_dates_pd)

    # 5. Optimized Left Join in Polars
    return df_pl.join(df_dates_pl, on=date_col, how='left')

def add_time_features_pandas(df_pd, date_col='date', periods=[1, 7, 14, 28, 90], max_order=2):
    """
    Extract seasonal time features entirely in Pandas.
    """
    # 1. Extract unique dates in Pandas
    df_dates_pd = df_pd[[date_col]].drop_duplicates().copy()

    # 2. Augment fourier
    df_dates_pd = tk.augment_fourier(
        data=df_dates_pd,
        date_column=date_col,
        periods=periods,
        max_order=max_order,
        engine="pandas"
    )

    # 3. Augment timeseries signature
    df_dates_pd = tk.augment_timeseries_signature(
        data=df_dates_pd,
        date_column=date_col
    )

    # 4. Left Join in Pandas
    return df_pd.merge(df_dates_pd, on=date_col, how='left')

# =====================================================================
# 1. SAFE COLUMN DROPPING (DropColumnsPandas)
# =====================================================================
class DropColumnsPandas(BaseEstimator, TransformerMixin):
    """
    Safely drops a specified list of columns from a Pandas DataFrame.

    - Compatible with Scikit-Learn Pipelines.
    - Does not raise KeyError if columns are absent during fit or transform.
    - Implements get_feature_names_out for pipeline metadata tracking.
    """

    def __init__(self, var: Union[str, List[str]]):
        self.var = [var] if isinstance(var, str) else list(var)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DropColumnsPandas":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DropColumnsPandas requires a pandas DataFrame.")

        # Identify columns that actually exist in the training matrix
        self.columns_to_drop_ = [col for col in self.var if col in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DropColumnsPandas requires a pandas DataFrame.")

        # Double-check column existence to prevent transform-time KeyError
        cols_to_drop = [col for col in self.columns_to_drop_ if col in X.columns]
        if cols_to_drop:
            return X.drop(columns=cols_to_drop)
        return X.copy()

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        if input_features is None:
            return np.array([c for c in self.var if c not in self.columns_to_drop_], dtype=object)
        return np.array([c for c in input_features if c not in self.columns_to_drop_], dtype=object)


# Alias
DropColumns = DropColumnsPandas


# =====================================================================
# 2. CONSTANT & QUASI-CONSTANT DROPPING (DropConstantFeaturesPandas)
# =====================================================================
class DropConstantFeaturesPandas(BaseEstimator, TransformerMixin):
    """
    Identifies and removes zero-variance (constant) and quasi-constant features.

    Parameters:
    -----------
    threshold : float, default=1.0
        Maximum allowed proportion for the single most frequent value.
        - 1.0   : Drops strictly constant features (variance == 0 / n_unique <= 1).
        - 0.995 : Drops features where a single value represents >= 99.5% of rows.
    tol_na : float, default=1.0
        Maximum missingness threshold (1.0 = drops if 100% NaN).
    """

    def __init__(self, threshold: float = 1.0, tol_na: float = 1.0):
        self.threshold = threshold
        self.tol_na = tol_na

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DropConstantFeaturesPandas":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DropConstantFeaturesPandas requires a pandas DataFrame.")

        n_rows = len(X)
        if n_rows == 0:
            self.features_to_drop_: List[str] = []
            self.kept_features_: List[str] = list(X.columns)
            return self

        self.features_to_drop_ = []

        # 1. Total missingness check
        if self.tol_na < 1.0:
            null_ratios = X.isnull().mean()
            all_null_cols = null_ratios[null_ratios >= self.tol_na].index.tolist()
            self.features_to_drop_.extend(all_null_cols)

        # 2. Cardinality and frequency dominance audit
        cols_to_check = [c for c in X.columns if c not in self.features_to_drop_]
        for col in cols_to_check:
            series = X[col].dropna()
            if len(series) == 0 or series.nunique() <= 1:
                self.features_to_drop_.append(col)
                continue

            if self.threshold < 1.0:
                top_frequency = series.value_counts(normalize=True).iloc[0]
                if top_frequency >= self.threshold:
                    self.features_to_drop_.append(col)

        self.kept_features_ = [c for c in X.columns if c not in self.features_to_drop_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DropConstantFeaturesPandas requires a pandas DataFrame.")
        return X.drop(columns=self.features_to_drop_, errors="ignore")

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        if input_features is None:
            return np.array(self.kept_features_, dtype=object)
        return np.array([c for c in input_features if c not in self.features_to_drop_], dtype=object)


# Alias
DropConstantFeatures= DropConstantFeaturesPandas
# =====================================================================
# 1. POLYNOMIAL FEATURES WRAPPER
# =====================================================================
class SklearnPolynomialWrapper(BaseEstimator, TransformerMixin):
    """Wrapper for PolynomialFeatures returning a clean DataFrame."""
    def __init__(self, variables, degree=2, interaction_only=True, include_bias=False):
        self.variables = [variables] if isinstance(variables, str) else list(variables)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def fit(self, X, y=None):
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        self.poly_.fit(X[self.variables].fillna(0))
        return self

    def transform(self, X):
        X = X.copy()
        poly_matrix = self.poly_.transform(X[self.variables].fillna(0))
        feature_names = [f"poly_{f}" for f in self.poly_.get_feature_names_out(self.variables)]

        df_poly = pd.DataFrame(poly_matrix, columns=feature_names, index=X.index)
        df_poly = df_poly.drop(columns=[f"poly_{v}" for v in self.variables if f"poly_{v}" in df_poly.columns], errors='ignore')

        return pd.concat([X, df_poly], axis=1)

# =====================================================================
# 2. OPTIMIZED PERCENTILE RANKER
# =====================================================================
class PercentileRanker(BaseEstimator, TransformerMixin):
    """Calculates normalized percentile rank (0 to 1) using np.searchsorted."""
    def __init__(self, var):
        self.var = [var] if isinstance(var, str) else list(var)

    def fit(self, X, y=None):
        self.dist_ = {}
        for col in self.var:
            if col in X.columns:
                self.dist_[col] = X[col].dropna().sort_values().values
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var:
            if feature in self.dist_ and len(self.dist_[feature]) > 0:
                ref_dist = self.dist_[feature]
                valid_mask = X[feature].notnull()
                ranks = np.searchsorted(ref_dist, X.loc[valid_mask, feature], side='right')
                X.loc[valid_mask, f"{feature}_RANK"] = ranks / len(ref_dist)
                X.loc[~valid_mask, f"{feature}_RANK"] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None: return np.array(self.var, dtype=object)
        out = list(input_features)
        for col in self.var:
            if col in self.dist_: out.append(f"{col}_RANK")
        return np.array(out, dtype=object)

# =====================================================================
# 3. DISCRETIZATION (Bini)
# =====================================================================
class Bini(BaseEstimator, TransformerMixin):
    def __init__(self, var, nbins):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var
        self.nbins = nbins

    def fit(self, X, y=None):
        self.vari_ = []
        for vari in self.var:
            kbin = KBinsDiscretizer(n_bins=self.nbins, encode='ordinal', strategy='uniform')
            kbin.fit(np.array(X[vari].fillna(0)).reshape(-1, 1))
            self.vari_.append(kbin)
        return self

    def transform(self, X):
        X = X.copy()
        for i, feature in enumerate(self.var):
            X[f"{feature}_bin"] = self.vari_[i].transform(np.array(X[feature].fillna(0)).reshape(-1, 1)).flatten().astype(np.int32)
        return X

# =====================================================================
# 4. ONE-HOT ENCODING (Oneh)
# =====================================================================
class Oneh(BaseEstimator, TransformerMixin):
    def __init__(self, var,drop="first"):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var
        self.drop=drop

    def fit(self, X, y=None):
        self.kbin_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=self.drop)
        if hasattr(self.kbin_, "set_output"):
            self.kbin_.set_output(transform="pandas")
        self.kbin_.fit(X[self.var].astype(str))
        return self

    def transform(self, X):
        X = X.copy()
        encoded = self.kbin_.transform(X[self.var].astype(str))
        if not isinstance(encoded, pd.DataFrame):
            encoded = pd.DataFrame(encoded, columns=self.kbin_.get_feature_names_out(), index=X.index)
        X = pd.concat([X, encoded], axis=1)
        X.drop(columns=self.var, inplace=True)
        return X

# =====================================================================
# 5. ADVANCED IMPUTATION (Flagnan3)
# =====================================================================
class Flagnan3(BaseEstimator, TransformerMixin):
    def __init__(self, var, fill_zero_vars=None):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var
        self.fill_zero_vars = fill_zero_vars if fill_zero_vars else []

    def fit(self, X, y=None):
        self.vari_ = []
        self.val_ = []
        for vari in self.var:
            if vari not in X.columns: continue
            null_pct = X[vari].isnull().mean()
            if null_pct > 0 and null_pct != 1:
                if X[vari].dtype != 'O':
                    if pd.api.types.is_datetime64_any_dtype(X[vari]):
                        self.vari_.append(vari)
                        self.val_.append("interpolate")
                    elif vari in self.fill_zero_vars:
                        self.vari_.append(vari)
                        self.val_.append(0)
                    else:
                        skew_val = X[vari].skew()
                        val = X[vari].median() if abs(skew_val) > 0.75 else X[vari].mean()
                        self.vari_.append(vari)
                        self.val_.append(val)
                else:
                    self.vari_.append(vari)
                    modes = X[vari].mode()
                    self.val_.append(modes[0] if not modes.empty else "Missing")
            elif null_pct == 1:
                self.vari_.append(vari)
                self.val_.append("allnull")
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var:
            if feature in self.vari_:
                idx = self.vari_.index(feature)
                val = self.val_[idx]
                X[f"{feature}_NA"] = X[feature].isnull().astype(np.int32)
                if val == "interpolate":
                    X[feature] = X[feature].interpolate(method='linear')
                elif val == "allnull":
                    X.drop(columns=feature, inplace=True)
                else:
                    X[feature] = X[feature].fillna(val)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None: return np.array(self.var, dtype=object)
        out = list(input_features)
        for feature in self.var:
            if feature in self.vari_:
                idx = self.vari_.index(feature)
                val = self.val_[idx]
                if val == "allnull":
                    if feature in out: out.remove(feature)
                else:
                    out.append(f"{feature}_NA")
        return np.array(out, dtype=object)

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Flagnan4(BaseEstimator, TransformerMixin):
    """Maneja NaN con dos estrategias:

    - default (missingness aleatoria): crea columna `{col}_isna` (1/0) y
      deja el NaN original intacto para que el modelo downstream decida.

    - nan_abs=True (missingness estructural = ausencia): reemplaza el NaN
      por 0 directamente, sin columna flag. Se asume que el NaN ya codifica
      "esta característica no está presente".

    Parameters
    ----------
    var : list of str
        Columnas donde se crea el flag `_isna`.
    var_abs : list of str
        Columnas donde NaN se reemplaza por 0 (ausencia estructural).
        No se crea flag para estas.
    """

    def __init__(self, var=None, var_abs=None):
        self.var = var or []
        self.var_abs = var_abs or []

    def fit(self, X, y=None):
        # Validación temprana
        all_cols = set(X.columns)
        for c in self.var + self.var_abs:
            if c not in all_cols:
                raise ValueError(f"Columna '{c}' no está en X")
        overlap = set(self.var) & set(self.var_abs)
        if overlap:
            raise ValueError(f"Columnas en ambas listas: {overlap}")
        return self

    def transform(self, X):
        X = X.copy()

        # Régimen 1: flag + NaN preservado (missingness aleatoria)
        for c in self.var:
            X[f"{c}_isna"] = X[c].isna().astype(np.int8)

        # Régimen 2: NaN → 0 (missingness estructural)
        for c in self.var_abs:
            X[c] = X[c].fillna(0)

        return X
        
        
# =====================================================================
# 6. OUTLIERS WITH CLIPPING (Flagout2)
# =====================================================================
class Flagout2(BaseEstimator, TransformerMixin):
    def __init__(self, var, distance=3.0):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var
        self.distance = distance

    def fit(self, X, y=None):
        self.limits_ = {}
        for vari in self.var:
            if vari in X.columns:
                mu, sigma = X[vari].mean(), X[vari].std()
                if not np.isnan(mu) and not np.isnan(sigma):
                    self.limits_[vari] = (mu - self.distance * sigma, mu + self.distance * sigma)
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var:
            if feature in self.limits_:
                lb, ub = self.limits_[feature]
                X[f"{feature}_out"] = ((X[feature] < lb) | (X[feature] > ub)).astype(np.int32)
                X[feature] = X[feature].clip(lower=lb, upper=ub)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None: return np.array(self.var, dtype=object)
        out = list(input_features)
        for col in self.var:
            if col in self.limits_: out.append(f"{col}_out")
        return np.array(out, dtype=object)

# =====================================================================
# 7. SIMPLE IMPUTATION (Flagnan)
# =====================================================================
class Flagnan(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var

    def fit(self, X, y=None):
        self.vari_ = []
        self.val_ = []
        for vari in self.var:
            if vari in X.columns and X[vari].isnull().sum() > 0:
                if X[vari].dtype != 'O':
                    self.vari_.append(vari)
                    self.val_.append(X[vari].median())
                else:
                    self.vari_.append(vari)
                    modes = X[vari].mode()
                    self.val_.append(modes[0] if not modes.empty else "Missing")
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var:
            if feature in self.vari_:
                idx = self.vari_.index(feature)
                val = self.val_[idx]
                X[f"{feature}_NA"] = X[feature].isnull().astype(np.int32)
                X[feature] = X[feature].fillna(val)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None: return np.array(self.var, dtype=object)
        out = list(input_features)
        for col in self.var:
            if col in self.vari_: out.append(f"{col}_NA")
        return np.array(out, dtype=object)

# =====================================================================
# 8. OUTLIERS WITHOUT CLIPPING (Flagout)
# =====================================================================
class Flagout(BaseEstimator, TransformerMixin):
    def __init__(self, var, distance=3.0):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var
        self.distance = distance

    def fit(self, X, y=None):
        self.limits_ = {}
        for vari in self.var:
            if vari in X.columns:
                mu, sigma = X[vari].mean(), X[vari].std()
                lb = mu - self.distance * sigma
                ub = mu + self.distance * sigma
                if ((X[vari] < lb) | (X[vari] > ub)).any():
                    self.limits_[vari] = (lb, ub)
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var:
            if feature in self.limits_:
                lb, ub = self.limits_[feature]
                X[f"{feature}_out"] = ((X[feature] < lb) | (X[feature] > ub)).astype(np.int32)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None: return np.array(self.var, dtype=object)
        out = list(input_features)
        for col in self.var:
            if col in self.limits_: out.append(f"{col}_out")
        return np.array(out, dtype=object)

# =====================================================================
# 9. ZERO INFLATED TRANSFORMER
# =====================================================================
class ZeroInflatedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        self.var = [var] if isinstance(var, str) else list(var)

    def fit(self, X, y=None):
        self.dists_ = {}
        for col in self.var:
            if col in X.columns:
                non_zero = X[col].dropna()
                non_zero = non_zero[non_zero != 0].sort_values().values
                self.dists_[col] = non_zero
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.var:
            if col not in X.columns: continue
            ref = self.dists_.get(col, np.array([]))

            X[f"{col}_is_zero"] = (X[col] == 0).astype(int)

            if len(ref) > 0:
                mask = (X[col].notna()) & (X[col] != 0)
                if mask.any():
                    ranks = np.searchsorted(ref, X.loc[mask, col].values, side='right')
                    X.loc[mask, f"{col}_RANK"] = ranks / len(ref)
                X.loc[X[col] == 0, f"{col}_RANK"] = 0.0
            else:
                X[f"{col}_RANK"] = 0.0
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None: return np.array(self.var, dtype=object)
        out = list(input_features)
        for col in self.var:
            out.append(f"{col}_is_zero")
            out.append(f"{col}_RANK")
        return np.array(out, dtype=object)

# =====================================================================
# 10. SEQUENTIAL TARGET ENCODING (CateEncode)
# =====================================================================
class CateEncode(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var

    def fit(self, X, y):
        y_num = y.astype(int) if hasattr(y, "dtype") and pd.api.types.is_categorical_dtype(y) else y
        y_num = pd.Series(y_num, index=X.index)
        temp = pd.concat([X, y_num], axis=1)
        target_col = temp.columns[-1]
        self.encoder_dict_ = {}
        for feature in self.var:
            if feature in X.columns:
                ordered = temp.groupby(feature)[target_col].mean().sort_values().index
                self.encoder_dict_[feature] = {k: i for i, k in enumerate(ordered, 0)}
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var:
            if feature in self.encoder_dict_:
                X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X

# =====================================================================
# 11. TEMPORAL DIFFERENCE (TemporalVariableT)
# =====================================================================
class TemporalVariableT(BaseEstimator, TransformerMixin):
    def __init__(self, var, refvar):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var
        self.refvar = refvar

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var:
            if feature in X.columns and self.refvar in X.columns:
                X[feature] = (X[self.refvar] - X[feature]).dt.days if hasattr(X[self.refvar], 'dt') else X[self.refvar] - X[feature]
        return X

# =====================================================================
# 12. MANUAL MAPPING (Mapper)
# =====================================================================
class Mapper(BaseEstimator, TransformerMixin):
    def __init__(self, var, mapper):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var
        self.mapper = mapper

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var:
            if feature in X.columns:
                X[feature] = X[feature].map(self.mapper)
        return X

# =====================================================================
# 13. MEAN IMPUTATION (MeanImputer)
# =====================================================================
class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var

    def fit(self, X, y=None):
        self.imputer_dict_ = X[self.var].mean().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var:
            if feature in X.columns:
                X[feature] = X[feature].fillna(self.imputer_dict_[feature])
        return X

# =====================================================================
# 14. QUANTILE NORMALIZATION (Qnorm)
# =====================================================================
class Qnorm(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var

    def fit(self, X, y=None):
        from scipy.interpolate import interp1d
        X_sub = X[self.var]
        df_sorted = pd.DataFrame(np.sort(X_sub.values, axis=0), columns=self.var)
        reference_dist = df_sorted.mean(axis=1)
        ranks = np.arange(1, len(X) + 1)
        self.finterp_ = interp1d(ranks, reference_dist, fill_value="extrapolate")
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var:
            if feature in X.columns:
                actual_ranks = X[feature].rank()
                X[feature] = self.finterp_(actual_ranks)
        return X

# =====================================================================
# 15. POWER TRANSFORM (Powert)
# =====================================================================
class Powert(BaseEstimator, TransformerMixin):
    def __init__(self, var, method='yeo-johnson'):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var
        self.method = method

    def fit(self, X, y=None):
        self.vari_ = []
        for feature in self.var:
            if feature in X.columns:
                pt = PowerTransformer(method=self.method)
                pt.fit(X[[feature]])
                self.vari_.append((feature, pt))
        return self

    def transform(self, X):
        X = X.copy()
        for feature, pt in self.vari_:
            if feature in X.columns:
                X[feature] = pt.transform(X[[feature]]).flatten()
        return X

# =====================================================================
# 16. STANDARDIZATION (Escaleo)
# =====================================================================
class Escaleo(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var

    def fit(self, X, y=None):
        self.vari_ = []
        for feature in self.var:
            if feature in X.columns:
                sc = StandardScaler()
                sc.fit(X[[feature]])
                self.vari_.append((feature, sc))
        return self

    def transform(self, X):
        X = X.copy()
        for feature, sc in self.vari_:
            if feature in X.columns:
                X[feature] = sc.transform(X[[feature]]).flatten()
        return X

# =====================================================================
# 17. RARE LABEL ENCODER (RareLabel)
# =====================================================================
class RareLabelPandas(BaseEstimator, TransformerMixin):
    def __init__(self, var, tol=0.02):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var
        self.tol = tol

    def fit(self, X, y=None):
        self.encoder_dict_ = {}
        for feature in self.var:
            if feature in X.columns:
                freq = X[feature].value_counts(normalize=True)
                self.encoder_dict_[feature] = list(freq[freq >= self.tol].index)
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.var:
            if feature in self.encoder_dict_:
                X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]), X[feature], "Rare")
        return X

# =====================================================================
# 18. NLP FEATURE EXTRACTOR
# =====================================================================
class NLPFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, var, method='tfidf', max_features=30, ngram_range=(1,3),max_df=0.975):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_df = max_df
        

    def fit(self, X, y=None):
        self.vectorizers_ = {}
        for col in self.var:
            if col in X.columns:
                text_data = X[col].fillna("").astype(str)
                if self.method == 'tfidf':
                    vec = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range, max_df=self.max_df)
                else:
                    vec = HashingVectorizers(n_features=self.max_features, ngram_range=self.ngram_range, alternate_sign=False)
                vec.fit(text_data)
                self.vectorizers_[col] = vec
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.var:
            if col not in self.vectorizers_: continue
            text_data = X[col].fillna("").astype(str)
            vec = self.vectorizers_[col]
            matrix = vec.transform(text_data).toarray()

            if self.method == 'tfidf':
                names = [f"{col}_tfidf_{f}" for f in vec.get_feature_names_out()]
            else:
                names = [f"{col}_hash_{i}" for i in range(self.max_features)]

            df_nlp = pd.DataFrame(matrix, columns=names, index=X.index)
            X = pd.concat([X, df_nlp], axis=1).drop(columns=[col])
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None: features = self.var
        else: features = list(input_features)

        out_features = []
        for f in features:
            if f in self.vectorizers_:
                if self.method == 'tfidf':
                    out_features.extend([f"{f}_tfidf_{name}" for name in self.vectorizers_[f].get_feature_names_out()])
                else:
                    out_features.extend([f"{f}_hash_{i}" for i in range(self.max_features)])
            elif f not in self.var:
                out_features.append(f)
        return np.array(out_features, dtype=object)


import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, HashingVectorizer,
)


class NLPFeatureExtractor2(BaseEstimator, TransformerMixin):
    """Extrae features de texto de una o varias columnas.

    Parameters
    ----------
    var : list of str
        Columnas a vectorizar. Deben existir en X.
    method : {'tfidf', 'count', 'hash'}
        Vectorizador a usar.
    max_features : int
        Número máximo de features por columna.
    ngram_range : tuple
        Rango de n-gramas.
    min_df : float or int
        Frecuencia mínima (fracción o conteo absoluto) para retener un término.
    max_df : float or int
        Frecuencia máxima (fracción o conteo absoluto) para retener un término.
    binary : bool
        Si True, cada término se cuenta como 0/1 en lugar de frecuencia.
    sublinear_tf : bool
        Solo para method='tfidf'. Aplica 1+log(tf) en lugar de tf crudo.
    lowercase : bool
        Si True, normaliza a minúsculas antes de vectorizar.
    """

    def __init__(
        self,
        var,
        method='tfidf',
        max_features=30,
        ngram_range=(1, 3),
        min_df=0.025,
        max_df=0.975,
        binary=False,
        sublinear_tf=False,
        lowercase=True,
    ):
        if not isinstance(var, list):
            raise ValueError('var must be a list')
        self.var = var
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        self.sublinear_tf = sublinear_tf
        self.lowercase = lowercase

    def _make_vectorizer(self):
        if self.method == 'tfidf':
            return TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                binary=self.binary,
                sublinear_tf=self.sublinear_tf,
                lowercase=self.lowercase,
            )
        if self.method == 'count':
            return CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                binary=self.binary,
                lowercase=self.lowercase,
            )
        if self.method == 'hash':
            return HashingVectorizer(
                n_features=self.max_features,
                ngram_range=self.ngram_range,
                alternate_sign=False,
                binary=self.binary,
                lowercase=self.lowercase,
            )
        raise ValueError(f"method='{self.method}' not supported")

    def fit(self, X, y=None):
        self.vectorizers_ = {}

        for col in self.var:
            if col not in X.columns:
                continue

            text = X[col].fillna("").astype(str)
            vec = self._make_vectorizer()

            if self.method != 'hash':
                vec.fit(text)
            self.vectorizers_[col] = vec

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.var:
            if col not in self.vectorizers_:
                continue

            text = X[col].fillna("").astype(str)
            vec = self.vectorizers_[col]
            matrix = vec.transform(text)

            if sparse.issparse(matrix):
                matrix = matrix.toarray()

            names = self._feature_names(col, vec)
            df_nlp = pd.DataFrame(matrix, columns=names, index=X.index)
            X = pd.concat([X, df_nlp], axis=1).drop(columns=[col])

        return X

    def _feature_names(self, col, vec):
        if self.method == 'hash':
            return [f"{col}_hash_{i}" for i in range(self.max_features)]
        return [f"{col}_{self.method}_{f}" for f in vec.get_feature_names_out()]

    def get_feature_names_out(self, input_features=None):
        cols = list(input_features) if input_features is not None else list(self.var)
        out = []

        for f in cols:
            if f in self.vectorizers_:
                out.extend(self._feature_names(f, self.vectorizers_[f]))
            else:
                out.append(f)

        return np.array(out, dtype=object)
# =====================================================================
# 19. TARGET ENCODER FAST (WITH LEAKAGE SMOOTHING)
# =====================================================================
class TargetEncoderFast(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var

    def fit(self, X, y):
        self.maps_ = {}
        self.global_mean_ = y.mean()
        temp = X[self.var].copy()
        temp['__target__'] = y
        smoothing = 10  # Smoothing is imperative to prevent target leakage on rare categories
        for col in self.var:
            if col in X.columns:
                stats = temp.groupby(col)['__target__'].agg(['mean', 'count'])
                smoothed = (stats['count'] * stats['mean'] + smoothing * self.global_mean_) / (stats['count'] + smoothing)
                self.maps_[col] = smoothed.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.var:
            if col in self.maps_:
                X[f"{col}_TE"] = X[col].map(self.maps_[col]).fillna(self.global_mean_).astype(np.float64)
                X.drop(columns=[col], inplace=True)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None: return np.array(self.var, dtype=object)
        out = [c for c in input_features if c not in self.var]
        out.extend([f"{col}_TE" for col in self.var])
        return np.array(out, dtype=object)

# =====================================================================
# 20. ORDINAL ENCODER (OrdinalEnc)
# =====================================================================
class OrdinalEnc(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        if not isinstance(var, list): raise ValueError('variables must be a list')
        self.var = var

    def fit(self, X, y=None):
        self.maps_ = {}
        for col in self.var:
            if col in X.columns:
                cats = sorted(X[col].dropna().unique())
                self.maps_[col] = {cat: i for i, cat in enumerate(cats)}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.var:
            if col in self.maps_:
                X[col] = X[col].map(self.maps_[col]).astype(pd.Int64Dtype())
        return X

# =====================================================================
# 21. CUSTOM HASHING ENCODER
# =====================================================================
class HashingEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, var, n_features=8):
        self.var = [var] if isinstance(var, str) else list(var)
        self.n_features = n_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.var:
            if col not in X.columns: continue
            hasher = FeatureHasher(n_features=self.n_features, input_type='string')
            raw_tuples = [[(f"{col}_{val}", 1)] for val in X[col].astype(str)]
            matrix = hasher.transform(raw_tuples).toarray()

            cols = [f"{col}_hash_{i}" for i in range(self.n_features)]
            df_hash = pd.DataFrame(matrix, columns=cols, index=X.index)
            X = pd.concat([X.drop(columns=[col]), df_hash], axis=1)
        return X

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
# TARGET ENCODER FAST PANDAS
# (IN-PLACE + TYPE-SAFE + SMOOTHING)
# =====================================================================
class TargetEncoderFastPandas2(BaseEstimator, TransformerMixin):
    """
    Target Encoding con smoothing.

    Características:
    - Pandas puro.
    - Reemplaza la columna original (IN-PLACE SEMÁNTICO).
    - No crea columnas nuevas.
    - Mantiene exactamente el nombre original.
    - Categorías desconocidas -> global_mean_.
    - Valores nulos/desconocidos -> global_mean_.
    - Smoothing configurable.
    - Compatible con sklearn Pipeline.
    """

    def __init__(self, var, smoothing=10):
        self.var = [var] if isinstance(var, str) else list(var)
        self.smoothing = smoothing

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "TargetEncoderFastPandas2 requiere un pandas DataFrame."
            )

        # Validación básica de smoothing
        if self.smoothing < 0:
            raise ValueError("smoothing debe ser >= 0.")

        # Convertir y a Series numérica Float64
        if isinstance(y, pd.Series):
            y_pd = pd.to_numeric(y, errors="coerce").astype(float)
        else:
            y_pd = pd.Series(y, index=X.index, dtype="float64")

        # Alinear índice con X
        y_pd = y_pd.reindex(X.index)

        if len(y_pd) != len(X):
            raise ValueError(
                "X e y deben tener la misma cantidad de observaciones."
            )

        # Media global ignorando NaN
        self.global_mean_ = float(y_pd.mean())

        if np.isnan(self.global_mean_):
            raise ValueError(
                "El target no contiene valores numéricos válidos."
            )

        self.maps_ = {}

        # DataFrame temporal
        temp = X.copy()
        temp["__target__"] = y_pd.to_numpy()

        for col in self.var:

            if col not in X.columns:
                continue

            # GroupBy sobre la variable original.
            #
            # dropna=False permite conservar explícitamente la categoría
            # NaN durante el cálculo del mapa.
            stats = (
                temp.groupby(
                    col,
                    dropna=False,
                    sort=False,
                    observed=False
                )["__target__"]
                .agg(["mean", "count"])
            )

            # Smoothing:
            #
            # encoded =
            #   (count * category_mean + smoothing * global_mean)
            #   / (count + smoothing)
            #
            stats["encoded"] = (
                (
                    stats["count"] * stats["mean"]
                    + self.smoothing * self.global_mean_
                )
                /
                (
                    stats["count"] + self.smoothing
                )
            )

            # Guardamos Series indexada por categoría.
            #
            # Esto es más robusto que convertir todo a string:
            # conserva correctamente int, float, bool, categorías, etc.
            self.maps_[col] = stats["encoded"]

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "TargetEncoderFastPandas2 requiere un pandas DataFrame."
            )

        X_out = X.copy()

        for col in self.var:

            if col not in self.maps_:
                continue

            if col not in X_out.columns:
                continue

            mapping = self.maps_[col]

            # Map por índice de categorías.
            encoded = X_out[col].map(mapping)

            # Categorías nunca vistas en TRAIN + NaN:
            # global mean.
            encoded = encoded.fillna(self.global_mean_)

            # Garantizar salida numérica.
            X_out[col] = pd.to_numeric(
                encoded,
                errors="coerce"
            ).astype("float64")

            # Segundo blindaje:
            # cualquier conversión inesperada a NaN -> global mean.
            X_out[col] = X_out[col].fillna(self.global_mean_)

        return X_out

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            return np.array(self.var, dtype=object)

        return np.array(input_features, dtype=object)



# =====================================================================
# BINI PANDAS
# (IN-PLACE + TYPE-SAFE + BLINDADO CONTRA STRINGS)
# =====================================================================
class BiniPandas2(BaseEstimator, TransformerMixin):
    """
    Discretización uniforme compatible con BiniPolars2.

    Características:
    - Pandas puro.
    - Los bins se calculan SOLO durante fit.
    - Reemplaza la columna original.
    - No crea columnas nuevas.
    - No genera *_bin.
    - Convierte valores numéricos/string numérico de forma segura.
    - Valores no convertibles -> 0.
    - Misma lógica de np.linspace utilizada por BiniPolars2.
    """

    def __init__(self, var, nbins=5):
        self.var = [var] if isinstance(var, str) else list(var)
        self.nbins = nbins

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "BiniPandas2 requiere un pandas DataFrame."
            )

        if not isinstance(self.nbins, int) or self.nbins < 2:
            raise ValueError(
                "nbins debe ser un entero >= 2."
            )

        self.bins_ = {}

        for col in self.var:

            if col not in X.columns:
                continue

            # Equivalente conceptual a:
            # pl.col(col).cast(pl.Float64, strict=False)
            series = pd.to_numeric(
                X[col],
                errors="coerce"
            ).dropna()

            if len(series) == 0:
                continue

            min_val = float(series.min())
            max_val = float(series.max())

            # Igual que BiniPolars2:
            # si min == max no se crean breaks.
            if min_val < max_val:

                self.bins_[col] = np.linspace(
                    min_val,
                    max_val,
                    self.nbins + 1
                )[1:-1].tolist()

        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "BiniPandas2 requiere un pandas DataFrame."
            )

        X_out = X.copy()

        for col in self.var:

            if col not in self.bins_:
                continue

            if col not in X_out.columns:
                continue

            # Conversión segura
            values = pd.to_numeric(
                X_out[col],
                errors="coerce"
            )

            # Mismo concepto que:
            # fill_null(0.0)
            values = values.fillna(0.0)

            # pd.cut con labels=False devuelve:
            # 0, 1, 2, ...
            #
            # Los breaks son exactamente los mismos
            # calculados en fit().
            X_out[col] = (
                pd.cut(
                    values,
                    bins=[-np.inf] + self.bins_[col] + [np.inf],
                    labels=False,
                    include_lowest=True
                )
                .astype("int32")
            )

        return X_out

    def get_feature_names_out(self, input_features=None):

        if input_features is None:
            return np.array(self.var, dtype=object)

        return np.array(input_features, dtype=object)
# =====================================================================
# PERCENTILE RANKER PANDAS (IN-PLACE)
# =====================================================================
class PercentileRankerPandas2(BaseEstimator, TransformerMixin):
    """
    Calcula el rango percentil empírico (0 a 1) sobreescribiendo
    la columna original in-place sin generar columnas auxiliares (*_RANK).
    """
    def __init__(self, var):
        self.var = [var] if isinstance(var, str) else list(var)

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PercentileRankerPandas2 requiere un pandas DataFrame.")
        self.dist_ = {}
        for col in self.var:
            if col in X.columns:
                series = pd.to_numeric(X[col], errors="coerce").dropna()
                if len(series) > 0:
                    self.dist_[col] = np.sort(series.to_numpy())
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PercentileRankerPandas2 requiere un pandas DataFrame.")
        X_out = X.copy()
        for col in self.var:
            if col in self.dist_ and len(self.dist_[col]) > 0:
                ref = self.dist_[col]
                vals = pd.to_numeric(X_out[col], errors="coerce")
                valid_mask = vals.notnull()

                if valid_mask.any():
                    valid_vals = vals[valid_mask].to_numpy()
                    ranks = np.searchsorted(ref, valid_vals, side="right") / float(len(ref))
                    
                    # Reemplazo in-place respetando NaNs
                    res = pd.Series(np.nan, index=X_out.index, dtype="float64")
                    res.iloc[valid_mask.to_numpy().nonzero()[0]] = ranks
                    X_out[col] = res
                else:
                    X_out[col] = np.nan
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.var, dtype=object)
        return np.array(input_features, dtype=object)


# Alias de conveniencia
PercentileRanker2 = PercentileRankerPandas2


# =====================================================================
# ZERO INFLATED TRANSFORMER PANDAS (IN-PLACE)
# =====================================================================
class ZeroInflatedTransformerPandas2(BaseEstimator, TransformerMixin):
    """
    Transformación Zero-Inflated in-place:
    - 0 se mantiene como 0.0 (conserva la masa en cero).
    - Valores no nulos y != 0 se mapean a su percentil rank entre (0, 1].
    - Mantiene NaN como NaN.
    - Reemplaza la columna original sin crear '_is_zero' ni '_RANK'.
    """
    def __init__(self, var):
        self.var = [var] if isinstance(var, str) else list(var)

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZeroInflatedTransformerPandas2 requiere un pandas DataFrame.")
        self.dists_ = {}
        for col in self.var:
            if col in X.columns:
                series = pd.to_numeric(X[col], errors="coerce").dropna()
                non_zero = series[series != 0]
                if len(non_zero) > 0:
                    self.dists_[col] = np.sort(non_zero.to_numpy())
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZeroInflatedTransformerPandas2 requiere un pandas DataFrame.")
        X_out = X.copy()
        for col in self.var:
            if col in self.dists_ and len(self.dists_[col]) > 0:
                ref = self.dists_[col]
                vals = pd.to_numeric(X_out[col], errors="coerce")
                
                res = pd.Series(np.nan, index=X_out.index, dtype="float64")
                
                # 1. Ceros se quedan en 0.0
                zero_mask = (vals == 0)
                res[zero_mask] = 0.0
                
                # 2. Valores distintos de cero se rankean entre (0, 1]
                non_zero_mask = vals.notnull() & (~zero_mask)
                if non_zero_mask.any():
                    non_zero_vals = vals[non_zero_mask].to_numpy()
                    ranks = np.searchsorted(ref, non_zero_vals, side="right") / float(len(ref))
                    res[non_zero_mask] = ranks
                
                X_out[col] = res
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.var, dtype=object)
        return np.array(input_features, dtype=object)


