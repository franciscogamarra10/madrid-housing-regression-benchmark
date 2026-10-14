"""Project Configuration, Data Loading (Pandas & Polars), and Naming Utilities."""
from __future__ import annotations
import re
import pandas as pd
import polars as pl
from typing import Union

try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False


"""skim.py — Resúmenes rápidos de DataFrames con salida de texto."""

import warnings
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import tabulate

# Soporte para GroupBy
try:
    from pandas.core.groupby.generic import DataFrameGroupBy
except ImportError:
    DataFrameGroupBy = pd.core.groupby.generic.DataFrameGroupBy

Frame = pd.DataFrame | DataFrameGroupBy
Descriptor = Callable[[pd.Series], dict[str, Any]]

# ───────────────────────── Utilidades de texto ─────────────────────────
_BARS = " ▂▃▄▅▆▇"

def text_barchart(values: list[int] | np.ndarray) -> str:
    if len(values) == 0: return ""
    lo, hi = min(values), max(values)
    if lo == hi:
        lo, hi = (0, 1) if lo == 0 else (lo - 1, hi + 1)
    
    bins = np.linspace(lo, hi, len(_BARS) + 1)
    chars = pd.cut(values, bins=bins, labels=list(_BARS), include_lowest=True)
    return "".join(chars.dropna())

def text_histogram(values: pd.Series, bins: int = 10) -> str:
    # Optimización: dropna una sola vez y extraer valores numpy directamente
    valid_data = values.dropna().values
    if len(valid_data) == 0: return ""
    counts, _ = np.histogram(valid_data, bins=bins)
    return text_barchart(counts)

def shorten(text: str, width: int = 30, suffix: str = "[...]") -> str:
    return text if len(text) <= width else text[: width - len(suffix)] + suffix

def top_counts(column: pd.Series, num: int = 3) -> str:
    counts = column.value_counts().head(num)
    total_valid = column.count() 
    
    if total_valid == 0:
        return ""
        
    return ", ".join(
        f"{shorten(str(k))}: {v} ({(v / total_valid):.1%})" 
        for k, v in counts.items()
    )


# ───────────────────── Descriptores por tipo ─────────────────────

def _numeric(col: pd.Series) -> dict[str, Any]:
    # Cuantiles vectorizados
    q_vals = col.quantile([0, 0.25, 0.5, 0.75, 1]).tolist() if not col.dropna().empty else [pd.NA]*5
    
    return {
        "n_unique": col.nunique(),  # <-- Agregado para detectar variables discretas
        "mean":     col.mean(),
        "sd":       col.std(),
        "p0":       q_vals[0],
        "p25":      q_vals[1],
        "p50":      q_vals[2],
        "p75":      q_vals[3],
        "p100":     q_vals[4],
        "hist":     text_histogram(col),
    }

def _categorical(col: pd.Series) -> dict[str, Any]:
    return {"n_unique": col.nunique(), "top_counts": top_counts(col)}

def _boolean(col: pd.Series) -> dict[str, Any]:
    return {"mean": col.mean(), "top_counts": top_counts(col)}

def _datetime(col: pd.Series) -> dict[str, Any]:
    return {
        "n_unique": col.nunique(),
        "min":  col.min(),
        "max":  col.max(),
    }

# Diccionario de ruteo según tipo de dato
_DISPATCH: dict[str, Descriptor] = {
    "boolean":  _boolean,
    "number":   _numeric,
    "category": _categorical,
    "string":   _categorical,
    "datetime": _datetime,
    "object":   _categorical,
}

def _describe_column(col: pd.Series, func: Descriptor) -> dict[str, Any]:
    if pd.api.types.is_numeric_dtype(col):
        if col.isin([np.inf, -np.inf]).any():
            warnings.warn(f'Columna "{col.name}" contiene ±Inf; reemplazando por NA.')
            col = col.replace([np.inf, -np.inf], pd.NA)

    return {"name": col.name, "na_count": col.isna().sum(),"na_count_fr": col.isna().sum()/len(col), **func(col)}

def describe_columns(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame | None]]:
    for type_, func in _DISPATCH.items():
        exclude = "boolean" if type_ == "number" else None
        subset = df.select_dtypes(include=type_, exclude=exclude)
        
        if subset.empty:
            continue
            
        summary = pd.DataFrame([_describe_column(subset[c], func) for c in subset.columns])
        yield type_, summary

# ───────────────────────── Formatter ─────────────────────────

class TextFormatter:
    def __init__(self, obj: Frame, width: int = 100):
        if not isinstance(obj, (pd.DataFrame, DataFrameGroupBy)):
            raise TypeError(f"skim no soporta {type(obj).__name__}")
        self.obj = obj
        self.width = width
        self.sep = "─"

    def _header(self, title: str) -> str:
        lead = f"{self.sep * 2} {title} "
        return lead + self.sep * max(self.width - len(lead), 0)

    @staticmethod
    def _table(df: pd.DataFrame, **kw: Any) -> str:
        return tabulate.tabulate(df, headers="keys", tablefmt="simple", **kw)

    def overview(self) -> str:
        df = self.obj.obj if isinstance(self.obj, DataFrameGroupBy) else self.obj
        
        shape_df = pd.DataFrame({"type": ["Rows", "Columns"], "value": df.shape}).set_index("type")
        dtypes_df = df.dtypes.value_counts().to_frame("Count")

        parts = [
            self._header("Data Summary"),
            self._table(shape_df, floatfmt=".0f"),
            "\nColumn type frequency:",
            self._table(dtypes_df),
            "\n"
        ]
        return "\n".join(parts)

    def _describe(self, df: pd.DataFrame) -> str:
        blocks = []
        described_cols = set()

        for type_, summary in describe_columns(df):
            described_cols.update(summary["name"])
            blocks.append(
                f"{self._header(f'Variable type: {type_}')}\n"
                f"{self._table(summary, floatfmt='.3g')}"
            )

        if missing := set(df.columns) - described_cols:
            warnings.warn(f"Columnas sin descriptor: {missing}")

        return "\n\n".join(blocks) + "\n\n"

    def summaries(self) -> str:
        if isinstance(self.obj, pd.DataFrame):
            return self._describe(self.obj)

        blocks = []
        keys = [self.obj.keys] if isinstance(self.obj.keys, str) else list(self.obj.keys)
        
        for group, group_df in self.obj:
            vals = [group] if isinstance(group, (str, int, float)) else list(group)
            label = ", ".join(f"{k}={v}" for k, v in zip(keys, vals))
            
            blocks.append(
                f"{self._header(f'Group ({len(group_df)} rows): {label}')}\n"
                f"{self._describe(group_df)}"
            )
        return "\n".join(blocks)

def skim(obj: Frame, width: int = 100) -> None:
    fmt = TextFormatter(obj, width=width)
    print(fmt.overview())
    print(fmt.summaries())
    
    
def analyze_high_cardinality(df_pl, high_cat_cols):
    """
    Analiza columnas de alta cardinalidad para decidir si son
    categorías puras (ej. IDs, ZipCodes) o Texto Libre (ej. Títulos, Reseñas).
    """
    resultados = []
    try:
        df_pl = pl.from_pandas(df_pl)
    except AttributeError:
        pass
    for col in high_cat_cols:
        # Extraemos la serie sin nulos y forzamos a string
        serie = df_pl.select(pl.col(col).cast(pl.String).drop_nulls())

        if len(serie) == 0:
            continue

        n_unique = serie.select(pl.col(col).n_unique()).item()

        # Heurísticas de texto: Longitud promedio y cantidad de espacios
        avg_len = serie.select(pl.col(col).str.len_bytes().mean()).item()
        avg_spaces = serie.select(pl.col(col).str.count_matches(" ").mean()).item()

        # Regla simple: Si tiene más de 1 espacio en promedio y es largo, es texto
        is_text = (avg_spaces >= 1.0) and (avg_len > 10)

        resultados.append({
            "Feature": col,
            "Categorías_Únicas": n_unique,
            "Longitud_Promedio": round(avg_len, 2),
            "Espacios_Promedio": round(avg_spaces, 2),
            "Sugerencia": "NLP (CountVectorizer/TF-IDF)" if is_text else "Ordinal / Target Encoding"
        })

    return pd.DataFrame(resultados).sort_values("Categorías_Únicas", ascending=False)    
# ==============================================================================
# PANDAS DATA LOADER
# ==============================================================================
def load_data_rigorous(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Loads CSV files into Pandas with robust missing value handling and encoding fallback."""
    na_values = ["", " ", "NULL", "null", "NaN", "none", "NA", "undefined", "nan", "N/A"]
    try:
        return pd.read_csv(path, na_values=na_values, keep_default_na=True, encoding=encoding)
    except UnicodeDecodeError:
        print(f"⚠️ Pandas loading failed with '{encoding}'. Retrying with 'latin-1'...")
        return pd.read_csv(path, na_values=na_values, keep_default_na=True, encoding="latin-1")


# ==============================================================================
# POLARS DATA LOADER (LAZY & EAGER EVALUATION)
# ==============================================================================
def load_data_rigorous_polars(path: str, lazy: bool = True, encoding: str = "utf8-lossy") -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Loads CSV files into Polars supporting both Lazy evaluation (scan_csv) and Eager execution (read_csv).

    Args:
        path (str): File path to the CSV dataset.
        lazy (bool): If True, returns a Polars LazyFrame query plan. If False, returns an eager DataFrame.
        encoding (str): Character encoding fallback (defaults to 'utf8-lossy').

    Returns:
        pl.DataFrame or pl.LazyFrame
    """
    na_values = ["", " ", "NULL", "null", "NaN", "none", "NA", "undefined", "nan", "N/A"]

    try:
        if lazy:
            return pl.scan_csv(path, null_values=na_values, encoding=encoding)
        else:
            return pl.read_csv(path, null_values=na_values, encoding=encoding)
    except Exception:
        print(f"⚠️ Polars loading failed with '{encoding}'. Retrying with 'latin1'...")
        if lazy:
            return pl.scan_csv(path, null_values=na_values, encoding="latin1")
        else:
            return pl.read_csv(path, null_values=na_values, encoding="latin1")


def to_snake_case(name: str) -> str:
    """Converts string column names to clean snake_case format."""
    name = name.lower().strip().replace(" ", "_")
    if HAS_UNIDECODE:
        name = unidecode(name)
    return re.sub(r"\W+", "", name)


def rename_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Standardizes all DataFrame column names to snake_case."""
    mapping = {col: to_snake_case(col) for col in df.columns}
    return df.rename(columns=mapping, inplace=inplace)



# %%writefile src/config/cleaners.py
"""
cleaner.py - Utilidades simples de limpieza y downcasting para DataFrames.

Autor: 
"""

import re
import numpy as np
import pandas as pd
from typing import Iterable, Optional

__all__ = [
    "downcast_numerics",
    "to_category",
    "clean_column_names",
    "drop_empty_and_constant",
    "clean_dataframe",
]


# ---------------------------------------------------------------------------
# 1) DOWNCASTING
# ---------------------------------------------------------------------------
def downcast_numerics(
    df: pd.DataFrame,
    exclude: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Reduce el tamaño de columnas numéricas al menor dtype posible.

    - int64  -> int8 / int16 / int32 (si los valores lo permiten)
    - float64 -> float32 (o float16 si el rango es chico)

    Parameters
    ----------
    df : pd.DataFrame
    exclude : iterable de nombres de columnas a no tocar.

    Returns
    -------
    pd.DataFrame (copia)
    """
    df = df.copy()
    exclude = set(exclude or [])

    # Enteros
    int_cols = df.select_dtypes(include=["int64", "Int64"]).columns.difference(exclude)
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Floats
    float_cols = df.select_dtypes(include=["float64"]).columns.difference(exclude)
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


# ---------------------------------------------------------------------------
# 2) OBJECT -> CATEGORY
# ---------------------------------------------------------------------------
def to_category(
    df: pd.DataFrame,
    threshold: float = 0.05,
    exclude: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Convierte columnas 'object' a 'category' cuando la razón de valores
    únicos (incluyendo NaN) es menor que `threshold`.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float en [0, 1]. 0.05 = 5% de valores únicos.
    exclude : columnas a no convertir.

    Returns
    -------
    pd.DataFrame (copia)
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1")

    df = df.copy()
    exclude = set(exclude or [])
    n = len(df)
    if n == 0:
        return df

    for col in df.select_dtypes(include=["object"]).columns:
        if col in exclude:
            continue
        ratio = df[col].nunique(dropna=False) / n
        if ratio < threshold:
            df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------------------------
# 3) LIMPIAR NOMBRES DE COLUMNAS
# ---------------------------------------------------------------------------
def clean_column_names(df: pd.DataFrame, max_len: int = 40) -> pd.DataFrame:
    """
    Normaliza los nombres de columnas:
      - CamelCase -> snake_case
      - Quita/reescribe caracteres problemáticos
      - Colapsa guiones bajos repetidos
      - Trunca a `max_len` y resuelve duplicados con sufijo _2, _3...

    Returns
    -------
    pd.DataFrame (copia)
    """
    df = df.copy()

    def _clean(name: str) -> str:
        s = str(name)
        # CamelCase -> snake_case
        s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
        # Reemplazos comunes
        s = re.sub(r"[\s\-\.\(\)\[\]\{\}\/\\]+", "_", s)
        s = re.sub(r"[^\w]+", "_", s)          # cualquier otra cosa rara
        s = re.sub(r"_+", "_", s).strip("_").lower()
        if len(s) > max_len:
            s = s[:max_len].rstrip("_")
        return s or "col"

    new_cols = [_clean(c) for c in df.columns]

    # Resolver duplicados
    seen: dict[str, int] = {}
    fixed = []
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            fixed.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            fixed.append(c)

    df.columns = fixed
    return df


# ---------------------------------------------------------------------------
# 4) ELIMINAR COLUMNAS VACÍAS / CONSTANTES
# ---------------------------------------------------------------------------
def drop_empty_and_constant(
    df: pd.DataFrame,
    drop_empty_rows: bool = False,
) -> pd.DataFrame:
    """
    Elimina:
      - columnas 100% NaN
      - columnas con un único valor (incluye NaN)
      - opcionalmente filas 100% NaN

    Returns
    -------
    pd.DataFrame (copia)
    """
    df = df.copy()

    # Columnas vacías o constantes
    nunique = df.nunique(dropna=False)
    cols_drop = nunique[nunique <= 1].index
    df = df.drop(columns=cols_drop)

    if drop_empty_rows:
        df = df.dropna(how="all")

    return df


# ---------------------------------------------------------------------------
# 5) WRAPPER
# ---------------------------------------------------------------------------
def clean_dataframe(
    df: pd.DataFrame,
    *,
    clean_names: bool = True,
    drop_constants: bool = True,
    category: bool = True,
    cat_threshold: float = 0.05,
    downcast: bool = True,
    exclude: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pipeline simple de limpieza:

      1. clean_column_names
      2. drop_empty_and_constant
      3. to_category
      4. downcast_numerics

    Parameters
    ----------
    df : pd.DataFrame
    clean_names, drop_constants, category, downcast : bool
        Activar/desactivar cada paso.
    cat_threshold : float
        Umbral para convertir object -> category.
    exclude : iterable
        Columnas que no se deben downcastear ni convertir a category.
    verbose : bool
        Imprime un resumen del antes/después.

    Returns
    -------
    pd.DataFrame
    """
    original_shape = df.shape
    original_mem  = df.memory_usage(deep=True).sum() / 1024**2  # MB

    out = df.copy()

    if clean_names:
        out = clean_column_names(out)
    if drop_constants:
        out = drop_empty_and_constant(out)
    if category:
        out = to_category(out, threshold=cat_threshold, exclude=exclude)
    if downcast:
        out = downcast_numerics(out, exclude=exclude)

    if verbose:
        new_mem = out.memory_usage(deep=True).sum() / 1024**2
        pct = (1 - new_mem / original_mem) * 100 if original_mem else 0
        print(
            f"Filas:   {original_shape[0]:>8,} -> {out.shape[0]:>8,}\n"
            f"Columnas:{original_shape[1]:>8,} -> {out.shape[1]:>8,}\n"
            f"Memoria: {original_mem:>8.2f} MB -> {new_mem:>8.2f} MB "
            f"({pct:+.1f}%)"
        )

    return out

# df_limpio = clean_dataframe(
#     df,
#     cat_threshold=0.05,
#     exclude=["id_cliente"],   # IDs no se tocan
#     verbose=True,
# )

# # Solo downcasting rápido
# from limpieza import downcast_numerics
# df = downcast_numerics(df)
