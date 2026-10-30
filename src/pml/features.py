"""Dos utilidades que usan los grupos de variables (`features2.py`) y algunos notebooks.

  - La transformación asinh: el precio tiene colas muy pesadas (horas de más de 10,000 MXN/MWh) y a veces vale cero o
    menos, así que el logaritmo no sirve. asinh((precio − mediana) / MAD) se comporta como una línea cerca de la mediana
    y como un logaritmo en las colas. La mediana y la MAD se fijan sólo con datos de entrenamiento.
  - Los días feriados oficiales de México.
"""
import numpy as np
import pandas as pd


def asinh_fit(y_train: np.ndarray):
    """Mediana y MAD (escalada por 1.4826 para que se compare con una desviación estándar), sólo con entrenamiento."""
    med = float(np.median(y_train)); mad = float(1.4826 * np.median(np.abs(y_train - med)))
    return med, mad


def asinh_t(x, med, mad):
    """Del precio a la escala del modelo."""
    return np.arcsinh((x - med) / mad)


def asinh_inv(z, med, mad):
    """De la escala del modelo al precio."""
    return np.sinh(z) * mad + med


def feriados_mx(idx: pd.DatetimeIndex) -> pd.Series:
    """1 si el día es feriado oficial en México, 0 si no."""
    import holidays
    h = holidays.country_holidays("MX", years=sorted(set(idx.year)))
    return pd.Series([int(d.date() in h) for d in idx], index=idx)
