"""Carga de los precios descargados y paso a matrices día × hora.

Los precios vienen en formato largo (una fila por nodo, fecha y hora, con el PML y sus componentes: energía, pérdidas
y congestión). Casi todo el proyecto trabaja con una sola serie de zona, el promedio de los 9 nodos de Monclova, en
forma de matriz con una fila por día y una columna por hora (1 a 24).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def cargar_largo():
    """(MTR, MDA) de los 9 nodos de Monclova, en formato largo, tal como los dejó `download_all.py`."""
    mtr = pd.read_parquet(ROOT / "data/processed/pml_mtr.parquet")
    mda = pd.read_parquet(ROOT / "data/processed/pml_mda.parquet")
    return mtr, mda


def matriz(df: pd.DataFrame, col="pml", nodo=None) -> pd.DataFrame:
    """Matriz días × 24 horas de la columna `col` (pml, pml_ene, pml_per o pml_cng).
    Con nodo=None es el promedio de los nodos presentes (la zona); con un nodo, sólo ese nodo."""
    d = df if nodo is None else df[df.nodo == nodo]
    return d.groupby(["fecha", "hora"])[col].mean().unstack("hora")
