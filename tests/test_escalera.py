"""La escalera de decisiones: los pasos que no se probaron no se inventan y la pieza de cada modelo es su mejor versión según el peor mes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pml import escalera


def _datos():
    idx = pd.date_range("2024-07-01", periods=40)
    Y = pd.DataFrame(100.0, index=idx, columns=range(1, 25))
    X = Y + 10.0                                           # el MDA se equivoca por 10
    return idx, Y, X


def test_mejor_version_prefiere_lo_sencillo_si_empata():
    idx, Y, X = _datos()
    pasos = {"M": {"sin MDA": Y + 4.0,                     # el mejor, pero "sin MDA" no cuenta en un modelo con MDA
                   "con MDA": Y + 8.0,
                   "+ nodos": Y + 6.0,
                   "todas las variables": Y + 5.0,
                   "PCA": Y + 4.99}}                       # 0.1 puntos mejor que "todas": empate, gana la más sencilla
    T = escalera.trayectoria(pasos, Y, X, idx)
    assert list(T.paso) == ["sin MDA", "con MDA", "+ nodos", "todas las variables", "PCA"]
    assert escalera.mejor_version(pasos, T)["M"][0] == "todas las variables"


def test_el_peor_mes_castiga_un_mes_desastroso():
    """Un modelo que gana mucho cinco meses y pierde uno tiene mejor error medio que otro que gana poco todos los meses, pero peor 'peor mes'."""
    idx = pd.date_range("2024-07-01", periods=180)
    Y = pd.DataFrame(100.0, index=idx, columns=range(1, 25)); X = Y + 10.0
    A = Y + 5.0; A.loc[idx.month == 12] = Y.loc[idx.month == 12] + 12.0         # 50% mejor que el MDA cinco meses, 20% peor en diciembre
    B = Y + 9.0                                                               # 10% mejor que el MDA todos los meses
    pasos = {"A": {"con MDA": A}, "B": {"con MDA": B}}
    T = escalera.trayectoria(pasos, Y, X, idx).set_index("modelo")
    assert T.loc["A", "cambio_desarrollo"] < T.loc["B", "cambio_desarrollo"]          # por error medio, A gana
    assert T.loc["B", "peor_mes_desarrollo"] < T.loc["A", "peor_mes_desarrollo"]      # por peor mes, B gana


def test_los_pasos_sin_probar_no_se_inventan():
    idx, Y, X = _datos()
    T = escalera.trayectoria({"M": {"con MDA": Y + 8.0}}, Y, X, idx)
    assert list(T.paso) == ["con MDA"]


def test_grupos_de_la_escalera_dividen_todo():
    """Exógenas y modelado estadístico se reparten, sin traslape, todo lo que no es la base; las familias cubren todo."""
    from pml.todo import CALENDARIO, ESTADISTICO, EXOGENAS, FAMILIAS, NUCLEO, TODO
    assert sorted(EXOGENAS + ESTADISTICO) == sorted(set(TODO) - set(NUCLEO)) and not set(EXOGENAS) & set(ESTADISTICO)
    assert sorted(g for v in FAMILIAS.values() for g in v) == sorted(TODO)
    assert set(CALENDARIO) <= set(EXOGENAS)
