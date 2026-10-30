"""Mapas de los nodos (notebooks 06 y 09): los contornos de los estados de México y cada nodo en su municipio.

Los contornos vienen de un GeoJSON público de los estados de México (data/external/geo/mexico_estados.json) y la
ubicación de cada nodo de `red.ubicaciones()`. Se dibuja con matplotlib, sin librerías de mapas.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotstyle as ps

ROOT = Path(__file__).resolve().parents[2]
GEO = ROOT / "data/external/geo/mexico_estados.json"
URL = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
NORESTE = dict(lon=(-104.5, -96.5), lat=(21.5, 29.5))         # recuadro para el acercamiento al noreste


def _estados():
    """Contornos de los estados de México (se descargan la primera vez y se guardan en data/external/geo)."""
    if not GEO.exists():
        import urllib.request
        GEO.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, GEO)
    return json.loads(GEO.read_text())


def fondo(ax, lon=(-118, -86), lat=(14, 33)):
    """Los estados de México en gris claro, en coordenadas lon/lat."""
    for f in _estados()["features"]:
        g = f["geometry"]
        polis = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for p in polis:
            anillo = np.array(p[0])
            ax.fill(anillo[:, 0], anillo[:, 1], color="#f1f0ec", ec="#c3c2b7", lw=0.5, zorder=0)
    ax.set_xlim(*lon); ax.set_ylim(*lat)
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(lat))))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def separar(df: pd.DataFrame, paso=0.18) -> pd.DataFrame:
    """Varios nodos comparten municipio (misma coordenada). Los acomoda en un pequeño círculo para que se vean todos."""
    df = df.copy()
    for _, g in df.groupby(["lat", "lon"]):
        if len(g) > 1:
            ang = np.linspace(0, 2 * np.pi, len(g), endpoint=False)
            r = paso * (1 + len(g) / 8)
            df.loc[g.index, "lat"] += r * np.sin(ang); df.loc[g.index, "lon"] += r * np.cos(ang)
    return df


def nodos(ax, ub: pd.DataFrame, valor: pd.Series = None, cmap="Blues", vmin=None, vmax=None, etiqueta=None, tam=34, leyenda=False):
    """Nodos de la red coloreados por `valor` (o por grupo si no hay valor) y Monclova como estrella naranja.
    Devuelve el objeto de color para la barra de escala."""
    ub = separar(ub)
    red = ub[ub.grupo != "monclova"]
    sc = None
    marcas = {"31 originales": "o", "400 kV nuevos": "s"}
    for g, mk in marcas.items():
        d = red[red.grupo == g]
        if valor is not None:
            v = valor.reindex(d.index)
            sc = ax.scatter(d.lon, d.lat, c=v, cmap=cmap, vmin=vmin, vmax=vmax, marker=mk, s=tam, ec="white", lw=0.6, zorder=3,
                            label=f"{g} ({len(d)})")
        else:
            ax.scatter(d.lon, d.lat, color=ps.AZUL if g == "31 originales" else ps.AQUA, marker=mk, s=tam, ec="white", lw=0.6,
                       zorder=3, label=f"{g} ({len(d)})")
    m = ub[ub.grupo == "monclova"]
    ax.scatter(m.lon.mean(), m.lat.mean(), marker="*", s=tam * 9, color=ps.NARANJA, ec="#0b0b0b", lw=0.8, zorder=5,
               label="Monclova (9 nodos)")
    if etiqueta:
        ax.annotate("Monclova", (m.lon.mean(), m.lat.mean()), xytext=(8, 8), textcoords="offset points", fontsize=9,
                    fontweight="bold", color="#0b0b0b", zorder=6)
    if leyenda:                                   # marcadores neutros: el color de los nodos es el valor, no el grupo
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker=mk, ls="", mfc=ps.GRIS, mec="white", ms=7, label=f"{g} ({(red.grupo == g).sum()})")
             for g, mk in marcas.items()]
        h.append(Line2D([], [], marker="*", ls="", mfc=ps.NARANJA, mec="#0b0b0b", ms=14, label="Monclova (9 nodos)"))
        ax.legend(handles=h, loc="lower left", fontsize=8)
    return sc
