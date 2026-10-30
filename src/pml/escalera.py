"""La escalera de decisiones: cómo cambia el error de cada modelo al pasar por el mismo pipeline (notebook 08).

Pasos: sin MDA → con MDA → + nodos (64) → y, desde ahí, por separado + exógenas y + modelado estadístico → todas las
variables (las dos juntas) → PCA (rezagos, MDA y calendario tal cual; el resto en 20 componentes). Desde "+ nodos" todos
los modelos se entrenan con las filas de Monclova y de los 64 nodos. Los pronósticos salen de las cachés que dejan los
notebooks 05 y 06 y `servidor.py`; un paso que no se probó en un modelo queda vacío y no se inventa.
"""
import pickle
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .metrics import mae, peor_mes

PASOS = ["sin MDA", "con MDA", "+ nodos", "+ exógenas", "+ modelado estadístico", "todas las variables", "PCA"]
RAMAS = ("+ exógenas", "+ modelado estadístico")      # salen de "+ nodos" por separado; "todas las variables" las junta
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/exp_cache"


def cargar_pasos(R):
    """{modelo: {paso: pronóstico en MXN/MWh o None}}. `R` viene de referencias.cargar(). Todo se lee de caché."""
    from . import dl, exp, features2 as f2, multinodo as mn
    from .servidor import GRUPOS
    from .todo import EXTRA_RED, EXTRA_SIN_MDA, K_PC
    cb, cl = R.cb, R.cl
    cbn, cln = f2.con_nodos(cb), f2.con_nodos(cl)
    DIAS = exp.dias_eval(R.c0)
    nombres = dict(zip(["+ nodos", "+ exógenas", "+ modelado estadístico", "todas las variables", "PCA"], GRUPOS))
    Xtb = f2._t(cb)[1]
    inv_s = lambda P: f2.inv(P + Xtb.loc[P.index], cb)
    lear = lambda c, cc, p: exp.pronostico(cc, mn.lear(c, DIAS, grupos=GRUPOS[p], troncal=True, comp="nucleo" if p == "pca" else None, k=K_PC))[0]
    gb = lambda p: inv_s(pickle.loads((CACHE / f"esc_gb_{p}.pkl").read_bytes()))
    def red(variante, hp, nombre, extra, p):
        P = dl.correr_dl_red(cbn, variante, hp, DIAS, troncal=True, nombre=nombre, extra=extra if p else (),
                             comp="nucleo" if p == "pca" else None, k=K_PC if p == "pca" else 20)
        return dl.a_precios(cb, P, variante)
    hp_tide = replace(R.hp["tide"], batch=256, l1=R.hp["tide"].l1 / 10)      # la configuración elegida con 64 nodos (notebook 06)
    hp_nb = replace(R.hp["nbeats"], batch=256)
    tide = lambda p: red("spread", hp_tide, "tide_troncal", EXTRA_RED, p)
    sin = lambda p: red("sin_mda", hp_nb, "nbeats_troncal", EXTRA_SIN_MDA, p)
    pasos = {
        "LEAR": {"sin MDA": R.P["LEAR solo MTR"], "con MDA": R.P["LEAR base"], **{s: lear(cbn, cb, p) for s, p in nombres.items()}},
        "LEAR en proporción": {"con MDA": R.P["LEAR log-razón"], **{s: lear(cln, cl, p) for s, p in nombres.items()}},
        "GB": {"con MDA": R.P["GB base"], **{s: gb(p) for s, p in nombres.items()}},
        "TiDE": {"con MDA": R.P["TiDE V3"], "+ nodos": tide(None), "todas las variables": tide("todas"), "PCA": tide("pca")},
        "red sin MDA": {"sin MDA": R.P["NBEATSx V1 sin MDA"], "+ nodos": sin(None), "todas las variables": sin("todas"), "PCA": sin("pca")},
    }
    return {k: {s: v.get(s) for s in PASOS} for k, v in pasos.items()}


def trayectoria(pasos, Y, X, idx_dev, idx_extra=None):
    """Tabla larga: error de cada paso, su cambio contra el MDA y la peor mes (el criterio de elección), en
    desarrollo y (con `idx_extra`) en el test."""
    f = lambda M, idx: mae(Y.loc[idx].values, M.loc[idx].values)
    filas = []
    for modelo, d in pasos.items():
        for paso in PASOS:
            P = d.get(paso)
            if P is None: continue
            fila = dict(modelo=modelo, paso=paso, MAE_desarrollo=f(P, idx_dev), cambio_desarrollo=100 * (f(P, idx_dev) / f(X, idx_dev) - 1),
                        peor_mes_desarrollo=peor_mes(Y, X, P, idx_dev))
            if idx_extra is not None:
                fila.update(MAE_test=f(P, idx_extra), cambio_test=100 * (f(P, idx_extra) / f(X, idx_extra) - 1), peor_mes_test=peor_mes(Y, X, P, idx_extra))
            filas.append(fila)
    return pd.DataFrame(filas)


def mejor_version(pasos, T, tolerancia=0.5):
    """Para cada modelo, la versión con menor peor mes en desarrollo (sin contar "sin MDA", salvo en la red que no
    lo ve). Si una versión más sencilla (un paso anterior) queda a menos de `tolerancia` puntos de esa, se usa la más
    sencilla: una diferencia así no se distingue del ruido. Son las piezas que se combinan en el notebook 10."""
    out = {}
    for modelo, g in T.groupby("modelo", sort=False):
        if modelo != "red sin MDA":
            g = g[g.paso != "sin MDA"]
        ok = g[g.peor_mes_desarrollo <= g.peor_mes_desarrollo.min() + tolerancia]
        paso = ok.iloc[0].paso                                       # la tabla sigue el orden de PASOS: el primero es el más sencillo
        out[modelo] = (paso, pasos[modelo][paso])
    return out


def figura(T, ps, con_test=False, archivo=None, titulo="Cambio del error frente al MDA en cada paso (%; menos es mejor)"):
    """Un panel por modelo. La línea sigue el camino principal (sin MDA → con MDA → + nodos → todas → PCA); las dos ramas
    (+ exógenas y + modelado estadístico) salen de "+ nodos" con una línea delgada. Con `con_test`, también el test."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    modelos = list(dict.fromkeys(T.modelo))
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4))
    series = [("cambio_desarrollo", ps.AZUL, "-", "o", "desarrollo (jul–dic 2024)")] + ([("cambio_test", ps.NARANJA, "--", "s", "test (ene–jun 2025)")] if con_test else [])
    for ax, k in zip(axs.ravel(), modelos):
        g = T[T.modelo == k].set_index("paso")
        for col, color, ls, mk, _ in series:
            prin = [p for p in PASOS if p in g.index and p not in RAMAS]
            ax.plot([PASOS.index(p) for p in prin], g.loc[prin, col], ls, color=color, lw=1.4, marker=mk, ms=6, zorder=3)
            for r in RAMAS:
                if r in g.index and "+ nodos" in g.index:
                    ax.plot([PASOS.index("+ nodos"), PASOS.index(r)], g.loc[["+ nodos", r], col], ls, color=color, lw=0.7, alpha=0.6)
                    ax.plot(PASOS.index(r), g.loc[r, col], mk, color=color, ms=6, zorder=3)
            for p in g.index:
                ax.annotate(f"{g.loc[p, col]:+.1f}", (PASOS.index(p), g.loc[p, col]), xytext=(0, 7 if col == "cambio_desarrollo" else -13),
                            textcoords="offset points", ha="center", fontsize=7, color=color)
        ax.axhline(0, color=ps.GRIS, lw=0.8); ax.set_title(k, fontsize=11); ax.margins(y=0.3)
        ax.set_xticks(range(len(PASOS)), PASOS, rotation=35, ha="right", fontsize=7); ax.set_xlim(-0.4, len(PASOS) - 0.6)
    for ax in axs.ravel()[len(modelos):]: ax.axis("off")
    if con_test:
        axs.ravel()[-1].legend(handles=[Line2D([], [], color=c, ls=ls, marker=mk, label=lab) for _, c, ls, mk, lab in series], loc="center", fontsize=9, frameon=False)
    fig.suptitle(titulo, fontsize=12, y=0.995); fig.tight_layout()
    if archivo: fig.savefig(archivo, bbox_inches="tight")
    return fig
