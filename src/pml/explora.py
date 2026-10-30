"""Exploración de una fuente exógena contra el MTR, siempre con el mismo formato (notebook 03).

Para cada feature miramos su relación con tres cosas distintas, porque cada una dice algo diferente:
  - el nivel del MTR: si la feature acompaña al precio. El MDA ya lo sabe, así que esto por sí solo no ayuda;
  - la diferencia MTR − MDA: lo que el modelo pronostica. Una relación aquí es señal útil;
  - la magnitud |MTR − MDA|: si la feature avisa de días en que el MDA se equivoca mucho, sin decir hacia dónde.
Todo con días hasta el fin del desarrollo (2024-12-31); el test no se mira aquí.
"""
import numpy as np
import pandas as pd

from . import features2 as f2

PERIODOS = {"antes del desarrollo": (f2.TRAIN_START, pd.Timestamp("2024-06-30")), "desarrollo": (f2.DEV_INI, f2.DEV_FIN)}
EPISODIOS = [("2024-04-01", "2024-06-30", "ola de calor 2024")]


def objetivos(c: f2.Ctx) -> pd.DataFrame:
    """Promedios diarios en precio (MXN/MWh): nivel del MTR, del MDA, diferencia y magnitud; y la diferencia en asinh."""
    Y, X = c.Y.asfreq("D"), c.X.asfreq("D")
    Yt, Xt = f2._t(c)
    return pd.DataFrame({"MTR": Y.mean(axis=1), "MDA": X.mean(axis=1), "MTR − MDA": (Y - X).mean(axis=1),
                         "|MTR − MDA|": (Y - X).abs().mean(axis=1), "MTR − MDA mediana": (Y - X).median(axis=1), "spread asinh": (Yt - Xt).mean(axis=1),
                         "pico": (Y.max(axis=1) > 3000).astype(float)})


def correlaciones(F: pd.DataFrame, O: pd.DataFrame, cols) -> pd.DataFrame:
    """Spearman de cada columna con nivel del MTR, diferencia (asinh) y magnitud, por periodo."""
    out = {}
    for per, (a, b) in PERIODOS.items():
        ix = F.index[(F.index >= a) & (F.index <= b)]
        for nombre, t in (("nivel MTR", "MTR"), ("MTR − MDA", "spread asinh"), ("|MTR − MDA|", "|MTR − MDA|")):
            out[(per, nombre)] = {col: F.loc[ix, col].corr(O.loc[ix, t], method="spearman") for col in cols}
    return pd.DataFrame(out).round(2)


def quintiles(F: pd.DataFrame, O: pd.DataFrame, col) -> pd.DataFrame:
    """Promedios por quintil de la feature (todo el periodo previo al test). La media de MTR − MDA la dominan los días
    con pico; la mediana (de las medianas diarias por hora) muestra el día típico."""
    ix = F.index[(F.index >= f2.TRAIN_START) & (F.index <= f2.DEV_FIN)]
    x = F.loc[ix, col]
    moda = x.mode().iloc[0]
    if x.nunique() <= 8:                                            # variables casi binarias: un grupo por valor
        g = x.round(2).astype(str)
    elif (x == moda).mean() > 0.3:                                  # muchos ceros: un grupo para la moda y cuartiles del resto
        r = x[x != moda]
        g = pd.Series(f"= {moda:.3g}", index=x.index, dtype=object)
        g[r.index] = pd.qcut(r.rank(method="first"), 4, labels=[f"Q{i}" for i in range(1, 5)]).astype(str)
    else:
        g = pd.qcut(x.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
    T = O.loc[ix].groupby(g.values).agg(**{"días": ("MTR", "size"), "MTR medio": ("MTR", "mean"), "MDA medio": ("MDA", "mean"),
                                          "MTR − MDA medio": ("MTR − MDA", "mean"),
                                          "MTR − MDA mediana": ("MTR − MDA mediana", "median"), "|MTR − MDA| medio": ("|MTR − MDA|", "mean"),
                                          "% días con pico": ("pico", "mean")})
    T["rango de la feature"] = x.groupby(g.values).agg(lambda s: f"{s.min():.3g} a {s.max():.3g}")
    T["% días con pico"] *= 100
    return T[["rango de la feature", "días", "MTR medio", "MDA medio", "MTR − MDA medio", "MTR − MDA mediana", "|MTR − MDA| medio", "% días con pico"]].round(1)


def rezagos(F: pd.DataFrame, O: pd.DataFrame, cols, lags=(0, 1, 2, 7)) -> pd.DataFrame:
    """Spearman con la diferencia (asinh) del día d cuando la feature se retrasa k días más (feb–dic 2024)."""
    ix = F.index[(F.index >= f2.TRAIN_START + pd.Timedelta(days=7)) & (F.index <= f2.DEV_FIN)]
    return pd.DataFrame({f"+{k} días": {col: F[col].shift(k).loc[ix].corr(O.loc[ix, "spread asinh"], method="spearman") for col in cols}
                         for k in lags}).round(2)


def panel(F: pd.DataFrame, O: pd.DataFrame, cols, clave, titulo, unidad, archivo=None, plt=None, ps=None):
    """Cuatro paneles: MTR y MDA diarios, la feature clave en el tiempo y, por grupo de la feature, la mediana de la
    diferencia (el día típico) y la magnitud media del error (dominada por los días con pico)."""
    import matplotlib.dates as mdates
    ix = F.index[(F.index >= f2.TRAIN_START) & (F.index <= f2.DEV_FIN)]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.3), gridspec_kw=dict(width_ratios=[1.3, 1.3, 0.8, 0.8]))
    for ax in axs[:2]:
        for a, b, _ in EPISODIOS:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#c3c2b7", alpha=.25, lw=0)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2, 4, 6, 8, 10, 12])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", labelsize=7)
    axs[0].plot(ix, O.loc[ix, "MTR"].clip(upper=3000), color="#0b0b0b", lw=.9, label="MTR")
    axs[0].plot(ix, O.loc[ix, "MDA"].clip(upper=3000), color=ps.NARANJA, lw=.9, label="MDA")
    axs[0].set_title("MTR y MDA, promedio diario\n(MXN/MWh, recortado a 3000)", fontsize=9); axs[0].legend(fontsize=7)
    axs[1].plot(ix, F.loc[ix, clave], color=ps.AZUL, lw=.9); axs[1].set_title(f"{titulo}\n({unidad})", fontsize=9)
    Q = quintiles(F, O, clave)
    x = np.arange(len(Q))
    axs[2].bar(x, Q["MTR − MDA mediana"], width=.6, color=ps.AZUL); axs[2].axhline(0, color="#0b0b0b", lw=.8)
    axs[2].set_title("MTR − MDA, mediana\n(día típico, MXN/MWh)", fontsize=9)
    axs[3].bar(x, Q["|MTR − MDA| medio"], width=.6, color=ps.AQUA)
    axs[3].set_title("|MTR − MDA|, media\n(MXN/MWh)", fontsize=9)
    for ax in axs[2:]:
        ax.set_xticks(x); ax.set_xticklabels(Q.index.astype(str), fontsize=7); ax.set_xlabel(f"grupo de {clave}", fontsize=7)
    fig.tight_layout()
    if archivo is not None:
        fig.savefig(archivo)
    plt.show()
    return Q
