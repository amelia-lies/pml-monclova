"""Los pronósticos ya establecidos en los notebooks 04 y 05 (y algunos de experimentos anteriores), cargados desde las cachés en una sola llamada.

Sirve para que los notebooks posteriores (06 a 11) no repitan decenas de líneas de código para reconstruir lo mismo. No
decide nada nuevo: cada pronóstico se arma con la configuración que eligió su notebook en el periodo de desarrollo.
"""
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import dl, exp, features2 as f2, gb2
from .metrics import mae

ROOT = Path(__file__).resolve().parents[2]
W = 0.01                       # recorte del 1% del objetivo al entrenar LEAR
BASE_G = ["G0", "G2"]          # rezagos del MTR y del MDA
LAM = 0.6                      # encogimiento hacia el MDA (se discute en el notebook 10)
COLS3 = ["mtr_m_std7", "spr_m_std7", "mda_ramp", "mda_sorpresa_h7", "mda_sorpresa_h8", "tx_export_mx_eve_d2", "cdmx_tmax", "cdmx_cdh24",
         "gdl_tmax", "rey_tmax", "hmo_ghi_rev", "juch_viento", "juch_viento_eve"]                       # selección anterior de exógenas (ver ideas descartadas)
COLS4 = ["spr_m_std7", "cdmx_cdh24", "hmo_ghi_rev", "juch_viento", "cngmda_h8", "red_oriente_cng", "an_h1", "an_h5", "an_h18", "an_h21"]  # selección anterior: la red y análogos


@dataclass
class Ref:
    """Los pronósticos establecidos y los contextos con que se calcularon."""
    c0: f2.Ctx
    cb: f2.Ctx                                   # escala asinh × 0.5 (la del proyecto)
    cl: f2.Ctx                                   # objetivo en log-razón
    P: dict = field(default_factory=dict)        # nombre -> pronóstico en MXN/MWh (días × 24)
    hp: dict = field(default_factory=dict)       # configuraciones de las redes elegidas en desarrollo

    @property
    def Y(self): return self.c0.Y

    @property
    def X(self): return self.c0.X

    def mae(self, P, idx=None):
        """MAE de un pronóstico; por omisión en desarrollo."""
        idx = exp.idx_dev() if idx is None else idx
        return mae(self.Y.loc[idx].values, P.loc[idx].values)


def cargar() -> Ref:
    """Reconstruye, desde caché, cada pronóstico de los notebooks 04 y 05 con la configuración que eligió su notebook."""
    c0 = f2.cargar_ctx(); cb = replace(c0, escala=0.5); cl = replace(c0, modo="log")
    DEV, DIAS = exp.idx_dev(), exp.dias_eval(c0)
    r = Ref(c0, cb, cl)
    m = lambda P: r.mae(P)
    corre = lambda nombre, g, cc=cb, **kw: exp.correr_y_evaluar(cc, nombre, g, winsor=W, centro="mediana", **kw)[0]
    # notebooks 04–05
    LEAR = corre("Bm x0.5 winsor 1%", BASE_G)
    r.P.update({"MDA": c0.X, "LEAR base": LEAR, f"LEAR encogido (λ = {LAM})": c0.X + LAM * (LEAR - c0.X)})
    # notebook 05: GB base, configuración elegida en desarrollo
    Xt = f2._t(cb)[1]
    GBP = pickle.loads((ROOT / "data/processed/gb2_cache.pkl").read_bytes())
    gbs = [f2.inv(GBP[("GB base G0+G2", i)] + Xt.loc[GBP[("GB base G0+G2", i)].index], cb) for i in range(len(gb2.GRID))]
    r.P["GB base"] = min(gbs, key=m)
    # notebook 05: redes, configuración elegida por el MAE de V3 en desarrollo
    for a in ("mlp", "nbeats", "tide"):
        sc = [m(dl.a_precios(cb, dl.correr_dl(cb, "spread", h, DEV, nombre=f"busq2_{a}{i}_spread"), "spread")) for i, h in enumerate(dl.busqueda_reg(a))]
        r.hp[a] = replace(dl.busqueda_reg(a)[int(np.argmin(sc))], seeds=4)
    red = lambda a, v, nombre: dl.a_precios(cb, dl.correr_dl(cb, v, r.hp[a], DIAS, nombre=nombre), v)
    r.P.update({"DNN V3": red("mlp", "spread", "mlp_spread_full_r2"), "NBEATSx V3": red("nbeats", "spread", "nbeats_spread_full_r2"),
                "DNN V1 sin MDA": red("mlp", "sin_mda", "mlp_sin_mda_full_r2"), "NBEATSx V1 sin MDA": red("nbeats", "sin_mda", "nbeats_sin_mda_full_r2"),
                "TiDE V3": red("tide", "spread", "tide_spread_full")})
    r.P["TiDE V3 log-razón"] = dl.a_precios(cl, dl.correr_dl(cl, "spread", r.hp["tide"], DIAS, nombre="tide_spread_full_log"), "spread")
    r.P["LEAR solo MTR"] = exp.pronostico(cb, exp.correr_config(cb, "solo G0 nivel", ["G0"], objetivo="mtr", winsor=W, centro="mediana"), "mtr")[0]
    r.P["LEAR log-razón"] = corre("F4 log-razon", BASE_G, cc=cl)
    # selección anterior: LEAR con las exógenas que sobrevivieron (ver ideas descartadas)
    Fx3, _ = f2.construir(cb, ["G4", "G9"]); r.P["LEAR + clima de centros de carga"] = corre("S reducido fase 5", BASE_G, F_extra=Fx3[COLS3])
    Fx4, _ = f2.construir(cb, ["G4", "G9", "G10", "G12"]); r.P["LEAR + red y análogos"] = corre("S reducido fase 6", BASE_G, F_extra=Fx4[COLS4])
    return r
