"""Grupos de variables (G0 a G14). Cada grupo es una función que recibe el contexto de datos y devuelve una fila de
variables por día. Todos respetan lo que se sabe al emitir el pronóstico del día d (en d−1, a mediodía):

  MTR            : sólo fechas <= d−8
  MDA            : fechas <= d
  clima (zona)   : pronóstico `previous_day2` para el día d (emitido >= 48 h antes de cada hora válida) y días previos
  FRED / EIA     : última observación con fecha de disponibilidad <= d − lag (lag=2 por defecto; sensibilidad con 8)
  ERCOT DF       : pronóstico de demanda del día anterior (EIA-930, tipo DF) del propio día d — supuesto documentado
  calendario     : determinista
  clima externo  : Monterrey y Texas, misma regla que el de la zona
  estados operativos de CENACE : sólo hasta d−2 (en d−1 a mediodía aún no se sabe si habrá alerta esa tarde)
  EIA-930 de ERCOT (real)      : días <= d−2 completos (se publica con horas de rezago; supuesto conservador)
  ERCOT precio RT              : hasta d−1 a las 11:00 (se publica cada 15 min)
  clima nacional               : pronóstico previous_day2 y previous_day3
  nodos de la red              : misma regla que Monclova (MTR hasta d−8, MDA hasta d)

`perturbar()` rellena con basura todo lo no disponible; el test de leakage comprueba que la fila de features no cambia.
"""
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dateutil.easter import easter

from .features import asinh_t, feriados_mx

LAG_MTR = 8
PISO_LOG = 30.0                      # MXN/MWh; el MDA llega a 0 o negativo en <0.1% de las horas
HORAS = list(range(1, 25))
EVE = [19, 20, 21, 22]
MED = [12, 13, 14, 15]
LAT, LON = 26.92, -101.44            # zona Monclova–Frontera (promedio de las cabeceras)


@dataclass
class Ctx:
    """Contexto de datos: los precios de Monclova y todas las fuentes externas, en matrices días × 24 o series diarias, más
    la transformación del precio (mediana, MAD, modo y escala) y los días para los que se construyen variables."""
    Y: pd.DataFrame            # MTR zona, días × 24 (historia previa a `dias`)
    X: pd.DataFrame            # MDA zona
    W: dict                    # variable -> días × 24 (pronóstico previous_day2)
    fred: dict                 # nombre -> Series diaria
    inv: pd.Series             # inventarios de gas semanales (índice = fin de semana medida)
    ercot: pd.DataFrame        # pronóstico día anterior de demanda ERCOT, días × 24
    med: float
    mad: float
    dias: pd.DatetimeIndex
    lag_precios: int = 2
    modo: str = "asinh"          # 'asinh' | 'lineal' (sólo estandarización robusta) | 'log' (log(precio / mediana), piso 30)
    escala: float = 1.0           # factor sobre la MAD dentro del asinh
    Wext: dict = field(default_factory=dict)   # punto -> {variable -> días × 24}: Monterrey, San Antonio, Midland
    eo: pd.Series = None          # estados operativos del SIN por fecha (1 alerta, 2 emergencia)
    Wnac: dict = field(default_factory=dict)   # dia_previo -> punto -> {variable -> días × 24}: centros de carga y renovables
    e930: dict = field(default_factory=dict)   # EIA-930 de ERCOT: demanda, eólica, solar, gas, intercambio con México (días × 24)
    rt: pd.DataFrame = None       # precio RT de ERCOT por hora (índice horario, hora de México), hubs y zona sur
    comp: dict = field(default_factory=dict)   # componentes de Monclova: {"MTR"/"MDA": {"ene"/"per"/"cng": días × 24}}
    red: dict = field(default_factory=dict)    # congestión por corredor de la red: {"MTR"/"MDA": {corredor: días × 24}}
    Wi: dict = field(default_factory=dict)     # índice nacional ponderado por consumo: {dia_previo: {variable: días × 24}}
    nodos: dict = field(default_factory=dict)  # precios de cada nodo de la red: {"MTR"/"MDA": {nodo: días × 24}} (G14)


def fw(x, c: Ctx):
    """Transformación de precios (a espacio de modelado)."""
    if c.modo == "log":                                            # el spread se vuelve log(MTR / MDA)
        return np.log(np.maximum(x, PISO_LOG) / c.med)
    z = (x - c.med) / c.mad
    return z if c.modo == "lineal" else np.arcsinh(z / c.escala)


def inv(z, c: Ctx):
    """Inversa de `fw`."""
    if c.modo == "log":
        return np.exp(z) * c.med
    return z * c.mad + c.med if c.modo == "lineal" else np.sinh(z) * c.escala * c.mad + c.med


def _t(c: Ctx):
    """(MTR, MDA) en la escala del proyecto, con un renglón por día del calendario."""
    return fw(c.Y.asfreq("D"), c), fw(c.X.asfreq("D"), c)


def _sh(M: pd.DataFrame, k: int, pref: str) -> pd.DataFrame:
    """La matriz días × 24 desplazada k días, con columnas `{pref}_h{hora}`."""
    s = M.shift(k)
    s.columns = [f"{pref}_h{h}" for h in M.columns]
    return s


def _ix(df, c):
    """Sólo los días pedidos en el contexto."""
    return df.loc[c.dias]


# ---------------------------------------------------------------------------------------- MTR / MDA (lags)
def g0(c):
    """G0: MTR de la misma hora en d−8, d−9, d−10 y d−14 (los más recientes ya publicados)."""
    Yt, _ = _t(c)
    return _ix(pd.concat([_sh(Yt, k, f"mtr_l{k}") for k in (8, 9, 10, 14)], axis=1), c)


def g0w(c):
    """Ventana semanal completa de MTR (d−8…d−14 = 168 h), estilo NBEATSx. Reemplaza a G0."""
    Yt, _ = _t(c)
    return _ix(pd.concat([_sh(Yt, k, f"mtr_l{k}") for k in range(8, 15)], axis=1), c)


def g0dow(c):
    """Mismo día de la semana más atrás: MTR en d−21 y d−28 (se agrega a G0)."""
    Yt, _ = _t(c)
    return _ix(pd.concat([_sh(Yt, k, f"mtr_l{k}") for k in (21, 28)], axis=1), c)


def g2(c):
    """G2: MDA de la misma hora en d, d−1 y d−7, y el error del MDA (MTR − MDA) en d−8 y d−14."""
    Yt, Xt = _t(c)
    spr = Yt - Xt
    return _ix(pd.concat([_sh(Xt, 0, "mda_l0"), _sh(Xt, 1, "mda_l1"), _sh(Xt, 7, "mda_l7"),
                          _sh(spr, 8, "spr_l8"), _sh(spr, 14, "spr_l14")], axis=1), c)


def g2w(c):
    """Ventana semanal completa: MDA d…d−7 (192 h) y spread d−8…d−14 (168 h). Reemplaza a G2."""
    Yt, Xt = _t(c)
    spr = Yt - Xt
    return _ix(pd.concat([_sh(Xt, k, f"mda_l{k}") for k in range(0, 8)] + [_sh(spr, k, f"spr_l{k}") for k in range(8, 15)], axis=1), c)


# ---------------------------------------------------------------------------------------- estadísticas móviles
def g4(c):
    """G4: estadísticas móviles del MTR y de su error en 1, 4 y 12 semanas publicadas, y la forma del MDA de mañana
    (nivel, máximo, rampa de la tarde y sorpresa frente al mismo día de las 4 semanas anteriores)."""
    Yt, Xt = _t(c)
    spr = Yt - Xt
    daily = pd.DataFrame({"mtr_m": Yt.mean(axis=1), "mtr_x": Yt.max(axis=1), "spr_m": spr.mean(axis=1),
                          "spr_a": spr.abs().mean(axis=1), "up": (c.Y.asfreq("D") > 2 * c.X.asfreq("D")).mean(axis=1)})
    base = daily.shift(LAG_MTR)                                   # sólo MTR conocido (<= d−8)
    cols = {}
    for w in (7, 28, 84):
        r = base.rolling(w, min_periods=7)
        cols[f"mtr_m_mean{w}"] = r["mtr_m"].mean(); cols[f"mtr_m_med{w}"] = r["mtr_m"].median(); cols[f"mtr_m_std{w}"] = r["mtr_m"].std()
        cols[f"spr_m_mean{w}"] = r["spr_m"].mean(); cols[f"spr_m_std{w}"] = r["spr_m"].std()
    for w in (7, 28):
        r = base.rolling(w, min_periods=7)
        cols[f"mtr_x_max{w}"] = r["mtr_x"].max(); cols[f"spr_a_mean{w}"] = r["spr_a"].mean(); cols[f"up_mean{w}"] = r["up"].mean()
    mda_m = Xt.mean(axis=1)                                      # MDA conocido hasta d
    cols["mda_m"] = mda_m
    cols["mda_x"] = Xt.max(axis=1)
    cols["mda_ramp"] = Xt[EVE].mean(axis=1) - Xt[MED].mean(axis=1)
    cols["mda_nivel28"] = mda_m - mda_m.shift(1).rolling(28, min_periods=7).mean()
    cols["mda_vol28"] = mda_m.shift(1).rolling(28, min_periods=7).std()
    cols["mda_vs_mtr7"] = mda_m - base["mtr_m"].rolling(7, min_periods=7).mean()
    out = pd.DataFrame(cols)
    med4 = pd.concat([Xt.shift(k) for k in (7, 14, 21, 28)]).groupby(level=0).median()   # mismo día de semana, 4 semanas
    sorp = (Xt - med4)
    sorp.columns = [f"mda_sorpresa_h{h}" for h in Xt.columns]
    return _ix(pd.concat([out, sorp], axis=1), c)


# ---------------------------------------------------------------------------------------- calendario
def _fourier(n, periodo, k, pref):
    """Términos seno y coseno de periodo `periodo` hasta el armónico k."""
    return {f"{pref}_{f}{i}": getattr(np, f)(2 * np.pi * i * n / periodo) for i in range(1, k + 1) for f in ("sin", "cos")}


def g1(c):
    """G1: calendario básico (día de la semana, feriado y estación del año con Fourier)."""
    d = c.dias
    out = pd.get_dummies(d.dayofweek).reindex(columns=range(7), fill_value=0).astype(float)
    out.index = d; out.columns = [f"dow{i}" for i in range(7)]
    doy = d.dayofyear.values / 365.25
    for k in (1, 2):
        out[f"sin{k}"] = np.sin(2 * np.pi * k * doy); out[f"cos{k}"] = np.cos(2 * np.pi * k * doy)
    out["feriado"] = feriados_mx(d).values.astype(float)
    return out


def g1f(c):
    """Alternativa compacta a las dummies de día de la semana: Fourier semanal (k=1..3) + anual (k=1,2) + feriado."""
    d = c.dias
    out = pd.DataFrame(_fourier(d.dayofweek.values, 7, 3, "wk"), index=d)
    doy = d.dayofyear.values / 365.25
    for k in (1, 2):
        out[f"sin{k}"] = np.sin(2 * np.pi * k * doy); out[f"cos{k}"] = np.cos(2 * np.pi * k * doy)
    out["feriado"] = feriados_mx(d).values.astype(float)
    return out


def _luz(d: pd.DatetimeIndex):
    """Amanecer, atardecer (hora reloj CST) y duración del día; aproximación astronómica sin ecuación del tiempo."""
    n = d.dayofyear.values
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + n) / 365)
    ha = np.arccos(np.clip(-np.tan(np.radians(LAT)) * np.tan(decl), -1, 1))
    dur = 2 * np.degrees(ha) / 15
    noon = 12 + (-90 - LON) / 15                                  # mediodía solar en hora estándar de UTC−6
    return noon - dur / 2, noon + dur / 2, dur


def g7(c):
    """Calendario ampliado: mes, víspera/post-festivo, puentes, Semana Santa, Navidad, vacaciones, geometría solar."""
    d = c.dias
    fer = feriados_mx(pd.date_range(d.min() - pd.Timedelta(days=7), d.max() + pd.Timedelta(days=7)))
    es = lambda x: fer.reindex(x).fillna(0).values.astype(float)
    out = pd.get_dummies(d.month).reindex(columns=range(1, 13), fill_value=0).astype(float)
    out.index = d; out.columns = [f"mes{m}" for m in range(1, 13)]
    out["vispera"] = es(d + pd.Timedelta(days=1)); out["post_feriado"] = es(d - pd.Timedelta(days=1))
    dow = d.dayofweek
    out["puente"] = (((dow == 4) & (es(d - pd.Timedelta(days=1)) == 1)) | ((dow == 0) & (es(d + pd.Timedelta(days=1)) == 1))).astype(float)
    pas = {y: pd.Timestamp(easter(y)) for y in range(d.year.min(), d.year.max() + 1)}
    e = pd.DatetimeIndex([pas[y] for y in d.year])
    out["semana_santa"] = ((d >= e - pd.Timedelta(days=7)) & (d <= e)).astype(float)
    out["pascua"] = ((d > e) & (d <= e + pd.Timedelta(days=7))).astype(float)
    out["navidad"] = (((d.month == 12) & (d.day >= 24)) | ((d.month == 1) & (d.day <= 6))).astype(float)
    out["vac_verano"] = (((d.month == 7) & (d.day >= 15)) | (d.month == 8) & (d.day <= 25)).astype(float)   # aprox. calendario SEP
    out["amanecer"], out["atardecer"], out["duracion_dia"] = _luz(d)
    return out


# Vacaciones escolares SEP (educación básica, calendario nacional que también aplica en Coahuila). Fechas de los
# calendarios oficiales 2023-24 y 2024-25 publicados por la SEP (p. ej. Expansión, 10-jun-2024). Son los días sin clases.
VAC_SEP = [("2023-12-18", "2024-01-05"), ("2024-03-25", "2024-04-05"), ("2024-07-17", "2024-08-25"),
           ("2024-12-19", "2025-01-07"), ("2025-04-14", "2025-04-25")]


def indice_vacaciones(d: pd.DatetimeIndex) -> pd.DataFrame:
    """Qué tan "de vacaciones" está la gente el día d, de 0 a 1. Se toma el máximo de varias causas, con pesos a juicio:
    vacaciones escolares 0.5 (afectan a las familias, no a la industria), jueves y viernes santo 1.0 (para casi todo el
    mundo), 24 dic–1 ene 0.9, feriado oficial 0.8, puente 0.7, víspera de feriado 0.3. El fin de semana va aparte."""
    fer = feriados_mx(pd.date_range(d.min() - pd.Timedelta(days=3), d.max() + pd.Timedelta(days=3)))
    es = lambda x: fer.reindex(x).fillna(0).values.astype(float)
    sep = np.zeros(len(d))
    for a, b in VAC_SEP:
        sep[(d >= a) & (d <= b)] = 1.0
    e = pd.DatetimeIndex([pd.Timestamp(easter(y)) for y in d.year])
    santos = ((d == e - pd.Timedelta(days=3)) | (d == e - pd.Timedelta(days=2))).astype(float)
    fin_ano = (((d.month == 12) & (d.day >= 24)) | ((d.month == 1) & (d.day == 1))).astype(float)
    dow = d.dayofweek
    puente = (((dow == 4) & (es(d - pd.Timedelta(days=1)) == 1)) | ((dow == 0) & (es(d + pd.Timedelta(days=1)) == 1))).astype(float)
    comp = np.vstack([0.5 * sep, 1.0 * santos, 0.9 * fin_ano, 0.8 * es(d), 0.7 * puente, 0.3 * es(d + pd.Timedelta(days=1))])
    out = pd.DataFrame({"vac_sep": sep, "vac_indice": comp.max(axis=0)}, index=d)
    out["vac_finde"] = out["vac_indice"] * (dow >= 5)
    return out


def g7v(c):
    """Índice de vacaciones: corrige la Semana Santa y el verano aproximados de G7 con las fechas reales SEP."""
    return indice_vacaciones(c.dias)


# ---------------------------------------------------------------------------------------- clima de la zona (pronóstico)
def g5(c):
    """G5: clima pronosticado para la zona de Monclova (calor, nubes, radiación, viento, presión) para el día d."""
    W = {k: v.asfreq("D") for k, v in c.W.items()}
    T, AT, RH, DP, CC, GHI, DNI, WS, PR, SP = (W[k] for k in ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
                                                              "dew_point_2m", "cloud_cover", "shortwave_radiation",
                                                              "direct_normal_irradiance", "wind_speed_10m", "precipitation",
                                                              "surface_pressure"])
    cols = {}
    cols["cdh24"] = (T - 24).clip(lower=0).sum(axis=1); cols["cdh28"] = (T - 28).clip(lower=0).sum(axis=1)
    cols["hdh15"] = (15 - T).clip(lower=0).sum(axis=1)
    cols["tmax"], cols["tmin"], cols["tmean"] = T.max(axis=1), T.min(axis=1), T.mean(axis=1)
    cols["trango"] = cols["tmax"] - cols["tmin"]; cols["hora_tmax"] = T.values.argmax(axis=1) + 1
    cols["t_eve"] = T[EVE].mean(axis=1); cols["at_max"] = AT.max(axis=1)
    cols["rh_mean"], cols["dp_mean"] = RH.mean(axis=1), DP.mean(axis=1)
    cols["nub_mean"], cols["nub_tarde"] = CC.mean(axis=1), CC[[12, 13, 14, 15, 16, 17]].mean(axis=1)
    cols["ghi_sum"], cols["dni_sum"] = GHI.sum(axis=1), DNI.sum(axis=1)
    cols["ghi_caida_tarde"] = GHI[[15, 16, 17]].mean(axis=1) - GHI[[18, 19]].mean(axis=1)      # rampa solar vespertina
    cols["viento_mean"], cols["viento_eve"] = WS.mean(axis=1), WS[EVE].mean(axis=1)
    cols["lluvia"] = PR.sum(axis=1); cols["presion"] = SP.mean(axis=1)
    cols["presion_d1"] = cols["presion"] - cols["presion"].shift(1)                # ambos son pronósticos ya emitidos
    cols["tmax_anom28"] = cols["tmax"] - cols["tmax"].shift(1).rolling(28, min_periods=14).mean()
    cols["cdh24_anom28"] = cols["cdh24"] - cols["cdh24"].shift(1).rolling(28, min_periods=14).mean()
    out = pd.DataFrame(cols)
    horarias = pd.concat([_sh(T, 0, "temp"), _sh(GHI, 0, "ghi")], axis=1)
    return _ix(pd.concat([out, horarias], axis=1), c)


# ---------------------------------------------------------------------------------------- combustibles y FX
def _diaria(s: pd.Series, hasta) -> pd.Series:
    """Serie llevada a diario hasta la fecha `hasta`, repitiendo la última observación en los días sin dato (fines de
    semana, feriados). El rezago de publicación se aplica después, con shift."""
    return s.reindex(pd.date_range(s.index.min(), hasta)).ffill()


def g6(c):
    """G6: combustibles y tipo de cambio (FRED), con el rezago de publicación del contexto."""
    L = c.lag_precios
    hasta = c.dias.max()
    cols, niv = {}, {}
    for nom in ("hh_spot", "wti", "brent", "usdmxn", "diesel_golfo"):
        sd = _diaria(c.fred[nom], hasta)
        niv[nom] = sd.shift(L)
        cols[nom] = sd.shift(L)
        cols[f"{nom}_d7"] = sd.shift(L) - sd.shift(L + 7)
        cols[f"{nom}_vol28"] = np.log(sd).diff().shift(L).rolling(28, min_periods=14).std()
    cols["hh_mxn"] = niv["hh_spot"] * niv["usdmxn"]
    return _ix(pd.DataFrame(cols), c)


def g6g(c):
    """Inventarios semanales de gas (EIA): el dato de la semana que termina el viernes se publica el jueves siguiente."""
    L, hasta = c.lag_precios, c.dias.max()
    pub = pd.Series(c.inv.values, index=c.inv.index + pd.Timedelta(days=6))
    d1 = pd.Series(c.inv.diff().values, index=pub.index)
    d4 = pd.Series((c.inv - c.inv.shift(4)).values, index=pub.index)
    cols = {"inv_gas": _diaria(pub, hasta).shift(L), "inv_gas_sem": _diaria(d1.dropna(), hasta).shift(L),
            "inv_gas_4sem": _diaria(d4.dropna(), hasta).shift(L)}
    return _ix(pd.DataFrame(cols), c)


def g6e(c):
    """Pronóstico de demanda del día anterior de ERCOT (EIA-930 tipo DF) para el día d. Supuesto: disponible en d−1."""
    DF = c.ercot.asfreq("D") / 1000.0                              # GW
    m = DF.mean(axis=1)
    out = pd.DataFrame({"ercot_df_mean": m, "ercot_df_max": DF.max(axis=1), "ercot_df_eve": DF[EVE].mean(axis=1),
                        "ercot_df_rel28": m - m.shift(1).rolling(28, min_periods=14).mean()})
    return _ix(out, c)


def g8(c):
    """Estrés del sistema: calor en Monterrey (la carga grande del Noreste), calor y heladas en Texas (demanda
    ERCOT y congelamiento de pozos de gas) y estados operativos recientes de CENACE."""
    W = {p: {k: v.asfreq("D") for k, v in w.items()} for p, w in c.Wext.items()}
    T = {p: W[p]["temperature_2m"] for p in W}
    cols = {}
    t = T["Monterrey"]
    cols["mty_tmax"] = t.max(axis=1); cols["mty_t_eve"] = t[EVE].mean(axis=1); cols["mty_cdh24"] = (t - 24).clip(lower=0).sum(axis=1)
    cols["mty_tmax_anom28"] = cols["mty_tmax"] - cols["mty_tmax"].shift(1).rolling(28, min_periods=14).mean()
    t = T["SanAntonio"]
    cols["sa_tmax"] = t.max(axis=1); cols["sa_cdh24"] = (t - 24).clip(lower=0).sum(axis=1)
    tmin_tx = pd.concat([T["SanAntonio"].min(axis=1), T["Midland"].min(axis=1)], axis=1).min(axis=1)
    cols["tx_tmin"] = tmin_tx
    cols["tx_helada"] = (tmin_tx < 0).astype(float)
    cols["tx_hdh0"] = sum((-T[p]).clip(lower=0).sum(axis=1) for p in ("SanAntonio", "Midland"))   # grados-hora bajo 0 °C
    cols["tx_helada_3d"] = cols["tx_helada"].rolling(3, min_periods=1).max()                   # pronósticos ya emitidos
    eo = (c.eo if c.eo is not None else pd.Series(dtype=float)).reindex(pd.date_range(c.Y.index.min(), c.dias.max())).fillna(0)
    cols["eo_d2"] = eo.shift(2)
    cols["eo_7d"] = (eo > 0).astype(float).shift(2).rolling(7, min_periods=1).sum()
    return _ix(pd.DataFrame(cols), c)


def g9(c):
    """Información pública del día anterior: operación real de ERCOT (EIA-930), precio en tiempo real de ERCOT y
    clima de los grandes centros de carga y de las zonas renovables del SIN, con la revisión del pronóstico en un día."""
    rango = pd.date_range(c.Y.index.min(), c.dias.max())
    cols = {}
    E = {k: v.reindex(rango) for k, v in c.e930.items()}
    if E:
        dem, df = E["demanda"], c.ercot.reindex(rango)
        cols["tx_dem_max_d2"] = (dem.max(axis=1) / 1000).shift(2)
        cols["tx_dem_sorpresa_d2"] = ((dem - df).mean(axis=1) / 1000).shift(2)                 # real − pronóstico del día anterior
        eol = E["eolica"].mean(axis=1) / 1000
        cols["tx_eolica_d2"] = eol.shift(2); cols["tx_eolica_anom_d2"] = (eol - eol.rolling(28, min_periods=7).mean()).shift(2)
        cols["tx_solar_d2"] = (E["solar"].mean(axis=1) / 1000).shift(2)
        cols["tx_gas_share_d2"] = (E["gas"].sum(axis=1) / dem.sum(axis=1)).shift(2)
        imx = E["intercambio_mx"]
        cols["tx_export_mx_d2"] = imx.mean(axis=1).shift(2); cols["tx_export_mx_eve_d2"] = imx[EVE].mean(axis=1).shift(2)
        cols["tx_export_mx_anom_d2"] = (imx.mean(axis=1) - imx.mean(axis=1).rolling(28, min_periods=7).mean()).shift(2)
    if c.rt is not None:
        z = np.arcsinh(c.rt / 50.0)                                                             # precios con colas enormes
        dia = z.index.normalize()
        g = z.groupby(dia)
        cols["tx_rt_hub_d2"] = g["HB_HUBAVG"].mean().reindex(rango).shift(2)
        cols["tx_rt_hub_max_d2"] = g["HB_HUBAVG"].max().reindex(rango).shift(2)
        cols["tx_rt_west_d2"] = g["HB_WEST"].mean().reindex(rango).shift(2)
        cols["tx_rt_south_d2"] = g["LZ_SOUTH"].mean().reindex(rango).shift(2)
        man = z[(z.index.hour < 11)].groupby(dia[z.index.hour < 11])["HB_HUBAVG"].mean().reindex(rango)
        cols["tx_rt_hub_manana_d1"] = man.shift(1)                                              # d−1 de 00:00 a 11:00
    if c.Wnac:
        W2, W3 = c.Wnac[2], c.Wnac[3]
        T = lambda W, p: W[p]["temperature_2m"].reindex(rango)
        for p, n in (("CDMX", "cdmx"), ("Guadalajara", "gdl"), ("Hermosillo", "hmo"), ("Reynosa", "rey")):
            cols[f"{n}_tmax"] = T(W2, p).max(axis=1); cols[f"{n}_cdh24"] = (T(W2, p) - 24).clip(lower=0).sum(axis=1)
        cols["nac_tmax"] = sum(cols[f"{n}_tmax"] for n in ("cdmx", "gdl", "hmo", "rey")) / 4
        cols["nac_tmax_rev"] = cols["nac_tmax"] - sum(T(W3, p).max(axis=1) for p in ("CDMX", "Guadalajara", "Hermosillo", "Reynosa")) / 4
        ghi = lambda W: W["Hermosillo"]["shortwave_radiation"].reindex(rango)
        cols["hmo_ghi"] = ghi(W2).sum(axis=1); cols["hmo_ghi_rev"] = cols["hmo_ghi"] - ghi(W3).sum(axis=1)
        cols["hmo_ghi_tarde"] = ghi(W2)[[16, 17, 18]].mean(axis=1)
        for p, n in (("Juchitan", "juch"), ("Reynosa", "rey")):
            ws2, ws3 = W2[p]["wind_speed_10m"].reindex(rango), W3[p]["wind_speed_10m"].reindex(rango)
            cols[f"{n}_viento"] = ws2.mean(axis=1); cols[f"{n}_viento_eve"] = ws2[EVE].mean(axis=1)
            cols[f"{n}_viento_rev"] = ws2.mean(axis=1) - ws3.mean(axis=1)
    return _ix(pd.DataFrame(cols, index=rango), c)


# ---------------------------------------------------------------------------------------- red, clima nacional como serie, análogos
def g10(c):
    """Congestión. La componente de energía es igual en todo el SIN; lo local es la congestión. El MDA de d trae su propia
    congestión (conocida al emitir) y el MTR tiende a encogerla; los corredores dicen hacia dónde fluye la potencia."""
    rango = pd.date_range(c.Y.index.min(), c.dias.max())
    lag = LAG_MTR
    cols = {}
    if c.comp:
        cm = c.comp["MDA"]["cng"].reindex(rango)
        cols.update({f"cngmda_h{h}": cm[h] for h in HORAS})
        cols["cngmda_dia"], cols["cngmda_eve"] = cm.mean(axis=1), cm[EVE].mean(axis=1)
        err = (c.comp["MTR"]["cng"] - c.comp["MDA"]["cng"]).reindex(rango)
        cols["cng_err7_d8"] = err.mean(axis=1).shift(lag).rolling(7, min_periods=3).mean()
        cols["cng_err7_eve_d8"] = err[EVE].mean(axis=1).shift(lag).rolling(7, min_periods=3).mean()
        cols["cng_encoge28_d8"] = (c.comp["MTR"]["cng"].abs().sum(axis=1) / c.comp["MDA"]["cng"].abs().sum(axis=1).clip(lower=1)).reindex(rango).shift(lag).rolling(28, min_periods=7).median()
    if c.red:
        for g, M in c.red["MDA"].items():
            M = M.reindex(rango)
            cols[f"red_{g}_cng"], cols[f"red_{g}_cng_eve"] = M.mean(axis=1), M[EVE].mean(axis=1)
            e = (c.red["MTR"][g] - c.red["MDA"][g]).reindex(rango).mean(axis=1)
            cols[f"red_{g}_err7_d8"] = e.shift(lag).rolling(7, min_periods=3).mean()
        norte = sum(c.red["MDA"][g].reindex(rango) for g in ("carbon", "saltillo", "monterrey")) / 3
        cols["red_grad_norte_centro"] = (norte - c.red["MDA"]["centro"].reindex(rango)).mean(axis=1)
        cols["red_grad_norte_centro_eve"] = (norte - c.red["MDA"]["centro"].reindex(rango))[EVE].mean(axis=1)
    return _ix(pd.DataFrame(cols, index=rango), c)


def _racha(b: pd.Series) -> pd.Series:
    """Días consecutivos (terminando en cada fecha) en que b es verdadero."""
    g = (~b).cumsum()
    return b.astype(int).groupby(g).cumsum()


def g11(c):
    """El clima nacional como serie, no como foto del día: índice ponderado por consumo (CFE 2024), calor acumulado en
    varios días, la noche, lluvia y nubosidad, la trayectoria de las revisiones del pronóstico (corridas con 4, 3 y 2 días
    de anticipación) y el error reciente del pronóstico contra lo observado (hasta d−2)."""
    if not c.Wi:
        return _ix(pd.DataFrame(index=c.dias), c)
    rango = pd.date_range(c.Y.index.min(), c.dias.max())
    T = {k: w["temperature_2m"].reindex(rango) for k, w in c.Wi.items()}
    W2 = {v: M.reindex(rango) for v, M in c.Wi[2].items()}
    cdh = {k: (t - 24).clip(lower=0).sum(axis=1) for k, t in T.items()}
    cols = {}
    t = T[2]
    cols["inac_tmean"], cols["inac_tmax"], cols["inac_t_eve"] = t.mean(axis=1), t.max(axis=1), t[EVE].mean(axis=1)
    cols["inac_t_noche"] = t[[1, 2, 3, 4, 5, 6]].mean(axis=1)                       # si la noche no refresca, la demanda no baja
    cols["inac_cdh24"] = cdh[2]
    cols["inac_cdh_3d"] = cdh[2] + cdh[2].shift(1) + cdh[2].shift(2)                # pronósticos emitidos >= 48 h antes de cada día
    cols["inac_cdh_anom28"] = cdh[2] - cdh[2].shift(1).rolling(28, min_periods=14).mean()
    cols["inac_racha_calor"] = _racha(cdh[2] > cdh[2].shift(1).rolling(28, min_periods=14).median())
    cols["inac_lluvia"] = W2["precipitation"].sum(axis=1)
    cols["inac_nub"], cols["inac_nub_tarde"] = W2["cloud_cover"].mean(axis=1), W2["cloud_cover"][[12, 13, 14, 15, 16, 17]].mean(axis=1)
    cols["inac_ghi"] = W2["shortwave_radiation"].sum(axis=1)
    if 3 in T and 4 in T:
        cols["inac_rev1"] = t.mean(axis=1) - T[3].mean(axis=1)
        cols["inac_rev2"] = T[3].mean(axis=1) - T[4].mean(axis=1)
        cols["inac_rev_cdh"] = cdh[2] - cdh[4]
    if 0 in T:
        err = T[0].mean(axis=1) - t.mean(axis=1)                                      # observado − pronóstico a 48 h
        cols["inac_err_d2"] = err.shift(2)
        cols["inac_err7_d2"] = err.shift(2).rolling(7, min_periods=3).mean()
        cols["inac_cdh_err7_d2"] = (cdh[0] - cdh[2]).shift(2).rolling(7, min_periods=3).mean()
    return _ix(pd.DataFrame(cols, index=rango), c)


K_ANALOGOS = 10


def g12(c):
    """Días análogos: para cada d, los K días con MTR ya publicado (<= d−8) cuyo MDA y calor nacional se parecen más al
    de d. Variables: mediana por hora de su MTR − MDA (escala transformada), su frecuencia de picos y qué tan parecidos son."""
    Yt, Xt = _t(c)
    S = (Yt - Xt)
    pico = (c.Y.asfreq("D").max(axis=1) > 3000).astype(float)
    Z = Xt.copy()
    Z = (Z - Z.mean(axis=1).values[:, None]) * 1.0
    Z["nivel"] = Xt.mean(axis=1) * 2                                                  # nivel del día pesa como dos horas
    if c.Wi:
        t = c.Wi[2]["temperature_2m"].reindex(Z.index)
        tmax = t.max(axis=1)
        Z["calor"] = (tmax - 28) / 3 * 2                                              # escala fija: sin estadísticas que vean el futuro
    Z = Z.dropna()
    filas = {}
    for d in c.dias:
        if d not in Z.index:
            continue
        cand = Z.index[(Z.index <= d - pd.Timedelta(days=LAG_MTR)) & Z.index.isin(S.dropna().index)]
        if len(cand) < K_ANALOGOS:
            continue
        dist = np.sqrt(((Z.loc[cand].values - Z.loc[d].values) ** 2).sum(axis=1))
        o = np.argsort(dist)[:K_ANALOGOS]
        vec = np.median(S.loc[cand[o]].values, axis=0)
        fila = {f"an_h{h}": vec[i] for i, h in enumerate(S.columns)}
        fila["an_dia"], fila["an_pico"], fila["an_dist"] = vec.mean(), pico.reindex(cand[o]).mean(), dist[o].mean()
        filas[d] = fila
    return pd.DataFrame.from_dict(filas, orient="index").reindex(c.dias)


# ---------------------------------------------------------------------------------------- varias escalas (último intento)
def _corr_filas(A: pd.DataFrame, B: pd.DataFrame) -> pd.Series:
    """Correlación, día por día, entre el perfil de 24 horas de A y el de B (¿tienen la misma forma?)."""
    a = A.sub(A.mean(axis=1), axis=0); b = B.sub(B.mean(axis=1), axis=0)
    return (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))


def _vecinos(V: np.ndarray, conocido: np.ndarray, objetivo: np.ndarray, dias_ix, k=5):
    """Para cada día pedido, los k días ya conocidos (<= d−8) cuyo vector V se parece más al de ese día.
    Devuelve (distancia media a esos vecinos, promedio de `objetivo` en ellos). V: un vector por fecha (filas)."""
    dist_m, obj_m = np.full(len(dias_ix), np.nan), np.full(len(dias_ix), np.nan)
    ok = ~np.isnan(V).any(axis=1)
    for j, i in enumerate(dias_ix):
        if i < 0 or not ok[i]:
            continue
        cand = np.where(ok[:max(i - LAG_MTR + 1, 0)] & conocido[:max(i - LAG_MTR + 1, 0)])[0]
        if len(cand) < k:
            continue
        d = np.sqrt(((V[cand] - V[i]) ** 2).mean(axis=1))
        o = np.argsort(d)[:k]
        dist_m[j], obj_m[j] = d[o].mean(), objetivo[cand[o]].mean()
    return dist_m, obj_m


def g13(c):
    """Patrones de varias escalas y de largo alcance (último intento de variables, notebook 07).

    Las estadísticas móviles del G4 fueron de lo más útil para el GB. Aquí resumimos la historia de otras formas:
      1. medias exponenciales del error del MDA a cinco escalas (vida media de 3 a 56 días) y la pendiente entre escalas;
      2. las colas recientes del error (cuantiles 10 y 90 a 28 y 84 días);
      3. persistencia y acoplamiento: autocorrelación del error, correlación MTR–MDA y si el perfil horario del MTR
         sigue la forma del MDA;
      4. dónde está el MDA de hoy respecto a su propia historia, a cuatro escalas (z de 7, 28, 84 y 364 días);
      5. el día contra el resto: parecido del perfil de hoy con el de ayer y la semana pasada, qué tan raro es el día
         (distancia a sus días más parecidos) y qué error tuvo el MDA en esos días, a escala de un día y de una semana;
      6. el error del MDA por hora, suavizado a dos escalas.
    Todo lo que usa MTR mira sólo días <= d−8; lo que usa MDA, días <= d. Nada se estandariza con el periodo completo."""
    Yt, Xt = _t(c)
    spr = Yt - Xt
    s = spr.mean(axis=1).shift(LAG_MTR)                   # error diario del MDA, ya publicado
    a = spr.abs().mean(axis=1).shift(LAG_MTR)             # su tamaño
    x = Xt.mean(axis=1)                                   # nivel del MDA (conocido hasta d)
    cols = {}
    # 1. varias escalas
    for hl in (3, 7, 14, 28, 56):
        cols[f"spr_ewm{hl}"] = s.ewm(halflife=hl, min_periods=7).mean()
        cols[f"abs_ewm{hl}"] = a.ewm(halflife=hl, min_periods=7).mean()
    cols["spr_tend_corta"] = cols["spr_ewm3"] - cols["spr_ewm14"]
    cols["spr_tend_larga"] = cols["spr_ewm14"] - cols["spr_ewm56"]
    # 2. colas recientes
    for w in (28, 84):
        r = s.rolling(w, min_periods=14)
        cols[f"spr_q10_{w}"], cols[f"spr_q90_{w}"] = r.quantile(0.1), r.quantile(0.9)
    # 3. persistencia y acoplamiento
    cols["spr_autocorr1"] = s.rolling(56, min_periods=21).corr(s.shift(1))
    cols["spr_autocorr7"] = s.rolling(56, min_periods=21).corr(s.shift(7))
    m8, x8 = Yt.mean(axis=1).shift(LAG_MTR), x.shift(LAG_MTR)
    for w in (28, 84):
        cols[f"mtr_mda_corr{w}"] = m8.rolling(w, min_periods=14).corr(x8)
    forma = _corr_filas(Yt, Xt).shift(LAG_MTR)
    cols["forma_corr7"], cols["forma_corr28"] = forma.rolling(7, min_periods=4).mean(), forma.rolling(28, min_periods=14).mean()
    # 4. el MDA de hoy contra su historia
    xm = Xt.max(axis=1)
    for w in (7, 28, 84, 364):
        r = x.shift(1).rolling(w, min_periods=min(w, 21))
        cols[f"mda_z{w}"] = (x - r.mean()) / r.std()
    r = xm.shift(1).rolling(84, min_periods=21)
    cols["mda_max_z84"] = (xm - r.mean()) / r.std()
    # 5. el día contra el resto
    cols["mda_forma_d1"], cols["mda_forma_d7"] = _corr_filas(Xt, Xt.shift(1)), _corr_filas(Xt, Xt.shift(7))
    idx = Xt.index
    pos = idx.get_indexer(c.dias)
    conocido = ~spr.isna().any(axis=1).values
    s_dia = spr.mean(axis=1).values                       # error del día t (sólo se usa en vecinos con t <= d−8)
    dia = np.column_stack([Xt.values, 2 * x.values[:, None]])                         # perfil y nivel del MDA
    sem = np.column_stack([x.shift(k).values for k in range(7)] + [xm.shift(k).values for k in range(7)])   # última semana
    rz_d, an_d = _vecinos(dia, conocido, s_dia, pos)
    rz_s, an_s = _vecinos(sem, conocido, s_dia, pos)
    out = pd.DataFrame(cols).reindex(c.dias)
    out["rareza_dia"], out["analogo_dia_spr"], out["rareza_semana"], out["analogo_semana_spr"] = rz_d, an_d, rz_s, an_s
    # 6. error del MDA por hora a dos escalas
    horas = [spr.shift(LAG_MTR).ewm(halflife=hl, min_periods=7).mean().set_axis([f"spr_ewm{hl}_h{h}" for h in spr.columns], axis=1)
             for hl in (7, 28)]
    return pd.concat([out] + [H.reindex(c.dias) for H in horas], axis=1)


_RED_AGR = {}


def _agregados_red(c):
    """Resúmenes diarios de toda la red (nodos de `c.nodos` más el propio contexto), en la escala del proyecto. Se
    guardan por contexto porque el GB global llama esta función una vez por cada uno de los 65 nodos."""
    clave = (id(c.nodos["MTR"]), id(c.nodos["MDA"]), c.med, c.mad, c.modo, c.escala)
    if clave in _RED_AGR and _RED_AGR[clave][0] is c.nodos:
        return _RED_AGR[clave][1]
    idx = c.X.asfreq("D").index
    sp = {n: (fw(c.nodos["MTR"][n], c) - fw(c.nodos["MDA"][n], c)).reindex(idx) for n in c.nodos["MTR"]}
    xd = pd.DataFrame({n: fw(M, c).reindex(idx).mean(axis=1) for n, M in c.nodos["MDA"].items()})      # MDA diario por nodo
    sd = pd.DataFrame({n: S.mean(axis=1) for n, S in sp.items()}).shift(LAG_MTR)                        # error diario, publicado
    sh = pd.concat(sp.values()).groupby(level=0).median().reindex(idx).shift(LAG_MTR)                   # error por hora, mediana
    med = sd.median(axis=1)
    r = sd.rolling(28, min_periods=14)
    sincronia = med.rolling(28, min_periods=14).var() / r.var().mean(axis=1)    # ≈1: toda la red se equivoca junta
    A = dict(xd=xd, sd=sd, sh=sh, med=med, sincronia=sincronia)
    _RED_AGR[clave] = (c.nodos, A)
    return A


def g14(c):
    """La red en el espacio (notebook 07): ¿el error del MDA es de todo el sistema o de este nodo?

      - dispersión del MDA de hoy entre los nodos (congestión prevista en d) y la posición del nodo en esa dispersión;
      - error mediano de la red ya publicado (d−8) y suavizado a 7 y 28 días, por día y por hora;
      - sincronía de la red en los últimos 28 días publicados: cerca de 1 si todos los nodos se equivocan igual
        (energía), cerca de 0 si cada uno va por su lado (congestión);
      - cuánto se separa el error de este nodo del de la red en los últimos 28 días publicados.
    Requiere `c.nodos` (ver `con_nodos`). MDA de la red hasta d; MTR hasta d−8."""
    A = _agregados_red(c)
    Yt, Xt = _t(c)
    propio = (Yt - Xt).mean(axis=1).shift(LAG_MTR)
    x = Xt.mean(axis=1)
    cols = {"red_mda_disp": A["xd"].std(axis=1), "red_mda_pos": x - A["xd"].median(axis=1),
            "red_spr": A["med"], "red_spr_ewm7": A["med"].ewm(halflife=7, min_periods=7).mean(),
            "red_spr_ewm28": A["med"].ewm(halflife=28, min_periods=7).mean(), "red_sincronia28": A["sincronia"],
            "nodo_vs_red28": (propio - A["med"]).rolling(28, min_periods=14).mean()}
    H = A["sh"].ewm(halflife=7, min_periods=7).mean()
    H.columns = [f"red_spr_ewm7_h{h}" for h in H.columns]
    return pd.concat([pd.DataFrame(cols), H], axis=1).reindex(c.dias)


# ---------------------------------------------------------------------------------------- complejidad y correlaciones
def _entropia_perm(x, m=4):
    """Entropía de permutación normalizada: 0 si la serie sigue siempre el mismo patrón de subidas y bajadas, 1 si los
    patrones (de m valores seguidos) aparecen al azar. Mide qué tan irregular es la serie."""
    import math
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < m + 10:
        return np.nan
    cod = (np.lib.stride_tricks.sliding_window_view(x, m).argsort(axis=1) * (m ** np.arange(m))).sum(axis=1)
    _, n = np.unique(cod, return_counts=True); p = n / n.sum()
    return float(-(p * np.log(p)).sum() / np.log(math.factorial(m)))


def _entropia_espectral(x):
    """Entropía del espectro de frecuencias, normalizada: baja si la serie tiene un ritmo dominante, alta si su energía
    está repartida en todas las frecuencias (como el ruido)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 28:
        return np.nan
    p = np.abs(np.fft.rfft(x - x.mean()))[1:] ** 2
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p) + 1e-12))


def _hurst(x):
    """Exponente de Hurst por rango reescalado: cerca de 0.5 si el error va al azar, mayor si sus rachas persisten,
    menor si tiende a revertirse."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 32:
        return np.nan
    tam, rs = [], []
    for s in (8, 16, 32, 64):
        if s > len(x):
            break
        trozos = x[: len(x) // s * s].reshape(-1, s)
        z = np.cumsum(trozos - trozos.mean(axis=1, keepdims=True), axis=1)
        r, sd = z.max(axis=1) - z.min(axis=1), trozos.std(axis=1)
        ok = sd > 0
        if ok.any():
            tam.append(s); rs.append(np.mean(r[ok] / sd[ok]))
    return float(np.polyfit(np.log(tam), np.log(rs), 1)[0]) if len(tam) >= 2 else np.nan


def _pendiente(y: pd.Series, x: pd.Series, w: int, minimo: int):
    """Pendiente móvil de la regresión de y sobre x en ventanas de w días (cuánto cambia y por cada unidad de x)."""
    r = lambda s: s.rolling(w, min_periods=minimo).mean()
    v = r(x * x) - r(x) ** 2
    return (r(x * y) - r(x) * r(y)) / v.where(v > 1e-9)


def g15(c):
    """Complejidad y correlaciones (notebook 07). Resume cómo se ha comportado el error del MDA, no sólo cuánto:

      complejidad del error publicado (<= d−8):
        - entropía de permutación del error hora por hora en 1 y 4 semanas (qué tan irregular es);
        - entropía espectral y exponente de Hurst del error diario (si tiene ritmo; si persiste o se revierte);
        - asimetría, curtosis (colas) y volatilidad de la volatilidad del error;
      complejidad del MDA (hasta d): entropía de permutación de la última semana;
      correlaciones:
        - coherencia entre horas: correlación media entre los errores de las 24 horas en 4 semanas;
        - estabilidad del perfil horario del error: la última semana publicada contra las tres anteriores;
        - sensibilidad del MDA al calor nacional y al gas (pendiente y correlación móviles);
        - sensibilidad del error del MDA al calor y a los saltos del MDA, y su efecto esperado hoy (sensibilidad × valor
          de hoy): así la relación aprendida en el pasado se vuelve una variable del día;
        - correlación del error de este nodo con el de la red (si hay nodos).
    Reglas: MTR hasta d−8, MDA hasta d, clima pronosticado (previous_day2) hasta d, gas con su rezago de publicación."""
    Yt, Xt = _t(c)
    idx = Xt.index
    S8 = (Yt - Xt).shift(LAG_MTR)                         # error horario ya publicado
    s = S8.mean(axis=1)                                   # error diario ya publicado
    x = Xt.mean(axis=1)
    pos = {d: i for i, d in enumerate(idx)}
    Sv, Xv = S8.values, Xt.values
    filas = {}
    for d in c.dias:
        i = pos[d]
        v7, v28 = Sv[max(i - 6, 0):i + 1].ravel(), Sv[max(i - 27, 0):i + 1].ravel()
        w28 = Sv[max(i - 27, 0):i + 1]; w28 = w28[~np.isnan(w28).any(axis=1)]
        coh = np.nan
        if len(w28) >= 14:
            C = np.corrcoef(w28.T); coh = float((C.sum() - np.trace(C)) / (24 * 23))
        prof = np.nan
        if len(w28) >= 21:
            prof = float(np.corrcoef(w28[-7:].mean(axis=0), w28[:-7].mean(axis=0))[0, 1])
        v28n = v28[~np.isnan(v28)]
        filas[d] = {"spr_entropia7": _entropia_perm(v7), "spr_entropia28": _entropia_perm(v28),
                    "spr_entropia_espectral56": _entropia_espectral(s.values[max(i - 55, 0):i + 1]),
                    "spr_hurst84": _hurst(s.values[max(i - 83, 0):i + 1]),
                    "spr_asimetria28": float(pd.Series(v28n).skew()) if len(v28n) > 48 else np.nan,
                    "spr_curtosis28": float(pd.Series(v28n).kurt()) if len(v28n) > 48 else np.nan,
                    "mda_entropia7": _entropia_perm(Xv[max(i - 6, 0):i + 1].ravel()),
                    "coherencia_horas28": coh, "estabilidad_perfil": prof}
    out = pd.DataFrame.from_dict(filas, orient="index")
    a = (Yt - Xt).abs().mean(axis=1).shift(LAG_MTR)
    out["spr_volvol28"] = a.rolling(7, min_periods=4).std().rolling(28, min_periods=14).std().reindex(c.dias)
    cols = {}
    if c.Wi:
        T = c.Wi[2]["temperature_2m"].reindex(idx).mean(axis=1)               # temperatura nacional pronosticada para cada día
        cols["mda_calor_pendiente28"] = _pendiente(x, T, 28, 14)
        cols["mda_calor_corr28"] = x.rolling(28, min_periods=14).corr(T)
        b = _pendiente(s, T.shift(LAG_MTR), 56, 21)                              # error publicado contra el calor de esos días
        cols["spr_calor_pendiente56"] = b
        cols["spr_calor_efecto"] = b * (T - T.shift(LAG_MTR).rolling(56, min_periods=21).mean())
    if "hh_spot" in c.fred:
        g = c.fred["hh_spot"].reindex(pd.date_range(c.fred["hh_spot"].index.min(), idx.max())).ffill().shift(c.lag_precios).reindex(idx)
        cols["mda_gas_corr56"] = x.rolling(56, min_periods=21).corr(g)
    dx = x - x.shift(1)                                                           # salto del MDA de un día al siguiente
    b = _pendiente(s, dx.shift(LAG_MTR), 56, 21)
    cols["spr_salto_pendiente56"] = b
    cols["spr_salto_efecto"] = b * dx
    if c.nodos:
        cols["corr_con_la_red28"] = s.rolling(28, min_periods=14).corr(_agregados_red(c)["med"])
    return pd.concat([out, pd.DataFrame(cols).reindex(c.dias)], axis=1)


def g16(c):
    """Complejidad y correlaciones de la serie de MTR misma (notebook 07). Es G15 aplicado al precio y no sólo al error del
    MDA: cómo se comporta el MTR, aparte de cuánto se separa del MDA.

      - entropía de permutación del MTR hora por hora en 1 y 4 semanas, entropía espectral y Hurst del MTR diario;
      - asimetría, curtosis y volatilidad de la volatilidad del MTR;
      - coherencia entre horas (qué tan juntas se mueven las 24 horas del MTR) y estabilidad de su perfil horario;
      - cuánto sigue el MTR diario al MDA diario (correlación en 4 semanas) y cuánto de cada salto del MDA se transmite al
        MTR (pendiente móvil), con su efecto esperado hoy (pendiente × salto de hoy);
      - sensibilidad del MTR al calor nacional pronosticado y correlación con el gas, y correlación con el MTR de la red.
    Reglas: MTR hasta d−8, MDA hasta d (el de hoy sólo en el salto de hoy), clima hasta d, gas con su rezago."""
    Yt, Xt = _t(c)
    idx = Xt.index
    Y8 = Yt.shift(LAG_MTR)                                # MTR ya publicado
    m = Y8.mean(axis=1)                                   # MTR diario publicado
    x = Xt.mean(axis=1); x8 = x.shift(LAG_MTR)
    pos = {d: i for i, d in enumerate(idx)}
    Yv = Y8.values
    filas = {}
    for d in c.dias:
        i = pos[d]
        v7, v28 = Yv[max(i - 6, 0):i + 1].ravel(), Yv[max(i - 27, 0):i + 1].ravel()
        w28 = Yv[max(i - 27, 0):i + 1]; w28 = w28[~np.isnan(w28).any(axis=1)]
        coh = prof = np.nan
        if len(w28) >= 14:
            C = np.corrcoef(w28.T); coh = float((C.sum() - np.trace(C)) / (24 * 23))
        if len(w28) >= 21:
            prof = float(np.corrcoef(w28[-7:].mean(axis=0), w28[:-7].mean(axis=0))[0, 1])
        v28n = v28[~np.isnan(v28)]
        filas[d] = {"mtr_entropia7": _entropia_perm(v7), "mtr_entropia28": _entropia_perm(v28),
                    "mtr_entropia_espectral56": _entropia_espectral(m.values[max(i - 55, 0):i + 1]),
                    "mtr_hurst84": _hurst(m.values[max(i - 83, 0):i + 1]),
                    "mtr_asimetria28": float(pd.Series(v28n).skew()) if len(v28n) > 48 else np.nan,
                    "mtr_curtosis28": float(pd.Series(v28n).kurt()) if len(v28n) > 48 else np.nan,
                    "mtr_coherencia_horas28": coh, "mtr_estabilidad_perfil": prof}
    out = pd.DataFrame.from_dict(filas, orient="index")
    out["mtr_volvol28"] = m.rolling(7, min_periods=4).std().rolling(28, min_periods=14).std().reindex(c.dias)
    cols = {"mtr_sigue_mda28": m.rolling(28, min_periods=14).corr(x8)}
    dm, dx = m.diff(), (x - x.shift(1))
    b = _pendiente(dm, dx.shift(LAG_MTR), 56, 21)                                  # cuánto del salto del MDA se transmite al MTR
    cols["mtr_salto_pendiente56"] = b
    cols["mtr_salto_efecto"] = b * dx
    if c.Wi:
        T = c.Wi[2]["temperature_2m"].reindex(idx).mean(axis=1)
        b = _pendiente(m, T.shift(LAG_MTR), 56, 21)
        cols["mtr_calor_pendiente56"] = b
        cols["mtr_calor_efecto"] = b * (T - T.shift(LAG_MTR).rolling(56, min_periods=21).mean())
    if "hh_spot" in c.fred:
        g = c.fred["hh_spot"].reindex(pd.date_range(c.fred["hh_spot"].index.min(), idx.max())).ffill().shift(c.lag_precios).reindex(idx)
        cols["mtr_gas_corr56"] = m.rolling(56, min_periods=21).corr(g)
    if c.nodos:
        mr = pd.DataFrame({n: fw(M, c).reindex(idx).mean(axis=1) for n, M in c.nodos["MTR"].items()}).shift(LAG_MTR).median(axis=1)
        cols["mtr_corr_con_la_red28"] = m.rolling(28, min_periods=14).corr(mr)
    return pd.concat([out, pd.DataFrame(cols).reindex(c.dias)], axis=1)


def comprimir(F: pd.DataFrame, k=20, hasta="2024-06-30", prefijo="pc", ajustar_con=None):
    """Resume muchas variables en `k` componentes principales, para que entren juntas a un modelo sin saturarlo.
    La estandarización y las componentes se ajustan SÓLO con días hasta `hasta` (antes del desarrollo) y después se aplican
    igual a cada día, así que la variable del día d sólo depende de lo que se sabía en d. Devuelve (componentes,
    fracción de la varianza que conservan). `ajustar_con`: otra tabla con las mismas columnas, con la que se ajusta en
    lugar de `F` (por ejemplo, las filas de todos los nodos apiladas; su índice puede repetir fechas)."""
    from sklearn.decomposition import PCA
    A = F if ajustar_con is None else ajustar_con[F.columns]
    tr = A[A.index <= pd.Timestamp(hasta)].dropna()
    mu, sd = tr.mean(), tr.std().replace(0, 1).fillna(1)
    p = PCA(k, random_state=0).fit(((tr - mu) / sd).values)
    Z = ((F - mu) / sd).fillna(0)
    return pd.DataFrame(p.transform(Z.values), index=F.index, columns=[f"{prefijo}{i + 1}" for i in range(k)]), float(p.explained_variance_ratio_.sum())


def cargas(F: pd.DataFrame, k=20, hasta="2024-06-30"):
    """Cargas (loadings) de las componentes principales de `comprimir`: una fila por componente y una columna por variable,
    y la fracción de la varianza que explica cada componente. El ajuste es el mismo (estandarizar y PCA con días <= `hasta`),
    así que las componentes son las mismas que usan los modelos."""
    from sklearn.decomposition import PCA
    tr = F.loc[:hasta].dropna()
    mu, sd = tr.mean(), tr.std().replace(0, 1).fillna(1)
    p = PCA(k, random_state=0).fit(((tr - mu) / sd).values)
    return pd.DataFrame(p.components_, index=[f"pc{i + 1}" for i in range(k)], columns=F.columns), pd.Series(p.explained_variance_ratio_, index=[f"pc{i + 1}" for i in range(k)])


def con_nodos(c: Ctx, troncal=True) -> Ctx:
    """El contexto con los precios de los nodos de la red (para G14). Monclova se incluye como un nodo más."""
    from . import red
    Ym, Xm = red.matrices_nodo("MTR", troncal=troncal), red.matrices_nodo("MDA", troncal=troncal)
    return replace(c, nodos={"MTR": {"monclova": c.Y, **Ym}, "MDA": {"monclova": c.X, **Xm}})


GRUPOS = {
    "G0": (g0,"MTR rezagos d−8, d−9, d−10, d−14 (24 h c/u)"), "G0w": (g0w, "MTR ventana semanal d−8…d−14 (168 h)"),
    "G0dow": (g0dow, "MTR mismo día de la semana d−21, d−28"),
    "G1": (g1, "calendario base: dow, feriado, Fourier anual k=1,2"), "G1f": (g1f, "calendario compacto: Fourier semanal+anual"),
    "G2": (g2, "MDA d, d−1, d−7 y spread d−8, d−14"), "G2w": (g2w, "MDA d…d−7 y spread d−8…d−14 (ventana semanal)"),
    "G4": (g4, "estadísticas móviles de MTR/spread (7/28/84 d) y del MDA"), "G5": (g5, "clima de la zona (pronóstico previous_day2)"),
    "G6": (g6, "combustibles y FX (FRED)"), "G6g": (g6g, "inventarios de gas (EIA)"), "G6e": (g6e, "demanda ERCOT día anterior (EIA-930)"),
    "G7": (g7, "calendario ampliado: mes, puentes, vacaciones, geometría solar"),
    "G7v": (g7v, "índice de vacaciones con calendario SEP real"),
    "G8": (g8, "estrés del sistema: clima Monterrey/Texas, heladas, estados operativos CENACE"),
    "G9": (g9, "información pública del día anterior: operación y precio RT de ERCOT, clima nacional y su revisión"),
    "G10": (g10, "congestión: componente de congestión del MDA de Monclova y de los corredores de la red"),
    "G11": (g11, "clima nacional como serie: índice ponderado por consumo, calor acumulado, lluvia, revisiones y error reciente"),
    "G12": (g12, "días análogos: qué hizo el MTR en los días pasados más parecidos"),
    "G13": (g13, "varias escalas y largo alcance: medias exponenciales, colas, persistencia, rareza del día y análogos"),
    "G14": (g14, "la red en el espacio: dispersión del MDA entre nodos, error de la red y su sincronía"),
    "G16": (g16, "complejidad y correlaciones del MTR: entropías, persistencia, coherencia entre horas, seguimiento del MDA y sensibilidades"),
    "G15": (g15, "complejidad y correlaciones: entropías, persistencia, coherencia entre horas y sensibilidades al calor, al gas y al MDA"),
}


def construir(c: Ctx, grupos) -> tuple:
    """Devuelve (F, grupo_por_columna). Los grupos alternativos (G0/G0w, G2/G2w, G1/G1f) no deben combinarse."""
    partes, g = [], {}
    for gr in grupos:
        df = GRUPOS[gr][0](c)
        dup = set(df.columns) & set(g)
        assert not dup, f"columnas duplicadas entre grupos: {sorted(dup)[:5]}"
        partes.append(df)
        g.update({col: gr for col in df.columns})
    F = pd.concat(partes, axis=1)
    return F, pd.Series(g)


# ---------------------------------------------------------------------------------------- prueba de leakage
def perturbar(c: Ctx, d: pd.Timestamp, rng=None) -> Ctx:
    """Copia del contexto con basura en todo lo que NO estaría disponible al emitir el pronóstico del día d."""
    rng = rng or np.random.default_rng(0)
    d = pd.Timestamp(d)
    basura = lambda df: pd.DataFrame(rng.normal(3000, 1500, df.shape), index=df.index, columns=df.columns)
    Y, X, ER = c.Y.copy(), c.X.copy(), c.ercot.copy()
    m = Y.index > d - pd.Timedelta(days=LAG_MTR); Y.loc[m] = basura(Y.loc[m]).values
    m = X.index > d; X.loc[m] = basura(X.loc[m]).values
    m = ER.index > d; ER.loc[m] = basura(ER.loc[m]).values
    W = {}
    for k, v in c.W.items():
        v = v.copy(); m = v.index > d; v.loc[m] = basura(v.loc[m]).values; W[k] = v
    Wext = {}
    for p, w in c.Wext.items():
        Wext[p] = {}
        for k, v in w.items():
            v = v.copy(); m = v.index > d; v.loc[m] = basura(v.loc[m]).values; Wext[p][k] = v
    Wnac = {}
    for k, dp in c.Wnac.items():
        Wnac[k] = {}
        for p, w in dp.items():
            Wnac[k][p] = {}
            for var, v in w.items():
                v = v.copy(); m = v.index > d; v.loc[m] = basura(v.loc[m]).values; Wnac[k][p][var] = v
    e930 = {}
    for k, v in c.e930.items():
        v = v.copy(); m = v.index > d - pd.Timedelta(days=2); v.loc[m] = basura(v.loc[m]).values; e930[k] = v
    rt = None
    if c.rt is not None:
        rt = c.rt.copy(); m = rt.index >= d - pd.Timedelta(days=1) + pd.Timedelta(hours=11)
        rt.loc[m] = rng.normal(3000, 1500, rt.loc[m].shape)
    eo = None
    if c.eo is not None:
        eo = c.eo.copy(); eo[eo.index > d - pd.Timedelta(days=2)] = 2
        eo = pd.concat([eo, pd.Series(2, index=pd.date_range(d - pd.Timedelta(days=1), d + pd.Timedelta(days=5)))]).groupby(level=0).max()
    L = c.lag_precios
    fred = {}
    for k, s in c.fred.items():
        s = s.copy(); m = s.index > d - pd.Timedelta(days=L); s.loc[m] = rng.normal(50, 20, m.sum()); fred[k] = s
    inv = c.inv.copy(); m = (inv.index + pd.Timedelta(days=6)) > d - pd.Timedelta(days=L); inv.loc[m] = rng.normal(3000, 500, m.sum())
    def tapa(M, corte):                                                     # basura después del corte
        """Copia de M con basura en todas las fechas posteriores al corte."""
        M = M.copy(); m = M.index > corte
        M.loc[m] = basura(M.loc[m]).values
        return M
    comp = {"MTR": {k: tapa(v, d - pd.Timedelta(days=LAG_MTR)) for k, v in c.comp.get("MTR", {}).items()},
            "MDA": {k: tapa(v, d) for k, v in c.comp.get("MDA", {}).items()}} if c.comp else {}
    red = {"MTR": {k: tapa(v, d - pd.Timedelta(days=LAG_MTR)) for k, v in c.red.get("MTR", {}).items()},
           "MDA": {k: tapa(v, d) for k, v in c.red.get("MDA", {}).items()}} if c.red else {}
    Wi = {k: {v: tapa(M, d - pd.Timedelta(days=2) if k == 0 else d) for v, M in w.items()} for k, w in c.Wi.items()}
    nodos = {"MTR": {n: tapa(M, d - pd.Timedelta(days=LAG_MTR)) for n, M in c.nodos["MTR"].items()},
             "MDA": {n: tapa(M, d) for n, M in c.nodos["MDA"].items()}} if c.nodos else {}
    return replace(c, Y=Y, X=X, W=W, fred=fred, inv=inv, ercot=ER, Wext=Wext, eo=eo, Wnac=Wnac, e930=e930, rt=rt, comp=comp, red=red,
                   Wi=Wi, nodos=nodos)


# ---------------------------------------------------------------------------------------- carga del contexto
DEV_INI, DEV_FIN = pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")
TEST_INI, TEST_FIN = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")
TRAIN_START = pd.Timestamp("2024-02-05")   # primer día con clima previous_day2 sin huecos y 14 días previos válidos


def cargar_ctx(lag_precios: int = 2) -> Ctx:
    """Contexto con todos los datos. Las constantes de asinh (mediana/MAD) se fijan SÓLO con MTR de
    TRAIN_START a 2024-06-30, es decir, antes del periodo de desarrollo y del test."""
    from . import exog
    from .data import cargar_largo, matriz
    from .features import asinh_fit
    mtr, mda = cargar_largo()
    Y, X = matriz(mtr), matriz(mda)
    med, mad = asinh_fit(Y.loc[TRAIN_START:"2024-06-30"].values.ravel())
    dias = pd.date_range(TRAIN_START, TEST_FIN)
    return Ctx(Y=Y, X=X, W=exog.zona(), fred=exog.combustibles(), inv=exog.inventarios_gas(), ercot=exog.ercot_demanda("DF"),
               med=med, mad=mad, dias=dias, lag_precios=lag_precios, Wext=exog.externo(), eo=exog.estados_operativos(),
               Wnac=exog.nacional(), e930=exog.eia930_ercot(), rt=exog.ercot_rt(), comp=_componentes(mtr, mda), red=_red(),
               Wi=exog.indice_nacional())


def _componentes(mtr, mda) -> dict:
    """Componentes (energía, pérdidas, congestión) de Monclova en MTR y MDA, como matrices días × 24."""
    from .data import matriz
    return {p: {k: matriz(df, f"pml_{k}") for k in ("ene", "per", "cng")} for p, df in (("MTR", mtr), ("MDA", mda))}


def _red() -> dict:
    """Congestión media de cada corredor de la red (G10), si ya se descargaron los nodos."""
    from . import red
    try:
        return red.matrices()
    except FileNotFoundError:                     # sin la descarga de la red los grupos G10 de corredores quedan vacíos
        return {}
