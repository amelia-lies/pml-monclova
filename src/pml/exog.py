"""Datos exógenos (FRED, EIA, Open-Meteo) con caché cruda y reglas explícitas de disponibilidad.

Convención de emisión: el pronóstico del día d se emite el día d−1 hacia el mediodía (hora local, UTC−6).
Toda serie diaria se alinea con `asof`: valor de la última observación cuya fecha de *disponibilidad*
sea <= d − lag. Las credenciales se leen de variables de entorno o de `.env` y nunca se imprimen.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import truststore

truststore.inject_into_ssl()
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/exog"
INI, FIN = "2023-11-01", "2025-06-30"
TRAIN_START = pd.Timestamp("2024-02-05")   # primer día con clima previous_day2 sin huecos y 14 días previos válidos

# Cabeceras (GeoNames vía Open-Meteo geocoding): Monclova y Ciudad Frontera, Coahuila
PUNTOS_ZONA = {"Monclova": (26.90687, -101.42056), "Frontera": (26.92814, -101.45212)}
# Notebook 03 (estrés del sistema): el error está en la energía (precio nacional), así que también importa el clima donde está la carga grande del
# Noreste (Monterrey) y en Texas, de donde viene el gas importado: calor en San Antonio (Eagle Ford / demanda ERCOT) y
# heladas en Midland (Cuenca Pérmica, congelamiento de pozos).
PUNTOS_EXT = {"Monterrey": (25.68661, -100.31611), "SanAntonio": (29.42412, -98.49363), "Midland": (31.99735, -102.07791)}
# Notebook 03 (clima nacional): clima de los grandes centros de carga y de las zonas renovables del SIN (la energía es un precio nacional).
PUNTOS_NAC = {"CDMX": (19.43261, -99.13321), "Guadalajara": (20.65970, -103.34961), "Hermosillo": (29.07297, -110.95592),
              "Juchitan": (16.43360, -95.01940), "Reynosa": (26.05080, -98.29790)}
VARS_METEO = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "dew_point_2m", "cloud_cover",
              "shortwave_radiation", "direct_normal_irradiance", "wind_speed_10m", "precipitation", "surface_pressure"]
ACUMULADAS = {"shortwave_radiation", "direct_normal_irradiance", "precipitation"}  # valor = promedio/suma de la hora previa


def _key(nombre: str) -> str:
    """Lee una llave de API de las variables de entorno o, si no está, de Solution/.env. Nunca se imprime."""
    v = os.environ.get(nombre)
    if v:
        return v
    f = ROOT / ".env"
    if f.exists():
        for linea in f.read_text().splitlines():
            if linea.startswith(nombre + "="):
                return linea.split("=", 1)[1].strip()
    raise RuntimeError(f"Falta la credencial {nombre} (variable de entorno o .env)")


def _get(url: str, params: dict, timeout=120, reintentos=5):
    """GET con reintentos y espera exponencial ante 429/5xx o fallas de red."""
    ultimo = None
    for i in range(reintentos):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code < 400:
                return r
            ultimo = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            ultimo = type(e).__name__
        time.sleep(min(3 * 2 ** i, 40))
    raise RuntimeError(f"Falló {url.split('?')[0]}: {ultimo}")


def _cache(nombre: str, fn):
    """Lee de la caché cruda si existe; si no, llama a `fn()` y guarda el resultado."""
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / nombre
    if f.exists():
        return json.loads(f.read_text())
    data = fn()
    f.write_text(json.dumps(data))
    return data


def fred(sid: str, ini=INI, fin=FIN) -> pd.Series:
    """Serie diaria de FRED (índice = fecha de la observación; sin NaN)."""
    def pedir():
        r = _get("https://api.stlouisfed.org/fred/series/observations", dict(series_id=sid, api_key=_key("FRED_API_KEY"), file_type="json",
                                     observation_start=ini, observation_end=fin))
        return r.json()["observations"]
    obs = _cache(f"fred_{sid}.json", pedir)
    s = pd.Series({pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce") for o in obs}, name=sid)
    return s.dropna().sort_index()


def eia(ruta: str, nombre: str, facets: dict | None = None, frecuencia="daily", ini=INI, fin=FIN) -> pd.DataFrame:
    """Consulta paginada a EIA API v2. Devuelve DataFrame con period, series/type y value numérico."""
    def pedir():
        out, offset = [], 0
        while True:
            p = {"api_key": _key("EIA_API_KEY"), "frequency": frecuencia, "data[0]": "value", "start": ini, "end": fin,
                 "sort[0][column]": "period", "sort[0][direction]": "asc", "offset": offset, "length": 5000}
            for k, v in (facets or {}).items():
                p[f"facets[{k}][]"] = v
            r = _get(f"https://api.eia.gov/v2/{ruta}/data/", p)
            d = r.json()["response"]["data"]
            out += d
            if len(d) < 5000:
                return out
            offset += 5000
    df = pd.DataFrame(_cache(f"eia_{nombre}.json", pedir))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def meteo_previous(lat: float, lon: float, dia_previo: int = 2, ini=INI, fin=FIN) -> pd.DataFrame:
    """Pronóstico archivado emitido >= dia_previo×24 h antes de la hora válida (Open-Meteo Previous Runs).
    Índice horario local naive (America/Mexico_City). `dia_previo=2` garantiza emisión anterior a las 12:00 de d−1."""
    suf = f"_previous_day{dia_previo}" if dia_previo else ""      # 0 = lo mejor disponible (primeras horas de cada corrida ≈ observado)
    def pedir():
        r = _get("https://previous-runs-api.open-meteo.com/v1/forecast", dict(latitude=lat, longitude=lon, start_date=ini, end_date=str((pd.Timestamp(fin) + pd.Timedelta(days=1)).date()), timezone="America/Mexico_City",
                                     hourly=",".join(v + suf for v in VARS_METEO)), timeout=180)
        return r.json()
    j = _cache(f"meteo_prev{dia_previo}_{lat:.4f}_{lon:.4f}.json", pedir)
    h = pd.DataFrame(j["hourly"]).set_index("time")
    h.index = pd.to_datetime(h.index)
    h.columns = [c.replace(suf, "") for c in h.columns]
    return h


def asof(s: pd.Series, dias: pd.DatetimeIndex, lag: int, retraso_pub: int = 0) -> pd.Series:
    """Valor conocido al emitir para el día d: última observación con fecha+retraso_pub <= d − lag."""
    disp = pd.Series(s.values, index=s.index + pd.Timedelta(days=retraso_pub)).sort_index()
    corte = dias - pd.Timedelta(days=lag)
    return pd.Series(disp.reindex(disp.index.union(corte)).ffill().reindex(corte).values, index=dias, name=s.name)


def a_matriz_horaria(serie_h: pd.Series, acumulada: bool = False) -> pd.DataFrame:
    """Serie horaria local -> matriz días × horas 1..24 (hora 1 = 00:00–01:00).
    Variable instantánea: valor en t.hour = inicio de la hora (hora = t.hour+1).
    Acumulada/promedio de la hora previa: el valor en t pertenece a la hora que termina en t (hora = t.hour, 0→24 del día previo)."""
    t = serie_h.index
    if acumulada:
        fecha = (t - pd.Timedelta(hours=1)).normalize(); hora = ((t - pd.Timedelta(hours=1)).hour + 1)
    else:
        fecha = t.normalize(); hora = t.hour + 1
    m = pd.DataFrame({"fecha": fecha, "hora": hora, "v": serie_h.values}).pivot_table(index="fecha", columns="hora", values="v")
    return m[[h for h in range(1, 25) if h in m.columns]]


def zona(puntos=PUNTOS_ZONA, dia_previo=2) -> dict:
    """Matrices días × 24 por variable, promedio de los puntos de la zona."""
    hs = [meteo_previous(la, lo, dia_previo) for la, lo in puntos.values()]
    out = {}
    for v in VARS_METEO:
        ms = [a_matriz_horaria(h[v], v in ACUMULADAS) for h in hs]
        out[v] = pd.concat(ms).groupby(level=0).mean()     # promedio de los puntos disponibles (ignora faltantes)
    return out


def externo(puntos=PUNTOS_EXT, dia_previo=2) -> dict:
    """{punto: {variable: matriz días × 24}} para los puntos fuera de la zona. Misma regla de emisión que `zona`."""
    return {p: zona({p: ll}, dia_previo) for p, ll in puntos.items()}


def nacional(puntos=PUNTOS_NAC, dias_previos=(2, 3)) -> dict:
    """{dia_previo: {punto: {variable: matriz días × 24}}}. El pronóstico con 3 días de anticipación sirve para medir
    cuánto se revisó el pronóstico en el último día (todo emitido antes del mediodía de d−1)."""
    return {k: {p: zona({p: ll}, k) for p, ll in puntos.items()} for k in dias_previos}


# Notebook 03 (clima nacional como serie): índice ponderado por consumo. Un punto por división tarifaria de CFE del SIN (las de Baja California
# no están interconectadas); peso = participación en la energía vendida 2024 (CNE, memorias de cálculo de tarifas, vía
# el Atlas de Batu). Las tres divisiones del Valle de México comparten el punto de la CDMX.
PUNTOS_IDX = {
    "CDMX": ((19.43261, -99.13321), 0.033 + 0.042 + 0.046), "Monterrey": ((25.68661, -100.31611), 0.139),
    "Leon": ((21.12908, -101.67374), 0.106), "Chihuahua": ((28.63528, -106.08889), 0.089 / 2),
    "Torreon": ((25.54389, -103.41898), 0.089 / 2), "Hermosillo": ((29.07297, -110.95592), 0.084),
    "Guadalajara": ((20.65970, -103.34961), 0.067), "Merida": ((20.97537, -89.61696), 0.055),
    "Veracruz": ((19.18095, -96.14290), 0.048), "Puebla": ((19.03793, -98.20346), 0.048),
    "Villahermosa": ((17.98689, -92.93028), 0.046), "Tampico": ((22.24484, -97.86140), 0.045),
    "Cuernavaca": ((18.92110, -99.23404), 0.039), "Morelia": ((19.70078, -101.18443), 0.036),
}
VARS_IDX = ["temperature_2m", "precipitation", "cloud_cover", "shortwave_radiation"]


def indice_nacional(dias_previos=(0, 2, 3, 4)) -> dict:
    """{dia_previo: {variable: matriz días × 24}} promedio ponderado por consumo de los puntos de PUNTOS_IDX.
    dia_previo=0 es el clima "observado" (primeras horas de cada corrida); sólo puede usarse para días <= d−2."""
    w = pd.Series({p: v[1] for p, v in PUNTOS_IDX.items()}); w = w / w.sum()
    out = {}
    for k in dias_previos:
        hs = {p: meteo_previous(*ll, k) for p, (ll, _) in PUNTOS_IDX.items()}
        out[k] = {}
        for v in VARS_IDX:
            ms = {p: a_matriz_horaria(h[v], v in ACUMULADAS) for p, h in hs.items()}
            idx = sorted(set.intersection(*(set(m.index) for m in ms.values())))
            out[k][v] = sum(ms[p].loc[idx] * w[p] for p in ms)
    return out


def eia930_ercot() -> dict:
    """EIA-930 de ERCOT por hora, en hora de México: demanda real (D), generación neta (NG), generación eólica, solar y de
    gas, e intercambio ERCOT → CENACE (positivo = ERCOT exporta a México). Matrices días × 24."""
    def mat(df, col="value"):
        """Del formato horario largo a la matriz días × 24."""
        t = pd.to_datetime(df["period"], format="%Y-%m-%dT%H").dt.tz_localize("UTC").dt.tz_convert("America/Mexico_City").dt.tz_localize(None)
        return a_matriz_horaria(pd.Series(df[col].values, index=t).sort_index().groupby(level=0).last())
    rango = dict(frecuencia="hourly", ini="2023-11-01T00", fin="2025-07-01T23")
    out = {"demanda": ercot_demanda("D")}
    fu = eia("electricity/rto/fuel-type-data", "ercot_combustibles", facets={"respondent": "ERCO", "fueltype": ["WND", "SUN", "NG"]}, **rango)
    for f, n in (("WND", "eolica"), ("SUN", "solar"), ("NG", "gas")):
        out[n] = mat(fu[fu.fueltype == f])
    it = eia("electricity/rto/interchange-data", "ercot_intercambio", facets={"fromba": "ERCO", "toba": "CEN"}, **rango)
    out["intercambio_mx"] = mat(it)
    return out


def ercot_rt(anios=(2023, 2024, 2025), puntos=("HB_HUBAVG", "HB_SOUTH", "HB_WEST", "LZ_SOUTH")) -> pd.DataFrame:
    """Precio en tiempo real de ERCOT (settlement point price, 15 min) promediado por hora, en hora de México.
    Reporte público NP6-785 (Historical RTM Load Zone and Hub Prices), un zip anual sin registro."""
    import io, zipfile
    procesado = RAW / f"ercot_rt_{'_'.join(map(str, anios))}.parquet"            # leer los Excel tarda minutos
    if procesado.exists():
        return pd.read_parquet(procesado)
    partes, docs = [], None
    for a in anios:
        f = RAW / f"ercot_rt_{a}.zip"
        if not f.exists():
            if docs is None:
                lista = _get("https://www.ercot.com/misapp/servlets/IceDocListJsonWS", {"reportTypeId": 13061}).json()["ListDocsByRptTypeRes"]["DocumentList"]
                docs = {d["Document"]["FriendlyName"]: d["Document"]["DocID"] for d in lista}
            RAW.mkdir(parents=True, exist_ok=True)
            f.write_bytes(_get("https://www.ercot.com/misdownload/servlets/mirDownload", {"doclookupId": docs[f"RTMLZHBSPP_{a}"]}, timeout=300).content)
        z = zipfile.ZipFile(f)
        for nombre in z.namelist():
            for hoja in pd.read_excel(io.BytesIO(z.read(nombre)), sheet_name=None).values():
                partes.append(hoja[hoja["Settlement Point Name"].isin(puntos)])
    df = pd.concat(partes)
    # Delivery Hour 1..24 en hora central de Texas (con horario de verano; DSTFlag 'Y' marca la hora repetida)
    t = pd.to_datetime(df["Delivery Date"]) + pd.to_timedelta(df["Delivery Hour"].astype(int) - 1, unit="h")
    t = t.dt.tz_localize("America/Chicago", ambiguous=(df["Repeated Hour Flag"].astype(str).str.upper() == "N").values, nonexistent="shift_forward")
    df = df.assign(t=t.dt.tz_convert("America/Mexico_City").dt.tz_localize(None))
    out = df.pivot_table(index="t", columns="Settlement Point Name", values="Settlement Point Price", aggfunc="mean").sort_index()
    out.columns = [str(x) for x in out.columns]
    out.to_parquet(procesado)
    return out


def estados_operativos() -> pd.Series:
    """Estados operativos del SIN (1 = alerta, 2 = emergencia) por fecha, de `data/external/estados_operativos.csv`.
    Tabla armada a mano con comunicados de CENACE citados en prensa; cada fila trae su fuente. Si un día tuvo alerta y
    emergencia se queda el más grave."""
    df = pd.read_csv(ROOT / "data/external/estados_operativos.csv", parse_dates=["fecha"], comment="#")
    nivel = df["estado"].map({"alerta": 1, "emergencia": 2})
    return nivel.groupby(df["fecha"]).max().sort_index()


FRED_SERIES = {"hh_spot": "DHHNGSP", "wti": "DCOILWTICO", "brent": "DCOILBRENTEU", "usdmxn": "DEXMXUS",
               "diesel_golfo": "DDFUELUSGULF", "gasolina_ny": "DGASNYH", "combustoleo_ny": "DHOILNYH"}
EIA_INV_RETRASO = 6   # inventarios: semana que termina el viernes, se publica el jueves siguiente (periodo + 6 días)


def combustibles() -> dict:
    """Series diarias de FRED (fecha de observación = fecha de negociación)."""
    return {k: fred(sid) for k, sid in FRED_SERIES.items()}


def futuros_gas() -> pd.DataFrame:
    """Contratos de futuros de gas natural 1..4 (EIA, diario), columnas RNGC1..RNGC4."""
    df = eia("natural-gas/pri/fut", "gas_futuros", facets={"series": ["RNGC1", "RNGC2", "RNGC3", "RNGC4"]})
    out = df.pivot_table(index="period", columns="series", values="value")
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def inventarios_gas() -> pd.Series:
    """Gas en almacenamiento subterráneo, Lower 48, total (bcf). Índice = fin de semana medida (viernes)."""
    df = eia("natural-gas/stor/wkly", "gas_inventarios", frecuencia="weekly",
             facets={"series": ["NW2_EPG0_SWO_R48_BCF"]})
    s = df.set_index(pd.to_datetime(df["period"]))["value"].sort_index()
    s.name = "inv_gas"
    return s


def ercot_demanda(tipo: str = "DF") -> pd.DataFrame:
    """Demanda (D) o pronóstico de demanda del día anterior (DF) de ERCOT, EIA-930; matriz días × 24 en hora de México.
    EIA reporta `period` en UTC; se convierte a America/Mexico_City (UTC−6 sin horario de verano)."""
    df = eia("electricity/rto/region-data", f"ercot_{tipo}", frecuencia="hourly", facets={"respondent": "ERCO", "type": tipo},
             ini="2023-11-01T00", fin="2025-07-01T23")
    t = pd.to_datetime(df["period"], format="%Y-%m-%dT%H").dt.tz_localize("UTC").dt.tz_convert("America/Mexico_City").dt.tz_localize(None)
    return a_matriz_horaria(pd.Series(df["value"].values, index=t).sort_index().groupby(level=0).last())
