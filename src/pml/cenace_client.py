"""Descarga de precios del servicio web SW-PML de CENACE.

El servicio entrega el PML (y sus componentes de energía, pérdidas y congestión) por nodo, hora y proceso (MDA o MTR),
pero sólo en tramos de hasta 7 días y hasta 10 nodos por llamada. Este módulo parte el periodo en tramos, guarda cada
respuesta cruda en disco (si ya existe no se vuelve a pedir) y reintenta con espera creciente cuando el servicio falla.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import truststore

truststore.inject_into_ssl()

BASE = "https://ws01.cenace.gob.mx:8082/SWPML/SIM"
# Los 9 NodosP de la zona de carga Monclova (catálogo de CENACE)
NODOS = ["06AEO-115", "06AHM-400", "06FRO-115", "06GRA-115", "06MON-115",
         "06MON-230", "06RGS-115", "06TKS-115", "06XOC-115"]
MAX_DIAS = 7                              # el servicio no acepta tramos más largos
COLS_NUM = ["pml", "pml_ene", "pml_per", "pml_cng"]


def bloques(ini: date, fin: date, max_dias: int = MAX_DIAS):
    """Divide [ini, fin] en tramos consecutivos de a lo más max_dias días."""
    cur = ini
    while cur <= fin:
        end = min(cur + timedelta(days=max_dias - 1), fin)
        yield cur, end
        cur = end + timedelta(days=1)


def _url(sistema, proceso, nodos, ini: date, fin: date) -> str:
    """URL del servicio: sistema (SIN), proceso (MDA/MTR), nodos separados por coma, fechas y formato."""
    return (f"{BASE}/{sistema}/{proceso}/{','.join(nodos)}/"
            f"{ini:%Y/%m/%d}/{fin:%Y/%m/%d}/JSON")


def fetch_bloque(proceso: str, ini: date, fin: date, raw_dir: Path,
                 nodos=NODOS, sistema="SIN", reintentos=5, timeout=90) -> dict:
    """Un tramo de fechas. Si la respuesta ya está guardada en `raw_dir`, la lee de ahí; si no, la pide al servicio
    (con hasta `reintentos` intentos y espera creciente) y la guarda junto con la hora de descarga."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    f = raw_dir / f"{proceso}_{ini:%Y%m%d}_{fin:%Y%m%d}.json"
    if f.exists():
        return json.loads(f.read_text())
    url = _url(sistema, proceso, nodos, ini, fin)
    ultimo = None
    for i in range(reintentos):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                j = r.json()
                if str(j.get("status", "")).strip() == "OK":
                    j["_descargado_utc"] = datetime.now(timezone.utc).isoformat()
                    f.write_text(json.dumps(j))
                    return j
                ultimo = f"status={j.get('status')}"
            else:
                ultimo = f"HTTP {r.status_code}: {r.text[:120]}"
                if r.status_code in (400, 401, 404):  # no tiene caso reintentar
                    break
        except (requests.RequestException, ValueError) as e:
            ultimo = repr(e)
        time.sleep(min(2 ** i, 30))
    raise RuntimeError(f"{proceso} {ini}..{fin} falló: {ultimo}")


def a_dataframe(j: dict) -> pd.DataFrame:
    """JSON de la API -> DataFrame largo con tipos numéricos."""
    rows = [dict(nodo=r["clv_nodo"], proceso=j["proceso"], **v)
            for r in j.get("Resultados", []) for v in r["Valores"]]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["hora"] = df["hora"].astype(int)
    df[COLS_NUM] = df[COLS_NUM].apply(pd.to_numeric)
    return df


def descargar(proceso: str, ini: date, fin: date, raw_dir: Path, **kw) -> pd.DataFrame:
    """Todo el periodo [ini, fin] de un proceso, en formato largo y ordenado. `kw` pasa los nodos u otras opciones."""
    partes = [a_dataframe(fetch_bloque(proceso, a, b, raw_dir, **kw))
              for a, b in bloques(ini, fin)]
    df = pd.concat(partes, ignore_index=True)
    return df.sort_values(["proceso", "nodo", "fecha", "hora"]).reset_index(drop=True)
