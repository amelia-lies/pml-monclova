"""Nodos de la red fuera de Monclova (notebook 06): se usan como ejemplos extra para entrenar, nunca para evaluar.

La componente de energía del PML es la misma en todos los nodos del SIN, así que estos nodos no dicen nada nuevo de
la energía; lo que aportan es su congestión (y pérdidas), que muestra hacia dónde fluye la potencia y qué corredores
se saturan. Claves tomadas del Catálogo de NodosP de CENACE (v2026-10-23); se verifica que cada una traiga datos.
Mismas reglas de tiempo que Monclova: MDA conocido hasta d, MTR hasta d−8.
"""
from datetime import date
from pathlib import Path

import pandas as pd

from .cenace_client import descargar

ROOT = Path(__file__).resolve().parents[2]
INI, FIN = date(2023, 12, 18), date(2025, 6, 30)

# clave: (región, papel en la red)
NODOS_RED = {
    # corredor del carbón y Coahuila
    "06REC-400": ("Noreste", "Río Escondido, carbón 1.4 GW"),
    "06CBD-400": ("Noreste", "Carbón II, carbón 1.4 GW"),
    "06NAV-138": ("Noreste", "Nava, Piedras Negras"),
    "06RAP-400": ("Noreste", "Ramos Arizpe, Saltillo 400 kV"),
    "06SAL-115": ("Noreste", "Saltillo, carga"),
    # Monterrey y Tamaulipas
    "06ESC-400": ("Noreste", "Escobedo, Monterrey"),
    "06HUO-400": ("Noreste", "Huinalá, Monterrey"),
    "06VDG-230": ("Noreste", "Villa de García, Monterrey"),
    "06AER-400": ("Noreste", "Aeropuerto, Reynosa (ciclos combinados)"),
    "06CUF-230": ("Noreste", "Cumbres Frontera, enlace con Texas"),
    "06ALT-400": ("Noreste", "Altamira, gas y GNL"),
    # Norte
    "05MVI-400": ("Norte", "Laguna 400 kV, enlace Noreste-Norte"),
    "05AND-230": ("Norte", "Andalucía, Torreón"),
    "05ENO-230": ("Norte", "Encino, Chihuahua"),
    # Centro (CDMX)
    "01VMA-400": ("Central", "Valle de México 400 kV"),
    "01TOP-230": ("Central", "Topilejo, CDMX sur"),
    "01TUL-400": ("Central", "Tula, generación del centro"),
    # Occidente y Bajío
    "03TIM-400": ("Occidental", "Tierra Mojada, Guadalajara"),
    "03ATQ-230": ("Occidental", "Atequiza, Guadalajara"),
    "03SAU-400": ("Occidental", "El Sauz, Querétaro"),
    "03SLM-400": ("Occidental", "Salamanca, Bajío"),
    "03APT-230": ("Occidental", "Aguascalientes"),
    "03MNZ-400": ("Occidental", "Manzanillo, GNL"),
    # Oriente y sureste
    "02LAV-400": ("Oriental", "Laguna Verde, nuclear"),
    "02MMT-400": ("Oriental", "Chicoasén, hidro del Grijalva"),
    "02TUV-400": ("Oriental", "Tuxpan"),
    "02JUD-230": ("Oriental", "Juchitán, eólica del Istmo"),
    "02PDM-115": ("Oriental", "Puebla, carga"),
    # Noroeste y Peninsular
    "04HAE-230": ("Noroeste", "Hermosillo, solar"),
    "04ECD-400": ("Noroeste", "Empalme, ciclo combinado"),
    "08MDP-230": ("Peninsular", "Mérida, déficit de gas"),
}
# Resto de los nodos de 400 kV del SIN (catálogo de NodosP): la troncal completa, sin Baja California ni BCS (otro
# sistema, otro precio de energía) y sin 06AHM-400, que es uno de los 9 nodos de Monclova. Solo para entrenar.
NODOS_400 = {
    "06CDU-400": ("Noreste", "Dulces Nombres, Monterrey"), "06TRQ-400": ("Noreste", "Ternium Pesquería"),
    "06ATP-400": ("Noreste", "Anáhuac Potencia, Reynosa"), "06MES-400": ("Noreste", "Las Mesas, Tampico"),
    "06PAE-400": ("Noreste", "Puerto Altamira"), "06FAI-400": ("Noreste", "El Fraile, Monterrey"),
    "06LLC-400": ("Noreste", "Llera de Canales"), "06TEG-400": ("Noreste", "Termoeléctrica del Golfo"),
    "06TEP-400": ("Noreste", "Termoeléctrica Peñoles"), "06HYM-400": ("Noreste", "Hylsa, Monterrey"),
    "06TMS-400": ("Noreste", "Tres Mesas"), "06PGE-400": ("Noreste", "Presa Grande, Monterrey"),
    "06SCP-400": ("Noreste", "San Carlos Potencia"), "06TNM-400": ("Noreste", "Ternium Maniobras"),
    "03PNQ-400": ("Occidental", "Peñasquito, Zacatecas"), "03MFP-400": ("Occidental", "Fotovoltaico Potosí"),
    "03AGM-400": ("Occidental", "Aguamilpa, hidro"), "03PMY-400": ("Occidental", "Primero de Mayo, Zacatecas"),
    "03EVI-400": ("Occidental", "Vientos del Altiplano, eólica"), "03CJN-400": ("Occidental", "El Cajón, hidro"),
    "03LYE-400": ("Occidental", "La Yesca, hidro"), "03POM-400": ("Occidental", "Potrero, Aguascalientes"),
    "03MPM-400": ("Occidental", "Pachamama, Aguascalientes"),
    "02TTE-400": ("Oriental", "Tres Estrellas, Poza Rica"), "02MID-400": ("Oriental", "Minatitlán"),
    "02OJP-400": ("Oriental", "Ojo de Agua, Orizaba"), "02CYR-400": ("Oriental", "CYCNA, Puebla"),
    "02CBE-400": ("Oriental", "Ciclo Combinado Centro, Morelos"), "02ANG-400": ("Oriental", "Angostura, hidro"),
    "02MPS-400": ("Oriental", "Malpaso, hidro"),
    "01LCP-400": ("Central", "Lázaro Cárdenas Potencia"), "01PIT-400": ("Central", "Pitirera, Lázaro Cárdenas"),
    "04TPD-400": ("Noroeste", "Topolobampo II"),
    "08RMV-400": ("Peninsular", "Riviera Maya-Valladolid"),
}
GRUPO = 10                                         # nodos por llamada al SW-PML

# Agrupación para las features: con 17 meses no podemos darle al modelo 31 nodos × 24 horas; lo resumimos por corredor.
GRUPOS_RED = {
    "carbon": ["06REC-400", "06CBD-400", "06NAV-138"],
    "saltillo": ["06RAP-400", "06SAL-115"],
    "monterrey": ["06ESC-400", "06HUO-400", "06VDG-230"],
    "tamaulipas": ["06AER-400", "06CUF-230", "06ALT-400"],
    "norte": ["05MVI-400", "05AND-230", "05ENO-230"],
    "centro": ["01VMA-400", "01TOP-230", "01TUL-400"],
    "occidente": ["03TIM-400", "03ATQ-230", "03SAU-400", "03SLM-400", "03APT-230", "03MNZ-400"],
    "oriente": ["02LAV-400", "02MMT-400", "02TUV-400", "02JUD-230", "02PDM-115"],
    "noroeste": ["04HAE-230", "04ECD-400"],
    "peninsular": ["08MDP-230"],
}


_CORREDOR_400 = {"monterrey": ["06CDU-400", "06TRQ-400", "06FAI-400", "06HYM-400", "06PGE-400", "06TNM-400"],
                 "tamaulipas": ["06ATP-400", "06MES-400", "06PAE-400", "06LLC-400", "06TEG-400", "06TEP-400", "06TMS-400", "06SCP-400"]}
_REGION_CORREDOR = {"Occidental": "occidente", "Oriental": "oriente", "Central": "centro", "Noroeste": "noroeste", "Peninsular": "peninsular"}


def corredores(troncal=False) -> dict:
    """{nodo: corredor} para los nodos de entrenamiento. Los de 400 kV nuevos van al corredor de su región."""
    out = {n: g for g, ns in GRUPOS_RED.items() for n in ns}
    if troncal:
        extra = {n: g for g, ns in _CORREDOR_400.items() for n in ns}
        out.update({n: extra.get(n, _REGION_CORREDOR.get(r)) for n, (r, _) in NODOS_400.items()})
    return out


def descargar_red(proceso: str) -> pd.DataFrame:
    """Descarga los 31 nodos elegidos a mano, de 10 en 10 (el límite del servicio)."""
    claves = list(NODOS_RED)
    partes = []
    for i in range(0, len(claves), GRUPO):
        g = claves[i:i + GRUPO]
        partes.append(descargar(proceso, INI, FIN, ROOT / f"data/raw/red/g{i // GRUPO}", nodos=g))
    return pd.concat(partes, ignore_index=True)


def descargar_400(proceso: str) -> pd.DataFrame:
    """Descarga el resto de los nodos de 400 kV del SIN, de 10 en 10."""
    claves = list(NODOS_400)
    partes = [descargar(proceso, INI, FIN, ROOT / f"data/raw/red/t{i // GRUPO}", nodos=claves[i:i + GRUPO])
              for i in range(0, len(claves), GRUPO)]
    return pd.concat(partes, ignore_index=True)


def cargar(proceso: str, troncal=False) -> pd.DataFrame:
    """Nodos de la red (31). Con `troncal`, también el resto de los nodos de 400 kV (64 con datos: 06TNM-400 no tiene precios publicados)."""
    f = ROOT / f"data/processed/red_{proceso.lower()}.parquet"
    if not f.exists():
        descargar_red(proceso).to_parquet(f, index=False)
    df = pd.read_parquet(f)
    if troncal:
        f4 = ROOT / f"data/processed/red400_{proceso.lower()}.parquet"
        if not f4.exists():
            descargar_400(proceso).to_parquet(f4, index=False)
        df = pd.concat([df, pd.read_parquet(f4)], ignore_index=True)
    return df


def matrices(col="pml_cng") -> dict:
    """{proceso: {corredor: días × 24}} con el promedio de `col` en los nodos de cada corredor."""
    from .data import matriz
    out = {}
    for p in ("MDA", "MTR"):
        df = cargar(p)
        out[p] = {g: matriz(df[df.nodo.isin(n)], col) for g, n in GRUPOS_RED.items()}
    return out


def matrices_nodo(proceso: str, col="pml", troncal=False) -> dict:
    """{nodo: días × 24} para cada nodo de la red (entrenamiento con más nodos). Con `troncal`, los 64 con datos."""
    from .data import matriz
    df = cargar(proceso, troncal)
    nodos = list(NODOS_RED) + (list(NODOS_400) if troncal else [])
    return {n: matriz(df, col, n) for n in nodos if (df.nodo == n).any()}


# ------------------------------------------------------------------------------------------- ubicación (mapas)
MONCLOVA = ["06AEO-115", "06AHM-400", "06FRO-115", "06GRA-115", "06MON-115", "06MON-230", "06RGS-115", "06TKS-115", "06XOC-115"]


def _sin_acentos(t: str) -> str:
    """Mayúsculas sin acentos, para comparar nombres de estados y municipios."""
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFD", str(t).upper()) if unicodedata.category(ch) != "Mn")


# Cabeceras municipales que el geocodificador no ubica en el estado correcto (coordenadas aproximadas de la cabecera).
_UBICACION_MANUAL = {"05ENO-230": (28.60, -105.88),    # Aquiles Serdán, Chihuahua
                     "06FAI-400": (25.93, -100.37),    # El Carmen, Nuevo León
                     "02LAV-400": (19.62, -96.73)}     # Alto Lucero de Gutiérrez Barrios, Veracruz


def ubicaciones() -> pd.DataFrame:
    """Latitud y longitud aproximadas de cada nodo (Monclova y la red), para los mapas. El catálogo de CENACE da el
    municipio y el estado de cada nodo; se ubica la cabecera municipal con el geocodificador público de Open-Meteo y se
    guarda en data/external/geo/nodos_ubicacion.csv. Es una ubicación aproximada (la del municipio, no la subestación)."""
    import json, os, urllib.parse, urllib.request
    f = ROOT / "data/external/geo/nodos_ubicacion.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0)
    xl = [x for x in os.listdir(ROOT) if x.endswith(".xlsx") and "NodosP" in x][0]
    cat = pd.read_excel(ROOT / xl, sheet_name=0, header=1)
    cat.columns = [str(c).strip() for c in cat.columns]
    cat["CLAVE"] = cat["CLAVE"].str.strip()
    nodos = MONCLOVA + list(NODOS_RED) + list(NODOS_400)
    filas = {}
    for n in nodos:
        if not (cat.CLAVE == n).any():                    # 06TNM-400: no está en el catálogo local (tampoco tiene precios)
            continue
        r = cat[cat.CLAVE == n].iloc[0]
        mun, edo = r["MUNICIPIO (INEGI)"], r["ENTIDAD FEDERATIVA (INEGI)"]
        q = urllib.parse.urlencode({"name": str(mun).title(), "count": 20, "language": "es", "countryCode": "MX"})
        res = json.load(urllib.request.urlopen(f"https://geocoding-api.open-meteo.com/v1/search?{q}", timeout=30)).get("results", [])
        e = _sin_acentos(edo).split(" DE ")[0].split()[0]                       # "COAHUILA DE ZARAGOZA" -> "COAHUILA"
        ok = [x for x in res if e in _sin_acentos(x.get("admin1", ""))]          # sólo resultados del mismo estado
        mejor = max(ok, key=lambda x: (_sin_acentos(x.get("admin2", "")) == _sin_acentos(mun), x.get("population", 0))) if ok else {}
        if n in _UBICACION_MANUAL:                    # el geocodificador no encuentra el municipio en su estado
            mejor = dict(zip(("latitude", "longitude"), _UBICACION_MANUAL[n]))
        filas[n] = {"nombre": r["NOMBRE"], "municipio": mun, "estado": edo, "lat": mejor.get("latitude"), "lon": mejor.get("longitude"),
                    "grupo": "monclova" if n in MONCLOVA else ("31 originales" if n in NODOS_RED else "400 kV nuevos")}
    df = pd.DataFrame.from_dict(filas, orient="index")
    f.parent.mkdir(parents=True, exist_ok=True); df.to_csv(f)
    return df


if __name__ == "__main__":
    for p in ("MDA", "MTR"):
        df = cargar(p, troncal=True)
        print(p, df.shape, df.nodo.nunique(), df.fecha.min().date(), df.fecha.max().date(), flush=True)
