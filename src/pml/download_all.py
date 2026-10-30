"""Descarga completa de los precios de Monclova (MTR y MDA, del 2023-12-18 al 2025-06-30).

Se empieza dos semanas antes de 2024 para que los primeros días tengan sus rezagos. El resultado queda en
data/processed/pml_mtr.parquet y pml_mda.parquet. Uso: python src/pml/download_all.py
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pml.cenace_client import descargar

ROOT = Path(__file__).resolve().parents[2]
INI, FIN = date(2023, 12, 18), date(2025, 6, 30)  # 14 días previos para rezagos

if __name__ == "__main__":
    for p in ("MTR", "MDA"):
        df = descargar(p, INI, FIN, ROOT / "data/raw")
        df.to_parquet(ROOT / f"data/processed/pml_{p.lower()}.parquet", index=False)
        print(p, df.shape, df.fecha.min().date(), df.fecha.max().date(), flush=True)
