"""Corre en el servidor los experimentos nuevos de la escalera y de qué pierde cada modelo sin cada familia (notebooks 07 y 08).

Uso: python src/pml/servidor.py <config> [<config> ...]      (con `lista` imprime todas las configuraciones)
Una configuración es `modelo:paso` (p. ej. `lear:todas`, `gb:pca`, `tide:pca`) o `imp_lear:<n>` / `imp_gb:<n>`, con <n> el
número de la familia en `todo.FAMILIAS`. Cada una escribe en las mismas cachés que leen los notebooks, así que lo que ya está
calculado no se repite. Si una configuración falla, se imprime el error y se sigue con la siguiente.
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"   # un hilo por proceso de joblib
import pickle
import sys
import time
import traceback
import warnings
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from pml import dl, exp, features2 as f2, multinodo as mn, referencias as rf
from pml.todo import CALENDARIO, ESTADISTICO, EXOGENAS, EXTRA_RED, EXTRA_SIN_MDA, FAMILIAS, K_PC, NUCLEO, TODO

N = int(os.environ.get("PBS_NP", os.cpu_count()))
GRUPOS = {"nodos": NUCLEO, "exogenas": NUCLEO + EXOGENAS, "estadistico": NUCLEO + ESTADISTICO, "todas": TODO, "pca": TODO}
FAM = list(FAMILIAS)
CONFIGS = ([f"lear:{p}" for p in GRUPOS] + [f"learp:{p}" for p in GRUPOS] + [f"gb:{p}" for p in GRUPOS]
           + [f"{m}:{p}" for m in ("tide", "sinmda") for p in ("todas", "pca")]
           + [f"imp_lear:{i}" for i in range(len(FAM))] + [f"imp_gb:{i}" for i in range(len(FAM))] + ["mezclas:todas"])


def gb(c, grupos, dias, archivo, pca=False):
    """GB entrenado con Monclova y los 64 nodos; guarda el spread transformado de Monclova (días × 24)."""
    f = mn.CACHE / archivo
    if f.exists():
        return
    if pca:   # rezagos, MDA y calendario tal cual; el resto en componentes ajustadas con todas las filas apiladas
        L, y = mn.preparar_pc(c, NUCLEO + CALENDARIO, [g for g in grupos if g not in NUCLEO + CALENDARIO], "nucleo", K_PC, troncal=True)
    else:
        L, y = mn.preparar(c, con_red=True, troncal=True, grupos=grupos)
    f.write_bytes(pickle.dumps(mn.correr(L, y, dias, n_jobs=N)))


def correr(cfg, ctx, R):
    modelo, paso = cfg.split(":")
    DIAS, DEV = ctx["DIAS"], ctx["DEV"]
    if modelo in ("lear", "learp"):
        c = ctx["cbn"] if modelo == "lear" else ctx["cln"]
        mn.lear(c, DIAS, grupos=GRUPOS[paso], troncal=True, n_jobs=N, comp="nucleo" if paso == "pca" else None, k=K_PC)
    elif modelo == "gb":
        gb(ctx["cbn"], GRUPOS[paso], DIAS, f"esc_gb_{paso}.pkl", pca=paso == "pca")
    elif modelo in ("tide", "sinmda"):
        # la misma configuración que con 64 nodos en el notebook 06 (TiDE con L1 / 10, que fue la elegida; lote de 256)
        hp = replace(R.hp["tide"], batch=256, l1=R.hp["tide"].l1 / 10) if modelo == "tide" else replace(R.hp["nbeats"], batch=256)
        variante, extra, nombre = ("spread", EXTRA_RED, "tide_troncal") if modelo == "tide" else ("sin_mda", EXTRA_SIN_MDA, "nbeats_troncal")
        dl.correr_dl_red(ctx["cbn"], variante, hp, DIAS, troncal=True, nombre=nombre, n_jobs=N, extra=extra,
                         comp="nucleo" if paso == "pca" else None, k=K_PC)
    elif modelo == "imp_lear":   # todas las variables menos una familia, solo en desarrollo
        mn.lear(ctx["cbn"], DEV, grupos=[g for g in TODO if g not in FAMILIAS[FAM[int(paso)]]], troncal=True, n_jobs=N, nombre="imp_lear")
    elif modelo == "mezclas":   # todas las mezclas de 2 y 3 versiones de la escalera (notebook 10)
        from pml import escalera, mezclas
        F = {"MDA": R.X, **{f"{k} · {p}": P for k, d in escalera.cargar_pasos(R).items() for p, P in d.items() if P is not None}}
        mezclas.evaluar(R.Y, R.X, F, DEV, exp.idx_test(), n_jobs=N)
    elif modelo == "imp_gb":
        gb(ctx["cbn"], [g for g in TODO if g not in FAMILIAS[FAM[int(paso)]]], DEV, f"imp_gb_{paso}.pkl")
    else:
        raise ValueError(cfg)


if __name__ == "__main__":
    pedidas = sys.argv[1:]
    if pedidas == ["lista"]:
        print("\n".join(CONFIGS)); sys.exit()
    assert pedidas and all(p in CONFIGS for p in pedidas), f"configuraciones válidas: {CONFIGS}"
    t0 = time.time()
    c0 = f2.cargar_ctx(); cb, cl = replace(c0, escala=0.5), replace(c0, modo="log")
    ctx = dict(cbn=f2.con_nodos(cb), cln=f2.con_nodos(cl), DIAS=exp.dias_eval(c0), DEV=exp.idx_dev())
    R = rf.cargar() if any(p.split(":")[0] in ("tide", "sinmda", "mezclas") for p in pedidas) else None   # sólo las redes necesitan sus hiperparámetros
    print(f"núcleos: {N} | contexto listo en {time.time() - t0:.0f} s", flush=True)
    for cfg in pedidas:
        t = time.time()
        try:
            correr(cfg, ctx, R)
            print(f"OK    {cfg}  {(time.time() - t) / 60:.1f} min", flush=True)
        except Exception:
            print(f"ERROR {cfg}\n{traceback.format_exc()}", flush=True)
    print(f"FIN {(time.time() - t0) / 60:.1f} min", flush=True)
