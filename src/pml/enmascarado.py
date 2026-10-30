"""Modelo enmascarado de la serie ("tipo BERT"): aprende a rellenar huecos del MTR.

Al pronosticar el día d la situación es literalmente un hueco: el MTR se conoce hasta d−8, falta de d−7 a d (192 h) y el
MDA se conoce hasta d. En lugar de plantear una regresión del día d, el modelo aprende a reconstruir tramos ocultos de la
serie horaria a partir de lo que queda visible.

Estructura:
  - entrada: ventana de L días de dos canales (MTR y MDA en la escala asinh del proyecto) más dos canales de visibilidad;
  - parches de 24 h que avanzan de 12 en 12 h, así que cada hora cae en dos parches (salvo los bordes). Cada parche se
    convierte en un token; un transformer pequeño los relaciona; cada token reconstruye sus 24 horas y las horas que
    caen en dos parches se promedian (los "fragmentos que se traslapan");
  - pre-entrenamiento auto-supervisado sobre toda la historia conocida: ventanas que terminan en cualquier hora, con
    tramos de 1 a 8 días del MTR ocultos (la mitad de las veces, justo los 8 días finales, como en la operación). La
    pérdida es MAE sólo en lo oculto;
  - variantes: "solo_mtr" pre-entrena sin MDA (aprende cómo se comporta la serie del MTR) y "conjunto" con MDA visible,
    apagado al azar o recortado a sus últimos k días;
  - ajuste semanal con la máscara de operación y fragmentos aleatorios del MDA;
  - al pronosticar se generan varias reconstrucciones (contextos de 14, 21 y 28 días × fragmentos del MDA de 2 días,
    8 días o completo) y se promedian.

Reglas de tiempo: para un bloque que empieza en d0, todo el entrenamiento usa ventanas que terminan a más tardar al final
de d0−8, así que ningún MTR posterior entra. Pre-entrenamiento cada `cada_pre` días; ajuste cada 7.
"""
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import features2 as f2
from .dl import _torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/dl_cache"
LAG_MTR = 8
PASO = 12                                     # avance de los parches (h); con parches de 24 h cada hora cae en dos


@dataclass
class HPm:
    """Hiperparámetros del modelo enmascarado (tamaño, regularización, épocas de pre-entrenamiento y de ajuste,
    variante)."""
    variante: str = "conjunto"               # 'conjunto' | 'solo_mtr'
    d_model: int = 64
    capas: int = 2
    cabezas: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    wd: float = 1e-4
    L: int = 28                              # días de contexto en entrenamiento
    epocas_pre: int = 25
    epocas_ajuste: int = 8
    lr_ajuste: float = 3e-4
    batch: int = 64
    seeds: int = 2
    cada_pre: int = 28
    contextos: tuple = (14, 21, 28)
    fragmentos: tuple = (2, 8, 0)            # días visibles del MDA al final de la ventana (0 = todo)
    residual: bool = False                   # True: la salida se suma al MDA visible (pronostica el spread)
    ajustar: bool = True                     # False: se usa el modelo pre-entrenado tal cual (con 'solo_mtr' = puro MTR)
    nodos_red: bool = False                  # True: el pre-entrenamiento también usa los nodos de la red (notebook 06)
    troncal: bool = False                    # con nodos_red: los 64 nodos con datos (31 + resto de la troncal de 400 kV)

    def clave(self):
        """Texto que identifica la configuración en el nombre del archivo de caché."""
        d = {k: v for k, v in self.__dict__.items() if not (k == "troncal" and not v)}   # claves de caché previas intactas
        return "|".join(f"{k}={v}" for k, v in sorted(d.items()))


# ------------------------------------------------------------------------------------------------------ datos
def series(c: f2.Ctx):
    """MTR y MDA horarios en la escala del proyecto, con el índice de días."""
    Yt, Xt = f2._t(c)
    dias = Yt.index.intersection(Xt.index)
    m = Yt.loc[dias].values.ravel().astype("float32"); a = Xt.loc[dias].values.ravel().astype("float32")
    return dias, m, a


def series_red(c: f2.Ctx, D: pd.DatetimeIndex, troncal=False):
    """MTR y MDA de cada nodo de la red, alineados a los días de Monclova y en la misma escala."""
    from dataclasses import replace
    from . import red
    Ym, Xm = red.matrices_nodo("MTR", troncal=troncal), red.matrices_nodo("MDA", troncal=troncal)
    out = []
    for n in Ym:
        Yt, Xt = f2._t(replace(c, Y=Ym[n].reindex(D), X=Xm[n].reindex(D)))
        out.append((Yt.reindex(D).values.ravel().astype("float32"), Xt.reindex(D).values.ravel().astype("float32")))
    return out


def _ventana(m, a, fin_h, L_h, oculto_desde_h, mda_desde_h):
    """Ventana [fin_h − L_h, fin_h) con 4 canales: MTR visible, MDA visible y sus banderas de visibilidad.
    El MTR es visible antes de `oculto_desde_h` (hora absoluta) y el MDA desde `mda_desde_h` hasta el final."""
    ini = fin_h - L_h
    t = np.arange(ini, fin_h)
    vm = (t < oculto_desde_h) & ~np.isnan(m[t])
    va = (t >= mda_desde_h) & ~np.isnan(a[t])
    X = np.stack([np.where(vm, m[t], 0.0), np.where(va, a[t], 0.0), vm.astype("float32"), va.astype("float32")], 0)
    return X.astype("float32"), np.nan_to_num(m[t]).astype("float32"), (~vm & ~np.isnan(m[t])).astype("float32")


def _tiempo(fin_h, L_h):
    """Hora del día y día de la semana del inicio de cada parche (sin/cos), para que el modelo sepa dónde está."""
    n = (L_h - 24) // PASO + 1
    ini = fin_h - L_h + PASO * np.arange(n)
    h = (ini % 24) / 24.0; dw = ((ini // 24 + 0) % 7) / 7.0          # día 0 del arreglo = primer día de la serie
    return np.stack([np.sin(2 * np.pi * h), np.cos(2 * np.pi * h), np.sin(2 * np.pi * dw), np.cos(2 * np.pi * dw)], 1).astype("float32")


# ------------------------------------------------------------------------------------------------------ red
def _red(torch, nn, hp: HPm, max_tok):
    class Red(nn.Module):
        """Transformer pequeño sobre parches de 24 horas: recibe MTR, MDA y la máscara, y reconstruye el MTR de las
        horas ocultas."""
        def __init__(s):
            super().__init__()
            s.emb = nn.Linear(24 * 4 + 4, hp.d_model)
            s.pos = nn.Parameter(torch.zeros(max_tok, hp.d_model))   # posición contada desde el final de la ventana
            capa = nn.TransformerEncoderLayer(hp.d_model, hp.cabezas, 2 * hp.d_model, hp.dropout, batch_first=True)
            s.enc = nn.TransformerEncoder(capa, hp.capas)
            s.cab = nn.Linear(hp.d_model, 24)

        def forward(s, X, T):
            # X: n × 4 × L_h ; T: n × tok × 4
            n, _, L_h = X.shape
            tok = X.unfold(2, 24, PASO)                              # n × 4 × tok × 24
            ntok = tok.shape[2]
            z = torch.cat([tok.permute(0, 2, 1, 3).reshape(n, ntok, 96), T], 2)
            z = s.emb(z) + s.pos[-ntok:]
            y = s.cab(s.enc(z))                                      # n × tok × 24
            # reconstrucción horaria: promedio de los parches que cubren cada hora
            idx = (PASO * torch.arange(ntok)[:, None] + torch.arange(24)[None]).reshape(-1)
            suma = torch.zeros(n, L_h).index_add_(1, idx, y.reshape(n, -1))
            cuenta = torch.zeros(L_h).index_add_(0, idx, torch.ones(idx.shape[0]))
            out = suma / cuenta
            if hp.residual:
                out = out + X[:, 1] * X[:, 3]                          # MDA visible como ancla
            return out
    return Red()


def _tensor(torch, a):
    return torch.tensor(np.asarray(a, dtype=float).tolist(), dtype=torch.float32)   # sin puente numpy↔torch (torch 2.2)


def _entrena(red, torch, muestras, epocas, lr, wd, batch, rng):
    """muestras: lista de (X, objetivo, peso, T). Pérdida MAE ponderada (el peso marca lo oculto)."""
    if not muestras:
        return red
    opt = torch.optim.AdamW(red.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epocas):
        red.train()
        orden = rng.permutation(len(muestras))
        for i in range(0, len(orden), batch):
            b = [muestras[j] for j in orden[i:i + batch]]
            X = _tensor(torch, np.stack([x[0] for x in b])); y = _tensor(torch, np.stack([x[1] for x in b]))
            w = _tensor(torch, np.stack([x[2] for x in b])); T = _tensor(torch, np.stack([x[3] for x in b]))
            loss = ((red(X, T) - y).abs() * w).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
    return red


def _muestras_pre(m, a, corte_h, hp: HPm, rng, inicio_h):
    """Ventanas de pre-entrenamiento que terminan en cualquier hora hasta `corte_h` (todo su MTR es conocido)."""
    L_h = hp.L * 24
    finales = np.arange(inicio_h + L_h, corte_h + 1, 3)            # cada 3 h, para no repetir casi la misma ventana
    out = []
    for fin in finales:
        if rng.random() < 0.5:
            hueco = (fin - 192, fin)                               # como en la operación
        else:
            k = int(rng.integers(1, 9)) * 24
            ini = int(rng.integers(fin - L_h + 24, fin - k + 1)); hueco = (ini, ini + k)
        if hp.variante == "solo_mtr":
            mda_desde = fin + 1                                    # MDA nunca visible
        else:
            u = rng.random()
            mda_desde = fin + 1 if u < 0.3 else (fin - 24 * int(rng.integers(1, 9)) if u < 0.6 else 0)
        X, y, _ = _ventana(m, a, fin, L_h, oculto_desde_h=10 ** 9, mda_desde_h=mda_desde)
        t = np.arange(fin - L_h, fin)
        oc = (t >= hueco[0]) & (t < hueco[1])
        X[0, oc] = 0.0; X[2, oc] = 0.0
        w = (oc & ~np.isnan(m[t])).astype("float32")
        out.append((X, y, w, _tiempo(fin, L_h)))
    return out


def _muestras_ajuste(m, a, corte_dia, hp: HPm, rng, inicio_dia):
    """Ventanas con la máscara de operación: terminan al final de un día e ≤ corte, MTR oculto en sus últimos 8 días."""
    L_h = hp.L * 24
    out = []
    for e in range(max(inicio_dia, hp.L - 1), corte_dia + 1):
        fin = (e + 1) * 24
        u = rng.random()
        mda_desde = 0 if u < 0.5 else fin - 24 * int(rng.integers(1, 9))
        X, y, w = _ventana(m, a, fin, L_h, oculto_desde_h=fin - 192, mda_desde_h=mda_desde)
        w[-24:] *= 2.0                                             # el día que se pronostica pesa doble
        out.append((X, y, w, _tiempo(fin, L_h)))
    return out


def _predice(red, torch, m, a, dia, hp: HPm, solo_mtr=False):
    """Promedio de las reconstrucciones del día `dia` (índice) con varios contextos y fragmentos del MDA."""
    fin = (dia + 1) * 24
    preds = []
    red.eval()
    with torch.no_grad():
        for L in hp.contextos:
            for k in ((0,) if solo_mtr else hp.fragmentos):
                mda_desde = fin + 1 if solo_mtr else (0 if k == 0 else fin - 24 * k)
                X, _, _ = _ventana(m, a, fin, L * 24, oculto_desde_h=fin - 192, mda_desde_h=mda_desde)
                p = red(_tensor(torch, X[None]), _tensor(torch, _tiempo(fin, L * 24)[None]))
                preds.append(np.array(p[0, -24:].tolist()))
    return np.mean(preds, 0), np.array(preds)


def _bloque_pre(corte_dia, m, a, hp: HPm, seed, inicio_dia, extra=()):
    torch, nn = _torch(); torch.set_num_threads(1); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    red = _red(torch, nn, hp, (max(hp.contextos + (hp.L,)) * 24 - 24) // PASO + 1)
    muestras = _muestras_pre(m, a, (corte_dia + 1) * 24, hp, rng, inicio_dia * 24)
    if extra:                                                      # otros nodos: en total, 3 veces las ventanas de Monclova
        frac = 3.0 / len(extra)
        for me, ae in extra:
            muestras += [x for x in _muestras_pre(me, ae, (corte_dia + 1) * 24, hp, rng, inicio_dia * 24) if rng.random() < frac]
    _entrena(red, torch, muestras, hp.epocas_pre, hp.lr, hp.wd, hp.batch, rng)
    return {k: v.clone() for k, v in red.state_dict().items()}


def _bloque_ajuste(d0, d1, estado, m, a, hp: HPm, seed, inicio_dia, multiple=True):
    torch, nn = _torch(); torch.set_num_threads(1); torch.manual_seed(seed + 1000)
    rng = np.random.default_rng(seed + 1000)
    red = _red(torch, nn, hp, (max(hp.contextos + (hp.L,)) * 24 - 24) // PASO + 1)
    red.load_state_dict(estado)
    if hp.ajustar:
        _entrena(red, torch, _muestras_ajuste(m, a, d0 - LAG_MTR, hp, rng, inicio_dia), hp.epocas_ajuste, hp.lr_ajuste, hp.wd, hp.batch, rng)
    solo = (hp.variante == "solo_mtr") and not hp.ajustar
    out = {}
    for d in range(d0, d1 + 1):
        p, todas = _predice(red, torch, m, a, d, hp, solo_mtr=solo)
        out[d] = (p, todas[-1] if not solo else todas[-1])              # promedio y una sola reconstrucción (28 días, MDA completo)
    return out


def correr(c: f2.Ctx, hp: HPm, dias, nombre, n_jobs=-1):
    """Backtest: pre-entrena cada `cada_pre` días y ajusta cada 7. Devuelve dict con 'prom' (promedio de
    reconstrucciones) y 'una' (una sola reconstrucción), días × 24 en la escala del proyecto. Caché en disco."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dias = pd.DatetimeIndex(dias)
    clave = hashlib.md5(f"{hp.clave()}|{c.escala}|{c.med:.4f}|{dias[0]}|{dias[-1]}".encode()).hexdigest()[:12]
    f = CACHE / f"enm_{nombre}_{clave}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    D, m, a = series(c)
    pos = {d: i for i, d in enumerate(D)}
    inicio = 0
    bloques = [(pos[dias[i]], pos[dias[min(i + 6, len(dias) - 1)]]) for i in range(0, len(dias), 7)]
    cortes = sorted({b[0] - (b[0] - bloques[0][0]) % hp.cada_pre - LAG_MTR for b in bloques})
    extra = series_red(c, D, hp.troncal) if hp.nodos_red else ()
    estados = Parallel(n_jobs=n_jobs)(delayed(_bloque_pre)(k, m, a, hp, s, inicio, extra) for k in cortes for s in range(hp.seeds))
    E = {(k, s): estados[i * hp.seeds + s] for i, k in enumerate(cortes) for s in range(hp.seeds)}
    corte_de = lambda d0: d0 - (d0 - bloques[0][0]) % hp.cada_pre - LAG_MTR
    res = Parallel(n_jobs=n_jobs)(delayed(_bloque_ajuste)(d0, d1, E[(corte_de(d0), s)], m, a, hp, s, inicio) for d0, d1 in bloques for s in range(hp.seeds))
    prom, una = {}, {}
    for r in res:
        for d, (p, u) in r.items():
            prom.setdefault(d, []).append(p); una.setdefault(d, []).append(u)
    idx = [D[d] for d in sorted(prom)]
    out = {"prom": pd.DataFrame([np.mean(prom[d], 0) for d in sorted(prom)], index=idx, columns=range(1, 25)),
           "una": pd.DataFrame([np.mean(una[d], 0) for d in sorted(una)], index=idx, columns=range(1, 25))}
    f.write_bytes(pickle.dumps(out))
    return out
