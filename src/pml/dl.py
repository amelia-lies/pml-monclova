"""Redes neuronales para el PML día-adelante: DNN de Lago (MLP de 2 capas) y NBEATSx-lite.

Mismas reglas de tiempo que el resto del proyecto: para pronosticar el día d sólo se entrena con días <= d−8 (el MTR más
reciente que ya existe) y las entradas son las de `features2` (rezagos de MTR >= 8 días, MDA hasta el día d, calendario).
Se recalibra cada 7 días (no a diario como en la literatura) por costo; el bloque de 7 días entrena con días <= d0−8.

Variantes (misma red, distinto papel del MDA):
  V1 "sin_mda" : MTR directo, sin ninguna entrada del MDA (techo de aprender sólo del MTR histórico)
  V2 "nivel"   : MTR directo, el MDA entra como covariable más
  V3 "spread"  : se predice MTR − MDA y se suma el MDA (anclaje fuerte, como LEAR)
     "spread_w": lo mismo, con la ventana completa del MDA (d−7 … d, 192 horas) y del error ya publicado (para `dual`)
  V4 "mda_drop": MTR directo; durante el entrenamiento el bloque del MDA se apaga al azar (prob. p) y se avisa con una
                 bandera. Al pronosticar se obtienen dos salidas, con y sin MDA, y se mezclan con un peso w.

Otras formas de plantear la red (TiDE está en el notebook 05; "conv" quedó como idea descartada, ver el README):
  arch "tide"  : TiDE ligero (Das et al. 2023). Un codificador denso resume todo en un código por hora y un decodificador
                 temporal, compartido entre horas, combina ese código con las covariables de esa misma hora (MDA y
                 rezagos de la hora h). Más un atajo lineal de las entradas a la salida.
  arch "conv"  : red con ramas (idea descartada, ver el README): una convolución sobre la ventana de 7 días de MTR para aprender
                 movimientos generales (ciclo diario, tendencia de la semana), una rama pequeña para el MDA y otra para el
                 resto. `cuello` aprieta la rama del MDA a pocas neuronas y `p_ramal` la apaga al azar al entrenar, para que
                 la red no se limite a copiar el MDA. Las dos opciones también valen para la DNN (`mlp`).
  dist "jsu"   : salida distribucional Johnson SU por hora (Marcjasz et al. 2023), entrenada por verosimilitud. El punto
                 es la mediana de la distribución y los cuantiles salen gratis.
"""
import hashlib
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import features2 as f2

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/dl_cache"
LAG_MTR = 8
GRUPOS_V = {"sin_mda": ["G0w", "G1"], "nivel": ["G0w", "G2", "G1"], "spread": ["G0w", "G2", "G1"], "spread_w": ["G0w", "G2w", "G1"], "mda_drop": ["G0w", "G2", "G1"]}


@dataclass
class HP:
    """Hiperparámetros de una red. Los mismos para todas las arquitecturas; algunos sólo aplican a una (ver
    comentarios)."""
    arch: str = "mlp"            # 'mlp' | 'nbeats' | 'lineal'
    n1: int = 100
    n2: int = 50
    act: str = "relu"            # relu | selu | prelu
    dropout: float = 0.1
    lr: float = 1e-3
    l1: float = 1e-4             # penalización L1 sobre los pesos de entrada
    wd: float = 1e-4
    epochs: int = 150
    patience: int = 15
    batch: int = 32
    p_mask: float = 0.4          # sólo V4
    n_bloques: int = 2           # sólo nbeats: bloques por stack
    seeds: int = 4
    winsor: float = 0.01
    dist: str = "none"           # 'none' (pérdida MAE) | 'jsu' (verosimilitud Johnson SU)
    afinar: int = 0              # con la red: épocas extra sólo con Monclova, a un tercio de la tasa de aprendizaje
    cuello: int = 0              # mlp y conv: neuronas por las que pasa el MDA antes de unirse al resto (0 = sin cuello)
    p_ramal: float = 0.0         # mlp y conv: probabilidad de apagar la salida del cuello del MDA en cada ejemplo al entrenar
    k_conv: int = 8              # conv y dual: canales de la convolución
    fusion: str = "concat"       # dual: cómo se unen las dos redes: 'concat' (se pegan) o 'film' (el MDA modula a la red de historia)

    def clave(self):
        """Texto que identifica la configuración en el nombre del archivo de caché."""
        d = {k: v for k, v in self.__dict__.items() if not ((k == "dist" and v == "none") or (k in ("afinar", "cuello") and v == 0) or (k == "p_ramal" and v == 0.0) or (k == "k_conv" and v == 8) or (k == "fusion" and v == "concat"))}   # claves de caché previas intactas
        return "|".join(f"{k}={v}" for k, v in sorted(d.items()))


# Búsqueda aleatoria acotada (fija y reproducible): se evalúa sólo en desarrollo con la variante V2 y 2 semillas.
def busqueda(arch, n=10, seed=0):
    """Primera búsqueda aleatoria de hiperparámetros (fija y reproducible): `n` configuraciones con L1 débil, 2
    semillas."""
    r = np.random.default_rng(seed + (0 if arch == "mlp" else 100))
    out = []
    for _ in range(n):
        out.append(HP(arch=arch, n1=int(r.choice([64, 128, 256])), n2=int(r.choice([32, 64, 128])), act=str(r.choice(["relu", "selu", "prelu"])),
                      dropout=float(r.choice([0.0, 0.1, 0.3])), lr=float(r.choice([3e-4, 1e-3, 3e-3])), l1=float(r.choice([0.0, 1e-4, 1e-3])),
                      wd=float(r.choice([1e-5, 1e-4, 1e-3])), n_bloques=int(r.choice([1, 2])), seeds=2))
    return out


def busqueda_reg(arch, n=12, seed=7):
    """Segunda ronda: la primera usaba L1 <= 1e-3, demasiado débil con ~300 filas y ~300 entradas (la red lineal de control
    quedaba 15% peor que LEAR; con L1 = 3 la iguala). Aquí L1 entre 0.3 y 10, el resto igual."""
    r = np.random.default_rng(seed + (0 if arch == "mlp" else 100))
    out = []
    for _ in range(n):
        out.append(HP(arch=arch, n1=int(r.choice([64, 128, 256])), n2=int(r.choice([32, 64, 128])), act=str(r.choice(["relu", "selu", "prelu"])),
                      dropout=float(r.choice([0.0, 0.1, 0.3])), lr=float(r.choice([3e-4, 1e-3, 3e-3])), l1=float(r.choice([0.3, 1.0, 3.0, 10.0])),
                      wd=float(r.choice([1e-5, 1e-4, 1e-3])), n_bloques=int(r.choice([1, 2])), seeds=2))
    return out


# -------------------------------------------------------------------------------------------------- datos
def preparar(c: f2.Ctx, variante: str, extra=(), comp=None, k=20, ajustar_con=None, fijos=()):
    """Devuelve F (features), Yt (objetivo transformado) y metadatos de columnas. `extra`: grupos exógenos adicionales.
    `comp` (notebook 07): None = las variables tal cual; 'nucleo' = las de siempre tal cual y las `extra` resumidas en `k`
    componentes principales; 'todo' = todo resumido en `k` componentes (la red ya no ve la ventana de MTR ni los rezagos).
    `ajustar_con`: función que da la tabla con la que se ajustan las componentes (por omisión, la propia F).
    `fijos`: grupos que entran tal cual aunque haya `comp` (el calendario, que es categórico)."""
    if comp is None:
        F, _ = f2.construir(c, GRUPOS_V[variante] + list(extra))
    else:
        Fb, _ = f2.construir(c, GRUPOS_V[variante] + list(fijos)); Fe, _ = f2.construir(c, list(extra))
        if comp == "nucleo":
            F = pd.concat([Fb, f2.comprimir(Fe, k, ajustar_con=None if ajustar_con is None else ajustar_con(Fe.columns))[0]], axis=1)
        else:
            Fa = pd.concat([Fb, Fe], axis=1)
            F = f2.comprimir(Fa, k, ajustar_con=None if ajustar_con is None else ajustar_con(Fa.columns))[0]
    Yt, Xt = f2._t(c)
    tgt = (Yt - Xt) if variante.startswith("spread") else Yt
    cols = F.columns.tolist()
    # ventana de retrospectiva en orden cronológico (d−14 → d−8): para NBEATSx-lite
    mtr = [col for col in cols if col.startswith("mtr_l")]
    key = lambda s: (-int(re.match(r"mtr_l(\d+)_h(\d+)", s).group(1)), int(re.match(r"mtr_l(\d+)_h(\d+)", s).group(2)))
    back = sorted(mtr, key=key)
    mda = sorted([col for col in cols if col.startswith("mda_l")], key=lambda s: (-int(re.match(r"mda_l(\d+)_h(\d+)", s).group(1)), int(re.match(r"mda_l(\d+)_h(\d+)", s).group(2))))
    return F, tgt.loc[F.index], Xt.loc[F.index], dict(back=back, mda=mda, exo=[col for col in cols if col not in back])


# -------------------------------------------------------------------------------------------------- redes
def _modelo(torch, nn, n_in, hp: HP, meta_idx):
    """Construye la red pedida (TiDE, lineal, DNN o NBEATSx) con `n_in` entradas y 24 salidas (96 con Johnson SU).
    Todas tienen `pen()`, la penalización L1 sobre los pesos de entrada."""
    act = {"relu": nn.ReLU, "selu": nn.SELU, "prelu": nn.PReLU}[hp.act]
    n_out = 24 * (4 if hp.dist == "jsu" else 1)                    # JSU: (xi, lambda, gamma, delta) por hora

    if hp.arch == "tide":
        cov = torch.tensor(meta_idx["cov"], dtype=torch.long).reshape(24, len(meta_idx["cov"][0]))   # 24 × k: posiciones de las covariables de cada hora
        p = 8

        class Res(nn.Module):
            def __init__(s, a, h, b):
                super().__init__(); s.f = nn.Sequential(nn.Linear(a, h), act(), nn.Linear(h, b), nn.Dropout(hp.dropout)); s.sk = nn.Linear(a, b)
            def forward(s, x): return s.f(x) + s.sk(x)

        class TiDE(nn.Module):
            def __init__(s):
                super().__init__()
                s.l1 = nn.Linear(n_in, hp.n1)
                s.enc = nn.Sequential(act(), Res(hp.n1, hp.n1, hp.n2), Res(hp.n2, hp.n2, 24 * p))
                s.dec = Res(p + cov.shape[1] + 24, hp.n2, n_out // 24)
                s.skip = nn.Linear(n_in, n_out)
                s.register_buffer("cov", cov); s.register_buffer("eye", torch.eye(24))
            def forward(s, x):
                n = x.shape[0]
                code = s.enc(s.l1(x)).view(n, 24, p)
                ch = x[:, s.cov]                                   # n × 24 × k
                d = s.dec(torch.cat([code, ch, s.eye.expand(n, 24, 24)], dim=2))   # n × 24 × (1 o 4)
                d = d.permute(0, 2, 1).reshape(n, n_out)            # mismo orden que la salida plana: parámetro, hora
                return s.skip(x) + d
            def pen(s): return s.l1.weight.abs().sum() + s.skip.weight.abs().sum()
        return TiDE()

    if hp.arch == "lineal":                                       # control de cordura: debería parecerse a LEAR
        class Lin(nn.Module):
            def __init__(s):
                super().__init__(); s.l1 = nn.Linear(n_in, n_out)
            def forward(s, x): return s.l1(x)
            def pen(s): return s.l1.weight.abs().sum()
        return Lin()

    pos = meta_idx.get("pos", {})
    mi = [pos[c] for c in meta_idx.get("mda", []) if c in pos]                        # columnas del MDA
    bi = [pos[c] for c in meta_idx.get("back", []) if c in pos]                       # ventana de MTR en orden cronológico
    oi = [i for i in range(n_in) if i not in set(mi)]                                 # todo lo que no es MDA (y la bandera de V4)

    class RamaMDA(nn.Module):
        """El MDA pasa por un cuello de `hp.cuello` neuronas (o sin cuello, tal cual) y, al entrenar, se apaga entero en
        una fracción `hp.p_ramal` de los ejemplos. Así la red no puede depender sólo de copiar el MDA."""
        def __init__(s):
            super().__init__()
            s.mi = torch.tensor(mi, dtype=torch.long)
            s.w = nn.Linear(len(mi), hp.cuello) if (hp.cuello and mi) else None
            s.dim = hp.cuello if (hp.cuello and mi) else len(mi)
        def forward(s, x):
            z = x[:, s.mi]
            if s.w is not None: z = s.w(z)
            if s.training and hp.p_ramal > 0 and s.dim:
                z = z * (torch.rand(z.shape[0], 1) >= hp.p_ramal).float()
            return z
        def pen(s): return s.w.weight.abs().sum() if s.w is not None else torch.zeros(())

    if hp.arch == "mlp":
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.mda = RamaMDA(); s.oi = torch.tensor(oi, dtype=torch.long); s.cuello = bool(hp.cuello and mi)
                s.l1 = nn.Linear(len(oi) + s.mda.dim if s.cuello else n_in, hp.n1)      # sin cuello: la red de siempre, intacta
                s.net = nn.Sequential(act(), nn.Dropout(hp.dropout), nn.Linear(hp.n1, hp.n2), act(), nn.Dropout(hp.dropout), nn.Linear(hp.n2, n_out))
            def forward(s, x): return s.net(s.l1(torch.cat([x[:, s.oi], s.mda(x)], dim=1) if s.cuello else x))
            def pen(s): return s.l1.weight.abs().sum() + s.mda.pen()
        return M()

    if hp.arch == "dual":
        kc, ke = hp.k_conv, (hp.cuello or 8)
        class Dual(nn.Module):
            """Dos redes unidas. Una convolución mira la ventana del MDA (8 días de 24 horas) y resume cómo se comporta el
            MDA en general, en `ke` números; es la rama que se apaga al azar al entrenar (`p_ramal`). Otra red, densa y con
            penalización L1, aprende de la historia real del MTR (ventana de 7 días, error ya publicado, calendario y las
            variables extra). Se unen pegando los dos resúmenes ('concat') o dejando que el resumen del MDA module a la red
            de historia ('film': la escala y desplaza)."""
            def __init__(s):
                super().__init__()
                s.mi = torch.tensor(mi, dtype=torch.long); s.hi = torch.tensor([i for i in range(n_in) if i not in set(mi)], dtype=torch.long)
                s.c1 = nn.Conv1d(1, kc, 3, padding=1); s.c2 = nn.Conv1d(kc, kc, 5, padding=6, dilation=3)
                s.pool = nn.AvgPool1d(24); s.cm = nn.Linear(kc * (len(mi) // 24), ke)
                s.h = nn.Sequential(nn.Linear(len(s.hi), hp.n1), act(), nn.Dropout(hp.dropout), nn.Linear(hp.n1, hp.n2), act())
                s.film = hp.fusion == "film"
                if s.film: s.fi = nn.Linear(ke, 2 * hp.n2)
                s.head = nn.Linear(hp.n2 + (0 if s.film else ke), n_out)
            def forward(s, x):
                em = s.cm(s.pool(act()(s.c2(act()(s.c1(x[:, s.mi].unsqueeze(1)))))).flatten(1))
                if s.training and hp.p_ramal > 0:
                    em = em * (torch.rand(em.shape[0], 1) >= hp.p_ramal).float()
                eh = s.h(x[:, s.hi])
                if s.film:
                    g, b = s.fi(em).chunk(2, dim=1); return s.head(eh * (1 + torch.tanh(g)) + b)
                return s.head(torch.cat([eh, em], dim=1))
            def pen(s): return s.h[0].weight.abs().sum() + s.cm.weight.abs().sum()
        return Dual()

    if hp.arch == "conv":
        kc = hp.k_conv
        class Conv(nn.Module):
            """Rama convolucional sobre la ventana de 7 días de MTR (168 horas), más rama del MDA y rama del resto.
            La primera convolución mira 3 horas; la segunda, con huecos, mira 5 puntos separados 3 horas (ciclo corto
            del día); el promedio por día deja un resumen de cada una de las 7 jornadas."""
            def __init__(s):
                super().__init__()
                s.bi = torch.tensor(bi, dtype=torch.long); nb = len(bi)
                s.mda = RamaMDA()
                s.ei = torch.tensor([i for i in oi if i not in set(bi)], dtype=torch.long)
                if nb:
                    s.c1 = nn.Conv1d(1, kc, 3, padding=1); s.c2 = nn.Conv1d(kc, kc, 5, padding=6, dilation=3)
                    s.pool = nn.AvgPool1d(24); s.cd = nn.Linear(kc * (nb // 24), 16)
                s.exo = nn.Linear(len(s.ei), 16)
                s.head = nn.Sequential(act(), nn.Dropout(hp.dropout), nn.Linear((16 if nb else 0) + 16 + s.mda.dim, hp.n1), act(), nn.Dropout(hp.dropout), nn.Linear(hp.n1, n_out))
                s.nb = nb
            def forward(s, x):
                z = [s.exo(x[:, s.ei]), s.mda(x)]
                if s.nb:
                    h = act()(s.c1(x[:, s.bi].unsqueeze(1))); h = act()(s.c2(h))
                    z.append(s.cd(s.pool(h).flatten(1)))
                return s.head(torch.cat(z, dim=1))
            def pen(s): return s.exo.weight.abs().sum() + s.mda.pen() + (s.cd.weight.abs().sum() if s.nb else 0)
        return Conv()

    # NBEATSx-lite: bloques doblemente residuales; stacks tendencia (polinomio) → estacionalidad (Fourier) → exógeno
    nb, ne = len(meta_idx["back"]), n_in - len(meta_idx["back"])
    tt_b = torch.linspace(0, 1, nb); tt_f = torch.linspace(0, 1, 24)

    def base_tend(g):
        """Base polinomial (tendencia) para la ventana de entrada y para las 24 horas de salida."""
        return (torch.stack([tt_b ** i for i in range(g + 1)]), torch.stack([tt_f ** i for i in range(g + 1)]))

    def base_est(k):
        """Base de Fourier (estacionalidad diaria) para la ventana de entrada y para las 24 horas de salida."""
        fb = [f(2 * np.pi * i * torch.arange(nb) / 24) for i in range(1, k + 1) for f in (torch.sin, torch.cos)]
        ff = [f(2 * np.pi * i * torch.arange(24) / 24) for i in range(1, k + 1) for f in (torch.sin, torch.cos)]
        return torch.stack(fb), torch.stack(ff)

    def _rep(B, Fh):                                               # JSU: la base se repite para cada parámetro
        return B, torch.block_diag(*[Fh] * (n_out // 24))

    class Bloque(nn.Module):
        def __init__(s, kind):
            super().__init__()
            s.kind = kind
            if kind == "tend": B, Fh = _rep(*base_tend(2))
            elif kind == "est": B, Fh = _rep(*base_est(2))
            if kind != "exo":
                s.register_buffer("B", B.float()); s.register_buffer("Fh", Fh.float()); d = B.shape[0]
                s.fc = nn.Sequential(nn.Linear(nb + ne, hp.n1), act(), nn.Dropout(hp.dropout), nn.Linear(hp.n1, hp.n1), act(), nn.Linear(hp.n1, d + Fh.shape[0]))
                s.d = d
            else:
                s.fc = nn.Sequential(nn.Linear(nb + ne, hp.n1), act(), nn.Dropout(hp.dropout), nn.Linear(hp.n1, hp.n1), act())
                s.out_f = nn.Linear(hp.n1, n_out); s.out_b = nn.Linear(hp.n1, nb)
        def forward(s, back, exo):
            h = s.fc(torch.cat([back, exo], dim=1))
            if s.kind == "exo": return s.out_b(h), s.out_f(h)
            tb, tf = h[:, :s.d], h[:, s.d:]
            return tb @ s.B, tf @ s.Fh

    class NB(nn.Module):
        def __init__(s):
            super().__init__()
            s.blq = nn.ModuleList([Bloque(k) for k in ("tend", "est", "exo") for _ in range(hp.n_bloques)])
            ei = [meta_idx["pos"][c] for c in meta_idx["exo"]] + list(range(len(meta_idx["pos"]), n_in))   # + bandera de V4
            s.bi = torch.tensor([meta_idx["pos"][c] for c in meta_idx["back"]], dtype=torch.long); s.ei = torch.tensor(ei, dtype=torch.long)
        def forward(s, x):
            back, exo = x[:, s.bi], x[:, s.ei]
            res = back; f = 0
            for b in s.blq:
                bb, ff = b(res, exo)
                res = res - bb; f = f + ff
            return f
        def pen(s): return sum(p.abs().sum() for b in s.blq for p in b.fc[0].parameters() if p.dim() == 2)
    return NB()


def _torch():
    """Importa torch silenciando el aviso de compatibilidad numpy 1.x/2.x (torch 2.2 no usa aquí el puente con numpy)."""
    import contextlib, io, warnings
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        from torch import nn
    return torch, nn


def _ajusta_predice(Xtr, ytr, Xte, hp: HP, seed, meta_idx, mda_pos=None, mda_mode=False, ft=None):
    """Entrena una red y devuelve predicciones sobre Xte (n × 24). Con mda_mode devuelve (con_mda, sin_mda)."""
    torch, nn = _torch()
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    # columnas que no cambian en el entrenamiento (p. ej., meses o feriados que aún no se han visto) se ponen en cero:
    # no enseñan nada y, divididas entre una desviación ~0, explotarían al pronosticar. El tope de ±10 desviaciones
    # protege de valores nunca vistos; con las variables de siempre no se alcanza (máximo 8.4 en desarrollo).
    mu, sd = Xtr.mean(0), Xtr.std(0)
    fija = sd < 1e-6; sd = np.where(fija, 1.0, sd)
    Xn = np.clip((Xtr - mu) / sd, -10, 10).astype("float32"); Xt = np.clip((Xte - mu) / sd, -10, 10).astype("float32")
    Xn[:, fija] = 0; Xt[:, fija] = 0
    if mda_mode:
        Xn = np.hstack([Xn, np.zeros((len(Xn), 1), "float32")]); Xt = np.hstack([Xt, np.zeros((len(Xt), 1), "float32")])
    n = len(Xn); idx = rng.permutation(n); nv = max(8, int(0.15 * n))
    va, tr = idx[:nv], idx[nv:]
    net = _modelo(torch, nn, Xn.shape[1], hp, meta_idx)
    opt = torch.optim.Adam(net.parameters(), lr=hp.lr, weight_decay=hp.wd)
    T_ = lambda a: torch.tensor(np.asarray(a, dtype=float).tolist(), dtype=torch.float32)   # sin puente numpy↔torch (torch 2.2 vs numpy 2)
    Xtr_t, ytr_t = T_(Xn), T_(ytr)
    mp = None if mda_pos is None else torch.tensor(list(mda_pos))

    def enmascara(x, mask):
        """V4: apaga el bloque del MDA en las filas marcadas y avisa a la red con una bandera en la última columna."""
        x = x.clone()
        x[torch.where(mask)[0][:, None], mp] = 0.0
        x[:, -1] = mask.float()
        return x

    jsu = hp.dist == "jsu"
    sp = torch.nn.functional.softplus

    def params(o):                                                 # salida plana -> (xi, lam, gam, dlt), cada uno n × 24
        """Johnson SU: de la salida plana a los cuatro parámetros por hora (posición, escala, forma y cola), con escala
        y cola positivas."""
        xi, lam, gam, dlt = o[:, :24], o[:, 24:48], o[:, 48:72], o[:, 72:96]
        return xi, sp(lam) + 1e-3, gam, sp(dlt) + 0.1

    def perdida(o, y):
        """Pérdida de entrenamiento: error absoluto medio, o menos log-verosimilitud si la salida es Johnson SU."""
        if not jsu:
            return (o - y).abs().mean()
        xi, lam, gam, dlt = params(o)
        u = (y - xi) / lam
        z = gam + dlt * torch.asinh(u)
        return (-(torch.log(dlt) - torch.log(lam) - 0.5 * torch.log1p(u ** 2) - 0.5 * z ** 2)).mean()

    def cuantiles(o, qs=(0.1, 0.5, 0.9)):
        """Johnson SU: cuantiles 10, 50 y 90 de la distribución pronosticada, por hora."""
        from scipy.stats import norm
        xi, lam, gam, dlt = params(o)
        return torch.cat([xi + lam * torch.sinh((float(norm.ppf(q)) - gam) / dlt) for q in qs], dim=1)

    mejor, mejor_sd, sin_mejora = np.inf, None, 0
    for ep in range(hp.epochs):
        net.train(); p = rng.permutation(tr)
        for i in range(0, len(p), hp.batch):
            b = torch.tensor(p[i:i + hp.batch].tolist()); xb, yb = Xtr_t[b], ytr_t[b]
            if mda_mode:
                xb = enmascara(xb, torch.rand(len(b)) < hp.p_mask)
            loss = perdida(net(xb), yb) + hp.l1 * net.pen() / max(1, Xn.shape[1])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            xv, yv = Xtr_t[va.tolist()], ytr_t[va.tolist()]
            if mda_mode:
                v = 0.5 * ((net(enmascara(xv, torch.zeros(len(va), dtype=torch.bool))) - yv).abs().mean() + (net(enmascara(xv, torch.ones(len(va), dtype=torch.bool))) - yv).abs().mean())
            else:
                v = perdida(net(xv), yv)
        if v.item() < mejor - 1e-5:
            mejor, sin_mejora = v.item(), 0; mejor_sd = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= hp.patience: break
    net.load_state_dict(mejor_sd); net.eval()
    if ft is not None and hp.afinar:                              # afinado: unas épocas más sólo con las filas de Monclova
        Xf, yf = T_(((ft[0] - mu) / sd).astype("float32")), T_(ft[1])
        opt = torch.optim.Adam(net.parameters(), lr=hp.lr / 3, weight_decay=hp.wd)
        net.train()
        for _ in range(hp.afinar):
            p = rng.permutation(len(Xf))
            for i in range(0, len(p), 32):
                b = torch.tensor(p[i:i + 32].tolist())
                loss = perdida(net(Xf[b]), yf[b]) + hp.l1 * net.pen() / max(1, Xn.shape[1])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
    _ajusta_predice.ultimo = dict(epocas=ep + 1, val=mejor)
    with torch.no_grad():
        xt = T_(Xt)
        if mda_mode:
            return (np.array(net(enmascara(xt, torch.zeros(len(xt), dtype=torch.bool))).tolist()), np.array(net(enmascara(xt, torch.ones(len(xt), dtype=torch.bool))).tolist()))
        return np.array((cuantiles(net(xt)) if jsu else net(xt)).tolist())


def _bloque(d0, d1, F, T, hp: HP, meta, variante, inicio):
    """Entrena con los días <= d0 − 8 (recortando el 1% extremo del objetivo) y pronostica la semana [d0, d1],
    promediando `hp.seeds` semillas."""
    fin = d0 - pd.Timedelta(days=LAG_MTR)
    tr = F.index[(F.index >= inicio) & (F.index <= fin)]
    te = F.index[(F.index >= d0) & (F.index <= d1)]
    Ytr = T.loc[tr].values
    if hp.winsor:
        lo, hi = np.quantile(Ytr, [hp.winsor, 1 - hp.winsor]); Ytr = np.clip(Ytr, lo, hi)
    cols = F.columns.tolist()
    pos = {c: i for i, c in enumerate(cols)}
    m = dict(meta, pos=pos)
    if hp.arch == "tide":                                          # covariables de cada hora: todas las columnas *_h{h}
        m["cov"] = [[pos[c] for c in cols if c.endswith(f"_h{h}")] for h in range(1, 25)]
    mda_pos = [pos[c] for c in meta["mda"]] if variante == "mda_drop" else None
    outs = [_ajusta_predice(F.loc[tr].values, Ytr, F.loc[te].values, hp, s, m, mda_pos, variante == "mda_drop") for s in range(hp.seeds)]
    if variante == "mda_drop":
        a = np.mean([o[0] for o in outs], axis=0); b = np.mean([o[1] for o in outs], axis=0)
        return te, a, b
    return te, np.mean(outs, axis=0), None


def correr_dl(c: f2.Ctx, variante: str, hp: HP, dias, nombre=None, n_jobs=-1, inicio=None, extra=(), comp=None, k=20):
    """Backtest de una red: bloques de 7 días, ensamble de `hp.seeds` semillas. Devuelve días × 24 en espacio transformado
    (para V3 es el spread transformado). V4 devuelve un dict {'con': ..., 'sin': ...}. Caché en disco. `comp`: ver
    `preparar` (resumir las variables en componentes principales)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    clave = hashlib.md5(f"{variante}|{hp.clave()}|{c.escala}|{c.med:.4f}|{dias[0]}|{dias[-1]}{'|' + '+'.join(extra) + f'|lag{c.lag_precios}' if extra else ''}{'' if c.modo == 'asinh' else '|' + c.modo}{f'|comp:{comp}{k}' if comp else ''}".encode()).hexdigest()[:12]
    f = CACHE / f"{nombre or variante}_{clave}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    F, T, _, meta = preparar(c, variante, extra, comp, k)
    inicio = inicio or F.index.min()
    dias = pd.DatetimeIndex(dias)
    bl = [(dias[i], dias[min(i + 6, len(dias) - 1)]) for i in range(0, len(dias), 7)]
    res = Parallel(n_jobs=n_jobs)(delayed(_bloque)(a, b, F, T, hp, meta, variante, inicio) for a, b in bl)
    cols = T.columns
    if hp.dist == "jsu":                                           # dict de cuantiles 0.1 / 0.5 / 0.9
        out = {q: pd.concat([pd.DataFrame(r[1][:, 24 * i:24 * (i + 1)], index=r[0], columns=cols) for r in res]) for i, q in enumerate((0.1, 0.5, 0.9))}
        f.write_bytes(pickle.dumps(out))
        return out
    A = pd.concat([pd.DataFrame(r[1], index=r[0], columns=cols) for r in res])
    out = A if variante != "mda_drop" else dict(con=A, sin=pd.concat([pd.DataFrame(r[2], index=r[0], columns=cols) for r in res]))
    f.write_bytes(pickle.dumps(out))
    return out


# ------------------------------------------------------------------------------ entrenar con la red (notebook 06)
def preparar_red(c: f2.Ctx, variante: str, troncal=False, extra=(), comp=None, k=20):
    """Features y objetivo de Monclova y de cada nodo de la red (su propio MTR y MDA, escala de Monclova), apilados,
    con el corredor del nodo como indicadora (`corr_0` = Monclova). Con `comp="nucleo"`, las `extra` se resumen en `k`
    componentes ajustadas con las filas de todos los nodos apiladas; el calendario entra tal cual."""
    from . import multinodo as mn
    from .todo import CALENDARIO
    ctxs = mn.contextos(c, True, troncal)
    fijos = [g for g in extra if g in CALENDARIO] if comp else []
    extra = [g for g in extra if g not in fijos]
    ajustar_con = None
    if comp:
        apilado = pd.concat([f2.construir(cn, list(extra))[0] for _, _, cn in ctxs])
        ajustar_con = lambda cols: apilado[cols]
    Fs, Ts, meta = [], [], None
    for n, cod, cn in ctxs:
        F, T, _, m = preparar(cn, variante, extra, comp, k, ajustar_con, fijos)
        meta = meta or m
        oh = pd.DataFrame(0.0, index=F.index, columns=[f"corr_{i}" for i in range(len(mn.CORR))]); oh[f"corr_{cod}"] = 1.0
        Fs.append(pd.concat([F, oh], axis=1)); Ts.append(T)
    F, T = pd.concat(Fs), pd.concat(Ts)
    return F, T, dict(meta, exo=meta["exo"] + [x for x in F.columns if x.startswith("corr_")])


def _bloque_red(d0, d1, Fv, Tv, fecha, cols, hp: HP, meta, inicio):
    """Como `_bloque`, con las filas de todos los nodos para entrenar y sólo las de Monclova (corr_0 = 1) para
    pronosticar."""
    fin = d0 - pd.Timedelta(days=LAG_MTR)
    tr = (fecha >= np.datetime64(inicio)) & (fecha <= np.datetime64(fin)) & ~np.isnan(Fv).any(axis=1) & ~np.isnan(Tv).any(axis=1)
    te = (Fv[:, cols.index("corr_0")] == 1) & (fecha >= np.datetime64(d0)) & (fecha <= np.datetime64(d1))
    Ytr = Tv[tr]
    if hp.winsor:
        lo, hi = np.quantile(Ytr, [hp.winsor, 1 - hp.winsor]); Ytr = np.clip(Ytr, lo, hi)
    pos = {x: i for i, x in enumerate(cols)}
    m = dict(meta, pos=pos)
    if hp.arch == "tide":
        m["cov"] = [[pos[x] for x in cols if x.endswith(f"_h{h}")] for h in range(1, 25)]
    mon = tr & (Fv[:, cols.index("corr_0")] == 1)
    ft = (Fv[mon], Ytr[mon[tr]]) if hp.afinar else None
    outs = [_ajusta_predice(Fv[tr], Ytr, Fv[te], hp, s, m, ft=ft) for s in range(hp.seeds)]
    return fecha[te], np.mean(outs, axis=0)


def correr_dl_red(c: f2.Ctx, variante: str, hp: HP, dias, troncal=False, nombre=None, n_jobs=-1, extra=(), comp=None, k=20):
    """Como `correr_dl`, pero cada bloque se entrena con las filas de Monclova y de todos los nodos de la red (<= d0 − 8)
    y se pronostica sólo Monclova. Con ~65 veces más filas, el lote sube a 256 (`hp.batch`) para que el costo sea manejable."""
    CACHE.mkdir(parents=True, exist_ok=True)
    clave = hashlib.md5(f"red|{troncal}|{variante}|{hp.clave()}|{c.escala}|{c.med:.4f}|{dias[0]}|{dias[-1]}{'' if c.modo == 'asinh' else '|' + c.modo}{'|' + '+'.join(extra) if extra else ''}{f'|comp:{comp}{k}' if comp else ''}".encode()).hexdigest()[:12]
    f = CACHE / f"{nombre or variante}_red_{clave}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    F, T, meta = preparar_red(c, variante, troncal, extra, comp, k)
    cols = F.columns.tolist()
    Fv, Tv, fecha = F.values.astype(float), T.values.astype(float), F.index.values
    dias = pd.DatetimeIndex(dias)
    bl = [(dias[i], dias[min(i + 6, len(dias) - 1)]) for i in range(0, len(dias), 7)]
    res = Parallel(n_jobs=n_jobs)(delayed(_bloque_red)(a, b, Fv, Tv, fecha, cols, hp, meta, F.index.min()) for a, b in bl)
    out = pd.concat([pd.DataFrame(r[1], index=pd.DatetimeIndex(r[0]), columns=T.columns) for r in res])
    f.write_bytes(pickle.dumps(out))
    return out


def a_precios(c: f2.Ctx, P: pd.DataFrame, variante: str) -> pd.DataFrame:
    """De la salida de la red (escala del proyecto) al precio en MXN/MWh. En V3 se le suma el MDA transformado."""
    Z = P + (f2._t(c)[1].loc[P.index] if variante.startswith("spread") else 0)
    return f2.inv(Z, c)
