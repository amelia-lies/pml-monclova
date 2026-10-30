"""Estilo común de gráficas (paleta categórica fija: 1 azul, 2 naranja, 3 aqua)."""
import matplotlib as mpl

AZUL, NARANJA, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def aplicar():
    """Aplica el estilo común a todas las figuras de matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": GRIS, "text.color": "#0b0b0b",
        "xtick.color": GRIS, "ytick.color": GRIS, "axes.grid": True,
        "grid.color": "#ececea", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.2, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 110,
    })
