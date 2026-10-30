"""Qué pierde un modelo si le quitamos cada familia de variables (experimento anterior; las familias actuales están en `todo.py`).

En vez de mirar coeficientes o "importancias" internas, que en modelos distintos significan cosas distintas, se
reentrena el modelo completo sin una familia de variables y se mide cuánto sube su error en desarrollo. Es la pregunta
más directa: ¿cuánto le cuesta al modelo no tener esa información? Si casi no sube, el modelo la tenía repetida en otra
familia o no le servía. Las familias agrupan variables que dicen lo mismo (por ejemplo, los cuatro rezagos del MTR),
porque quitar una sola de varias variables parecidas casi nunca cambia nada.
"""

# Gradient boosting con la red (formato largo: las variables horarias ya son una sola columna por familia de rezagos)
FAMILIAS_GB = {
    "MTR de hace 8 a 14 días": ["mtr_l8", "mtr_l9", "mtr_l10", "mtr_l14"],
    "MTR del mismo día hace 3 y 4 semanas": ["mtr_l21", "mtr_l28"],
    "MDA del día que se pronostica": ["mda_l0"],
    "MDA de ayer y de hace una semana": ["mda_l1", "mda_l7"],
    "error del MDA ya publicado (d−8 y d−14)": ["spr_l8", "spr_l14"],
    "estadísticas móviles del MTR": ["mtr_m_", "mtr_x_"],
    "estadísticas móviles del error del MDA": ["spr_m_", "spr_a_", "up_"],
    "forma y sorpresa del MDA de mañana": ["mda_m", "mda_x", "mda_ramp", "mda_nivel28", "mda_vol28", "mda_vs_mtr7", "mda_sorpresa"],
    "hora del día": ["hora"],
    "corredor del nodo": ["nodo"],
}

# LEAR (formato ancho: un prefijo por familia)
FAMILIAS_LEAR = {
    "MTR de hace 8 a 14 días": ("mtr_l",),
    "MDA (del día, de ayer y de hace una semana)": ("mda_l",),
    "error del MDA ya publicado (d−8 y d−14)": ("spr_l",),
}


def columnas(cols, prefijos) -> list:
    """Las columnas que pertenecen a una familia: nombre exacto o que empiezan con el prefijo (los que terminan en _)."""
    out = []
    for c in cols:
        for p in prefijos:
            if c == p or (p.endswith("_") and c.startswith(p)) or (p == "mda_sorpresa" and c.startswith(p)):
                out.append(c)
    return sorted(set(out))
