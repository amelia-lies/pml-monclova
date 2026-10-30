"""Qué grupos de variables entran cuando un modelo recibe "todo", y cómo se agrupan para la escalera (notebooks 07 y 08).

Todos los grupos que se construyeron en el proyecto, de G0 a G16, salvo las versiones alternativas que dicen lo mismo de
otra forma (G0w en lugar de G0, G1f en lugar de G1, G2w en lugar de G2). Son 744 variables.
"""
TODO = ["G0", "G0dow", "G1", "G2", "G4", "G5", "G6", "G6g", "G6e", "G7", "G7v", "G8", "G9", "G10", "G11", "G12", "G13", "G14", "G15", "G16"]
NUCLEO = ["G0", "G2"]            # lo que LEAR usa siempre (rezagos del MTR y del MDA); el resto se comprime
K_PC = 20                        # componentes principales del resto cuando se comprime
# Las redes ya traen su propia ventana de MTR (G0w), el MDA (G2) y el calendario (G1): se agrega todo lo demás.
EXTRA_RED = [g for g in TODO if g not in ("G0", "G1", "G2")]
# Para la red sin MDA, sólo lo que no viene del MDA: historia del MTR, clima, combustibles, ERCOT, calendario, operación.
EXTRA_SIN_MDA = ["G0dow", "G5", "G6", "G6g", "G6e", "G7", "G7v", "G8", "G9", "G11"]
# Escalera (notebook 08): sobre la base (NUCLEO) se agregan por separado las exógenas y el modelado estadístico del MTR y del
# MDA; "todas" son las dos juntas. Las dos listas dividen TODO − NUCLEO sin traslape.
EXOGENAS = ["G1", "G5", "G6", "G6e", "G6g", "G7", "G7v", "G8", "G9", "G10", "G11"]
ESTADISTICO = ["G0dow", "G4", "G12", "G13", "G14", "G15", "G16"]
# Variables de calendario (día de la semana, mes, feriados, vacaciones): son categóricas, así que nunca entran al PCA.
CALENDARIO = ["G1", "G7", "G7v"]
# Familias para medir qué pierde un modelo si se le quitan (notebook 07).
FAMILIAS = {"rezagos del MTR": ["G0", "G0dow"], "MDA y su error": ["G2"], "estadísticas móviles y varias escalas": ["G4", "G13"],
            "complejidad y correlación del error del MDA": ["G15"], "complejidad y correlación del MTR": ["G16"],
            "la red: congestión y red en el espacio": ["G10", "G14"], "días análogos": ["G12"],
            "calendario y vacaciones": ["G1", "G7", "G7v"], "clima": ["G5", "G11"], "combustibles, gas y Texas": ["G6", "G6g", "G6e"],
            "estado del sistema e información pública": ["G8", "G9"]}
