# Detalles del proyecto

Versión extendida del README: la historia por decisiones, la tabla completa del test, las ideas descartadas, el glosario y el mapa del código. El resumen está en el [README](../README.md).


Cada día *d−1*, al mediodía, se pronostica el precio horario del Mercado de Tiempo Real (MTR) del día *d* para los 9 NodosP de la zona de carga Monclova. Al emitir ya se conoce el precio del Mercado del Día en Adelanto (MDA) del día *d*, pero el MTR solo se conoce hasta *d−8*. Los datos van del 1 de enero de 2024 al 30 de junio de 2025.

## La historia en pocas líneas

El proyecto avanza por decisiones, todas tomadas en desarrollo (julio a diciembre de 2024). Desde el notebook 06, todos los modelos pasan por el mismo camino, para que la escalera de decisiones (notebook 08) se pueda leer modelo contra modelo.

1. **El MDA ya es un buen pronóstico del MTR, y hay que partir de él.** Casi todo su error está en la energía, un precio nacional que se decide en el tiempo real. Los modelos que lo ignoran quedan 16% a 21% peor; los que aprenden cuánto corregirlo lo igualan o lo mejoran poco (notebooks 02 a 05).
2. **Faltaban ejemplos.** El gradient boosting tenía unos 300 días. Entrenarlo con Monclova y 64 nodos de la red, con las mismas variables, lo mejora 3.1%; a LEAR en proporción también lo ayuda y a las redes les hace daño. Lo que cuenta es la variedad de nodos, no su número. Desde aquí, todos los modelos usan los 64 nodos (notebook 06).
3. **Qué información importa.** Lo que más pesa es el MDA con su historial de error y, para el gradient boosting, lo que describe cómo se mueve el precio (estadísticas móviles, la red, días parecidos). Las variables externas aportan poco y según el modelo. Dar todas las variables no suma, y resumirlas con PCA ayuda a LEAR pero le quita detalle al gradient boosting (notebook 07).
4. **La escalera.** Las decisiones grandes se sostienen en desarrollo y en el test: corregir al MDA, más nodos para el gradient boosting y LEAR en proporción, y el modelado estadístico para el gradient boosting. Las demás dependen del modelo (notebook 08).
5. **Los picos no se anticipan con lo que se sabe en *d−1*.** Son sorpresas del tiempo real y del sistema, no de Monclova, y con 17 meses cada tipo de evento raro aparece una sola vez. Por eso cambiar de modelo según el riesgo del día no ayuda (notebooks 09 y 10).
6. **Combinar modelos ayuda, y cómo elegir importa.** Cada modelo aprovecha distinto la información, así que se prueban todas las mezclas de 2 y 3 modelos entre las versiones de la escalera y se elige con el peor mes de desarrollo; el error medio, usado al principio, y la mediana eligen mezclas que en el test rinden menos (notebooks 10 y 11).

## Cómo leer el proyecto

Cada notebook empieza con la pregunta, muestra lo que se observa, dice qué significa y termina con lo que se decide.

| Notebook | La pregunta |
|---|---|
| `01_extraccion` | ¿Son correctos los datos? Los 9 nodos de CENACE, 547 días × 24 horas, completos. |
| `02_diagnostico` | ¿Cómo son los datos y dónde se equivoca el MDA? |
| `03_exogenas` | ¿Qué información externa existe y se conoce a tiempo? Inventario con su regla de disponibilidad y una conclusión por fuente. |
| `04_protocolo` | ¿Cómo se mide? Periodos, MAE en MXN/MWh y en % contra el MDA, prueba de Giacomini–White, referencias, estimación del modelo base y prueba de no fuga. |
| `05_modelos_base` | **Decisión 1.** ¿Cuánto aporta el MDA? LEAR, redes y gradient boosting con y sin MDA. |
| `06_mas_nodos` | **Decisión 2.** Pocos ejemplos: entrenar con más nodos. Por qué esos 31, el escalamiento (solo Monclova, 31, 64) y las series efectivas. |
| `07_que_informacion_importa` | **Decisión 3.** Con 64 nodos: exógenas, modelado estadístico, todas las variables, qué pierde cada modelo sin cada familia y resumen con PCA. |
| `08_escalera` | Todos los modelos por los mismos pasos, en desarrollo y en el test. |
| `09_riesgo` | ¿Se puede reconocer el riesgo del día? Intervalos, picos y anomalías. |
| `10_combinacion` | Combinar modelos: piezas, por qué mezclar ayuda, pesos aprendidos, cambio de modelo según el riesgo y todas las mezclas de 2 y 3 modelos entre las versiones de la escalera. Modelo final. |
| `11_evaluacion_final` | El test (enero–junio de 2025): resultado, todos los modelos, comparación de criterios de elección (el mejor de desarrollo contra el mejor del test), ahorro acumulado, estabilidad, picos e intervalos, nodos. |

## Resultado en el test (enero–junio de 2025, 181 días)

"−6.9%" significa 6.9% **menos** error que el MDA. Se probaron todas las mezclas de 1, 2 o 3 modelos entre las 27 versiones de la escalera y el MDA, con pesos múltiplos de 0.1 que suman 1 (121,366 candidatas; por ejemplo, 0.3·TiDE con MDA + 0.7·GB con todas las variables). Cada modelo aprovecha de manera distinta la información, así que mezclar versiones de modelos diferentes combina las perspectivas en que cada uno se especializó. El **modelo final** se eligió con el peor mes de desarrollo:

**0.5·LEAR en proporción con nodos + 0.4·GB con todas las variables + 0.1·red sin MDA con todas las variables**

| Modelo | Desarrollo | Test (MXN/MWh) | Test vs MDA | Lugar en el test (de 121,366) |
|---|---|---|---|---|
| MDA (la referencia) | 221.6 | 183.4 | — | — |
| **Modelo final** (elegido con el peor mes) | −5.4% | **170.7** | **−6.9%** (*p* < 0.001, 5 de 6 meses) | 116 |
| Elegido con el error medio | −6.5% | 171.2 | −6.7% | 341 |
| Elegido con la mediana de los meses | −6.2% | 181.3 | −1.1% | 81,486 |
| Elegido con la tendencia | −1.3% | 174.6 | −4.8% | 17,591 |
| Mejor del test (se sabe después) | −5.1% | 169.5 | −7.6% | 1 |

- **Cómo se eligió.** Al principio del proyecto se usaba el error medio de todo el desarrollo. Lo influyen los movimientos extremos y las fluctuaciones no uniformes, naturales en datos con cambios abruptos de un mes a otro. La mediana de los meses ignora los meses malos y elige una mezcla inestable. El peor mes castiga justo esos meses y elige una mezcla que gana parejo en los seis meses de desarrollo; en el test queda a 0.7 puntos de la mejor posible. Con el peor mes se volvió a correr toda la elección (versiones de la escalera, mezclas y modelo final).
- **Consideraciones.** El peor mes se escogió entre cuatro criterios después de compararlos con el test, y la mezcla se eligió entre 121,366 candidatas con seis meses de desarrollo: el −6.9% es optimista y falta comprobar que el criterio se sostenga con un periodo más largo o con datos nuevos. La buena convergencia entre desarrollo (−5.4%) y test (−6.9%) es consistente con lo que busca el criterio: un modelo que nunca pierde mucho en un mes. Los hiperparámetros dentro de cada modelo se escogieron con el error medio (la suposición inicial); queda pendiente volver a escogerlos con el peor mes y reentrenar.
- **Ahorro:** unos 55,000 MXN por cada MWh pronosticado cada hora en el semestre; gana en 11 de los 12 meses desde julio de 2024 y solo pierde en enero de 2025.
- **Lo que se sostiene es mezclar LEAR en proporción con el GB**, con una pequeña parte de la red sin MDA, que sola pierde contra el MDA pero mezclada compensa los errores de las demás. Las diez mejores mezclas del test tienen esa forma.
- **Qué información importa:** el MDA con su historial de error y lo que describe cómo se mueve el precio. Las variables externas aportan poco y distinto según el modelo (combustibles y clima a LEAR; estado del sistema e información pública del día anterior al gradient boosting). Vale la pena explorarlas un poco, sin invertir demasiado.
- **Dónde gana y dónde pierde:** gana 10.5% en días sin pico y 17.9% en fines de semana; mejora poco en la noche (horas 19 a 22), donde está el error grande; pierde 1.2% en días con pico y 11.6% en enero de 2025.
- **Intervalos:** los cuantiles del 10% al 90% cubren 84% de las horas (81% si se entrenan con 64 nodos), pero solo 51% en días con pico.
- **Direcciones futuras:** depender menos del MDA, con un modelo de más instancias que aprenda el comportamiento del sistema y genere series distintas para aproximar el MTR sin sobreajustarse al MDA; anticipar los picos, donde se podría ganar un porcentaje grande, y confirmar el criterio de elección con datos nuevos.
- **Advertencias de proceso:** el test se abrió en versiones anteriores del proyecto, y esta organización de los experimentos y el criterio de elección se hicieron después de verlo.

El detalle está en `notebooks/11_evaluacion_final.ipynb` y `reports/11_*.csv`.

## Ideas probadas y descartadas

Se probaron en un solo modelo (o fueron indiferentes) y no aportaron; se quitaron de los notebooks para no meter ruido. Las cifras son del desarrollo, salvo que se diga.

| Idea | Resultado |
|---|---|
| Pronosticar la forma del día con PCA del perfil de 24 horas | 5% peor que LEAR (3, 5 y 8 componentes) |
| Modelo "tipo BERT" que rellena huecos de la serie del MTR | 3.4% peor que el MDA (el mejor); la versión original, 21% peor |
| Redes con cuello para el MDA, convolución y dos ramas (convolución del MDA + red de historia) | Entre 0.1% y 3% peor que la DNN de siempre; la mejor, un empate |
| Otras formas de plantear el problema: Johnson SU, descomposición en energía y congestión, agregación en línea de expertos | Sin mejora (la agregación en línea tiene un techo de −0.6%) |
| Red con variables externas | 0.7% peor que sin ellas |
| Clima en los intervalos | Los angosta (347 contra 412) pero cubre 74% en lugar de 80% |
| Rasgos de pico o corrección conformal en los intervalos | Sin mejora; cubren 73% en días con pico |
| Decir de qué nodo viene cada fila, o afinar con Monclova (GB con la red) | Indistinto, o peor (afinar deshace la ganancia) |
| Compuerta de riesgo y compuerta de anomalías (Isolation Forest) | −0.5% y −0.7% en desarrollo; en el test, −0.5% (sin significancia) y +1.9% |
| Cambiar de modelo según el riesgo (notebook 10) | No cambia nada: el puntaje de riesgo casi no se activa en desarrollo |
| Pesos aprendidos con el pasado (notebook 10) | Peor mes −0.9% contra −1.8% del reparto fijo del que parten; no se usan |
| Elegir exógenas una por una con LEAR y GB entrenados solo con Monclova (importancia por permutación y modelos reducidos) | El mejor LEAR reducido, −0.7% contra su base en desarrollo; reemplazado por la medición por familias del notebook 07 |
| Resumir todo en PCA sin rezagos, en LEAR, TiDE y GB | Casi siempre peor que dejar los rezagos tal cual |

El código de algunos de estos experimentos sigue en `src/pml/` (`enmascarado.py`, `pcatarget.py`, `dl.py` con las arquitecturas, `agregacion.py`, `componentes.py`, `seleccion.py`, `importancia.py`) y varios tienen sus pruebas en `tests/`.

## Cómo se compara

- **Desarrollo:** de 2024-07-01 a 2024-12-31. Ahí se decide todo.
- **Test:** de 2025-01-01 a 2025-06-30. Se evalúa en el notebook 11; la escalera del notebook 08 también lo muestra, solo como contraste.
- **Entrenamiento:** desde el 2024-02-05, solo con días ≤ *d−8* para el MTR. Los modelos se reentrenan cada día (LEAR) o cada semana (gradient boosting y redes).
- **Medida principal:** el error absoluto medio (MAE) de la zona, en MXN/MWh. Para comparar se usa el cambio porcentual contra el MDA, acompañado de "en cuántos de los 6 meses gana" y de una *p* de Giacomini–White: la probabilidad de ver una diferencia así de consistente si los dos modelos fueran igual de buenos.
- **Una sola prueba formal en el test:** el modelo final se fija en desarrollo, así que la pregunta es una (¿le gana al MDA?). Las *p* de las demás filas son contexto, y con 13 modelos algunas saldrán bien por azar.
- **Sin información del futuro:** cada variable pasa una prueba mecánica (notebook 04) y cada modelo tiene su prueba en `tests/`.
- Cada configuración probada queda registrada en `reports/experimentos.csv`.

## Pequeño glosario

- **MDA / MTR:** Mercado del Día en Adelanto y Mercado de Tiempo Real. El MDA se fija un día antes; el MTR, en la operación.
- **PML:** precio marginal local = energía + pérdidas + congestión. La energía es igual en todo el sistema.
- **LEAR:** regresión lineal con selección automática de variables, una por hora del día. Es el estándar de la literatura de precios eléctricos.
- **GB (gradient boosting):** muchos árboles de decisión pequeños que corrigen los errores de los anteriores.
- **DNN, NBEATSx, TiDE:** redes neuronales para series de tiempo.
- **"En proporción":** pronosticar cuánto se aleja el MTR del MDA en porcentaje, no en pesos.
- **"Red sin MDA":** un modelo que no ve el MDA y aprende solo del MTR pasado. Sirve para mezclarlo con el MDA.
- **Entrenar con 64 nodos:** usar como ejemplos de entrenamiento los precios de 64 nodos del sistema (31 elegidos a mano y el resto de los de 400 kV), pronosticando y evaluando solo Monclova.
- **Resumen PCA:** resumir muchas variables en 20 combinaciones (componentes principales), ajustadas solo con días hasta junio de 2024.

## Reproducir

```bash
pip install -r requirements.txt
python src/pml/download_all.py                     # descarga de CENACE; caché en data/raw/
PYTHONPATH=src python -m pml.red                   # nodos de la red (~1,000 llamadas, una vez)
python src/pml/servidor.py lista                   # los experimentos pesados (pasos 3 a 7 de la escalera e importancia); se corrieron en un servidor PBS
cd notebooks && python3 -m nbconvert --to notebook --execute --inplace 01_extraccion.ipynb   # y así hasta el 11
python3 -m pytest tests
```

Las cachés de los experimentos están en `data/processed/exp_cache` y `data/processed/dl_cache`. Sin ellas, los notebooks 05 a 08 tardan muchas horas (el gradient boosting con la red y las redes son lo más lento). `data/external/estados_operativos.csv` es una tabla armada a mano, con la fuente de cada fila. Las credenciales de EIA y FRED van en `Solution/.env` (`EIA_API_KEY`, `FRED_API_KEY`); ese archivo no se versiona y las llaves nunca se imprimen.

## Código (`src/pml/`)

- `cenace_client.py`, `download_all.py`, `data.py`: descarga del SW-PML de CENACE (bloques de 7 días, reintentos, caché) y matrices días × 24.
- `exog.py`, `explora.py`: fuentes externas (FRED, EIA, ERCOT, Open-Meteo), cada una con su regla de disponibilidad, y la exploración estándar de cada fuente contra el MTR.
- `features2.py`: grupos de variables (G0 a G16). `perturbar()` es la base de las pruebas de no fuga. `todo.py`: el conjunto de las 744 variables, los grupos de la escalera (exógenas, modelado estadístico, calendario) y las familias de la importancia.
- `backtest.py`, `exp.py`: LEAR, selección en desarrollo, tablas de comparación (`exp.comparar`) y registro de experimentos.
- `referencias.py`: carga en un solo paso los pronósticos establecidos de los notebooks 04 y 05.
- `servidor.py`: corre los pasos 3 a 7 de la escalera, la importancia por familias (LEAR, GB, TiDE, red sin MDA, con 64 nodos) y todas las mezclas, y los deja en las mismas cachés que leen los notebooks.
- `escalera.py`: la escalera de decisiones por modelo, la mejor versión de cada uno según el peor mes y su figura (notebooks 08 y 10).
- `mezclas.py`: todas las mezclas de 1, 2 y 3 pronósticos con pesos múltiplos de 0.1 y su calificación con cada criterio (notebooks 10 y 11; se calculan en el servidor).
- `gb2.py`, `picos.py`: gradient boosting (error medio o cuantiles) y clasificador de picos.
- `seleccion.py`: frecuencia de selección e importancia por permutación (experimento anterior; ver ideas descartadas).
- `dl.py`: DNN, NBEATSx, TiDE y una red lineal de control; versión entrenada con la red (`correr_dl_red`); arquitecturas de los experimentos descartados.
- `red.py`, `multinodo.py`: los nodos de la red y el gradient boosting y LEAR entrenados con ellos; pronostican solo Monclova.
- `importancia.py`: familias por prefijo de columna (experimento anterior; las familias actuales están en `todo.py`).
- `anomalias.py`: puntuación de días anómalos y detectores de riesgo con información de *d−1* (notebook 09).
- `mapas.py`: mapas de México con los nodos (ubicación aproximada por municipio).
- `metrics.py`: MAE, RMSE, pruebas de Giacomini–White y Diebold–Mariano, MAE por mes y el peor mes (criterio de elección).
- `features.py`: la transformación asinh y el calendario de feriados.

## Notas técnicas

- LightGBM requiere `libomp` del sistema; se usa `HistGradientBoostingRegressor` de scikit-learn.
- El PyTorch instalado (2.2) se compiló contra NumPy 1.x. `dl.py` pasa los datos sin el puente NumPy↔PyTorch, así que funciona con NumPy 2. Los procesos paralelos imprimen un aviso de compatibilidad que no afecta los resultados.

