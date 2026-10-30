# reports/

Cada archivo lleva el número del notebook que lo genera. Los que terminan en `_dev` son del periodo de desarrollo (julio–diciembre de 2024). Los de `11_*` y `08_escalera.csv` incluyen el test (enero–junio de 2025).

## Columnas comunes

- **MAE:** error absoluto medio en MXN/MWh, sobre la zona (promedio de los 9 nodos), todas las horas.
- **MAE sin días de pico:** lo mismo, sin los días en que algún MTR horario superó 3,000 MXN/MWh.
- **vs MDA (%):** cambio del MAE contra el MDA como pronóstico, (MAE del modelo − MAE del MDA) / MAE del MDA. Negativo es mejor.
- **vs referencia (%):** lo mismo, contra la referencia de esa fila (por ejemplo, el mismo modelo sin la idea que se prueba).
- **meses mejor:** en cuántos de los 6 meses el modelo tuvo menor error que su referencia.
- **p:** probabilidad de ver una diferencia así de consistente si el modelo y su referencia fueran igual de buenos (prueba de Giacomini–White sobre el error diario). Menos de 0.05 indica que es difícil que sea suerte.
- **AUC** (picos): probabilidad de que un día con pico reciba mayor probabilidad que uno sin pico. 0.5 es azar y 1 es perfecto.
- **Brier** (picos): error cuadrático de la probabilidad; menor es mejor. Se compara con el Brier de decir siempre la tasa real de picos del periodo.

## Archivos por notebook

- `experimentos.csv`: registro de todas las configuraciones probadas, con su MAE de desarrollo. Sirve para dimensionar cuántas pruebas se hicieron.
- **02:** `02_dias_con_pico.csv`, cuántos días con pico hay en cada semestre y qué parte del error del MDA cargan.
- **03:** `03_resumen_correlaciones.csv`, relación de cada fuente externa con el nivel del MTR, con la diferencia MTR − MDA y con su magnitud.
- **04:** `04_referencias_dev.csv` y `04_estimador_dev.csv`, las referencias y las decisiones de estimación del modelo base.
- **05 (modelos base):** `05_variantes_dev.csv` (redes con y sin MDA), `05_gb_dev.csv` (gradient boosting solo con Monclova) y `05_proporcion_dev.csv` (medir en proporción). Figura `05_correccion_vs_sorpresa.png`.
- **06 (más nodos):** `06_gb31_dev.csv` (31 nodos), `06_troncal_dev.csv` (64 nodos en todos los modelos), `06_gb_base64_dev.csv` (el GB del pipeline común, con 64 nodos y solo las variables base), `06_series_efectivas_dev.csv` y `06_escalamiento_dev.csv`. Figuras `06_parecido_nodos.png`, `06_mapa_parecido.png`, `06_mapa_grupos.png`, `06_series_efectivas.png` y `06_escalamiento.png`.
- **07 (qué información importa):** `07_ramas_dev.csv` (exógenas y modelado estadístico sobre la base con 64 nodos), `07_todas_dev.csv`, `07_importancia_dev.csv` (qué pierde LEAR y el GB sin cada familia), `07_pca_grupos.csv`, `07_pca_cargas.csv` y `07_pca_dev.csv`. Figuras `07_complejidad.png`, `07_importancia_familias.png` y `07_pca_cargas.png`.
- **08 (escalera):** `08_escalera.csv` (cada modelo en cada paso, en desarrollo y en el test, con su error medio, su cambio contra el MDA y el peor mes). Figura `08_escalera.png`.
- **09 (riesgo):** `09_intervalos_dev.csv`, `09_picos_clasificador_dev.csv` y `09_detectores_dev.csv`. Figuras `09_anomalias_*.png` y `09_riesgo_vs_error.png`.
- **10 (combinación):** `10_piezas_dev.csv` (la mejor versión de cada modelo) y `10_mezclas_mejores_dev.csv` (las 10 mejores mezclas de 2 y 3 modelos según el peor mes de desarrollo). Figuras `10_correlacion_errores.png`, `10_peso_mda.png` y `10_pesos_aprendidos.png`.
- **11 (evaluación final):**
  - `11_conjunto_test.csv`: las 10 mejores mezclas del test, el modelo final, la mejor de desarrollo y el MDA, con su lugar y su error en desarrollo y en el test;
  - `11_por_regimen.csv`: tipo de día, calendario y mes;
  - `11_intervalos_test.csv` y `11_picos_test.csv`: intervalos y clasificador de picos en el test;
  - `11_criterios_eleccion.csv`: qué mezcla elige cada criterio (error medio, mediana, peor mes y tendencia) con los meses de desarrollo, cómo le va en el test y qué tanto la calificación ordena a las mezclas como el test;
  - `11_por_nodo.csv`: los 9 nodos.
  - Figuras: `11_conjunto_test.png`, `11_desarrollo_vs_test.png`, `11_criterios_eleccion.png`, `11_ahorro_acumulado.png`, `11_estabilidad.png`, `11_mae_hora.png` y `11_pronostico_vs_real.png`.
- `figures/`: gráficas, con el mismo prefijo que el notebook que las genera.
