# Fraud Prevention Fintech — MercadoLibre Data Scientist Challenge

Solución de Machine Learning para predecir transacciones fraudulentas **maximizando la ganancia
del negocio** (no solo la exactitud de la clasificación), bajo la regla de costos: +25% de margen
por transacción legítima aprobada, -100% del monto por fraude aprobado.

El detalle completo del análisis, resultados y conclusiones está consolidado en
[`04_results/Resumen_Proyecto_Fraude.pdf`](04_results/Resumen_Proyecto_Fraude.pdf) (también
disponible en `.pptx`). Este README documenta la arquitectura del repositorio y el contenido
técnico de cada notebook para quien necesite ejecutar o auditar el código.

## Orden de ejecución

El proyecto está diseñado para ejecutarse **secuencialmente, carpeta por carpeta, y dentro de
cada carpeta en orden numérico de archivo**:

```
00_analisis_inicial  →  01_EDA  →  02_feature_engineer  →  03_model_selection  →  04_results
```

Cada notebook lee el/los artefacto(s) que dejó el paso anterior (CSV en `dataset/`, configs en
`src/configs/`, o artifacts serializados en `03_model_selection/artifacts/`), por lo que no deben
ejecutarse fuera de este orden. Dentro de `01_EDA/`, `02_feature_engineer/` y
`03_model_selection/`, los notebooks también están numerados (`00_`, `01_`, `02_`) y deben
correrse en ese orden.

`05_monitoring/` es un paso adicional, independiente de esa cadena (solo depende de
`dataset/train.csv`, que ya existe desde el inicio): simula el monitoreo de drift que se haría
sobre el modelo una vez en producción, y puede ejecutarse en cualquier momento después de
`00_analisis_inicial`.

## Arquitectura del proyecto

```
DS_pro/
├── 00_analisis_inicial/     # Lectura del enunciado, planteamiento del problema y función de costo
│   ├── 00_analisis_inicial.ipynb
│   ├── enunciado/           # PDF del challenge original
│   └── docs/                # Papers de soporte (cost-sensitive learning, leakage, missing indicators...)
│
├── 01_EDA/                  # Exploración de datos crudos y del split de entrenamiento
│   ├── 01_EDA_raw.ipynb
│   ├── 02_EDA_train.ipynb
│   ├── figures/, figures_train/   # PNGs exportados por los notebooks
│   └── tables/, tables_train/     # CSVs exportados (estadísticos, correlaciones, VIF, etc.)
│
├── 02_feature_engineer/     # Construcción, validación estadística y agregación de features
│   ├── 00_feat_engineer_initial.ipynb
│   ├── 01_statistical_analysis.ipynb
│   ├── 02_aggregation_features.ipynb
│   ├── stats_outputs/       # Resultados de las pruebas estadísticas por feature
│   └── agg_stats_outputs/
│
├── 03_model_selection/      # Entrenamiento, evaluación, elección de modelo y simulación de prod.
│   ├── 00_train_models.ipynb
│   ├── 01_evaluate_models.ipynb
│   ├── 02_simulation_model_prod.ipynb
│   └── artifacts/           # Un pipeline serializado por modelo (ver más abajo)
│
├── 04_results/              # Entregables finales
│   ├── Resumen_Proyecto_Fraude.pdf
│   └── Resumen_Proyecto_Fraude.pptx
│
├── 05_monitoring/           # Simulación de monitoreo de drift post-despliegue
│   └── 00_drift_monitoring.ipynb
│
├── dataset/                  # CSVs de entrada/intermedios (crudo, train/test, features derivadas)
│
├── src/                       # Código reutilizable importado por los notebooks
│   ├── configs/                # Configuración declarativa del feature engineering
│   │   ├── feature_engineer_config.yml     # Transformaciones por columna (usado en 02_feature_engineer)
│   │   └── preprocessing_config.json       # Config general del ColumnTransformer de producción
│   ├── transformers_utils/     # Pipeline de PRODUCCIÓN (sklearn-compatible, serializable)
│   │   ├── prod_transformers.py    # ~20 Transformers (BaseEstimator/TransformerMixin)
│   │   ├── preprocessing_pipeline.py  # build_preprocessor(), filter_config_by_features()
│   │   └── _vectorized_ops.py       # helpers vectorizados compartidos por los transformers
│   ├── feature_engineer_utils/  # Funciones de LABORATORIO (pandas puro, usadas en notebooks EDA/FE)
│   │   ├── preprocessing_features_laboratory.py  # build_* equivalentes "de exploración" + clase `preprocessing`
│   │   ├── statistical_tests.py    # pruebas univariadas + clase `FeatureStatTests`
│   │   └── aggregation_features.py # ventanas móviles causales (equivalente de laboratorio)
│   ├── models/
│   │   ├── model_evaluation.py     # métricas de negocio, gráficos, persistencia de artifacts
│   │   └── model_utils.py          # train_test_split_time() (split temporal con align_to_day)
│   ├── monitoring_utils/
│   │   └── drift_tests.py          # ks_drift(), psi_categorical() (usado en 05_monitoring)
│   └── utils/
│       ├── eda.py                  # clase `EDA` (helpers de exploración reutilizados en 01_EDA)
│       └── graphics_utils.py       # gráficos de evolución diaria / impacto económico
│
└── requirements.txt
```

### Dos implementaciones paralelas de las mismas transformaciones

Un detalle de arquitectura importante: **el feature engineering existe dos veces**, a propósito:

- **`src/feature_engineer_utils/`** — funciones de laboratorio sobre `pandas.DataFrame` completos,
  usadas en `01_EDA` y `02_feature_engineer` para explorar, graficar y decidir qué features
  construir.
- **`src/transformers_utils/`** — los mismos cálculos reimplementados como `Transformer`
  de scikit-learn (`fit`/`transform`, vectorizados en `_vectorized_ops.py`), usados en
  `03_model_selection` dentro de un `ColumnTransformer`/`Pipeline` real que **sí se serializa y se
  reproduce en producción** (ver `02_simulation_model_prod.ipynb`).

Esto evita que la fase exploratoria (rápida de iterar, pero no productiva) contamine el pipeline
que efectivamente se despliega.

## Detalle técnico por notebook

### `00_analisis_inicial/00_analisis_inicial.ipynb`

- Traduce el enunciado del challenge en un problema de **optimización de ganancia bajo una función
  de costo asimétrica**, no una clasificación estándar.
- Deriva la función de negocio paso a paso a partir de la matriz de costos (TP=0, FP=-25%·monto,
  FN=-100%·monto, TN=+25%·monto):

  $$J_{general} = 0.25 \cdot TN - 1.00 \cdot FN - 0.25 \cdot FP = \underbrace{0.25 \cdot N_{leg}}_{C \text{ (constante)}} - 1.00 \cdot FN - 0.25 \cdot FP$$

  Como $C$ es constante (no depende del modelo), maximizar $J_{general}$ es equivalente a
  maximizar solo la parte variable:

  $$\boxed{J = -1.00 \cdot FN - 0.25 \cdot FP}$$

- Deriva el umbral óptimo teórico según Elkan (2001):
  $p^{*} = \frac{Cost(FP)}{Cost(FP)+Cost(FN)} = \frac{0.25}{1.25} = 0.20$ (referencia teórica; en la
  práctica el umbral se calibra por modelo maximizando $J$ sobre test — ver `01_evaluate_models`).
- Referencia bibliográfica de las 4 fuentes en `docs/` (cost-sensitive learning, missing
  indicators, data leakage/sampling en XGBoost, árboles vs. deep learning en datos tabulares).

### `01_EDA/01_EDA_raw.ipynb`

- Dataset crudo: **150,000 filas, 19 columnas**, periodo `2020-03-08` → `2020-04-21` (44 días).
- Identifica columnas candidatas a ID (alta cardinalidad, sin valor predictivo directo).
- Distribución del target `fraude`: **7,500 casos (5.0%)**, con tasa diaria fluctuando entre ~3% y
  ~6% sin tendencia marcada.
- Evolución diaria del monto de fraude, y una tabla de **impacto económico bajo la regla de
  negocio** (25% ganancia / 100% pérdida), asumiendo que el fraude confirmado en el dataset es
  fraude que no se logró contener.
- Análisis de concentración de variables categóricas (`a`, `g`, `j`, `o`, `p`) — nulos, cardinalidad
  y distribución de frecuencias.

### `01_EDA/02_EDA_train.ipynb`

- **Split temporal** vía `train_test_split_time` (`src/models/model_utils.py`), `test_size≈0.311`
  (14 de 45 días), con `align_to_day=True` para no fraccionar un mismo día entre train y test.
  Train: `2020-03-08→2020-04-09` (104,981 filas, 5.18% fraude). Test: `2020-04-10→2020-04-21`
  (45,019 filas, 4.57% fraude).
- Multicolinealidad: correlación `d`~`m` = 0.59, VIF máximo = 1.61, número de condición de la
  regresión logística = 2.2 → sin multicolinealidad severa (todas las variables continuas pueden
  incluirse).
- Sesgo (skewness) y outliers por variable numérica, diferenciado por target.
- Asociación de categóricas con el target vía **V de Cramér** + chi-cuadrado: `j` (≈0.30) y `o`
  (≈0.29) son las más fuertes, `n` (≈0.17) también relevante.
- Tasa de fraude (fraud rate) y **lift** por categoría — ej. `o="N"` → 22.8% fraude (lift 4.4x)
  frente a un baseline de 5.0%.

### `02_feature_engineer/00_feat_engineer_initial.ipynb`

- Enfoque **config-driven**: todas las transformaciones están declaradas en
  [`src/configs/feature_engineer_config.yml`](src/configs/feature_engineer_config.yml) (campos
  `process`, `column`, `new_column`, `params`) y se ejecutan vía la clase `preprocessing`
  (`src/feature_engineer_utils/preprocessing_features_laboratory.py`), sin lógica hardcodeada en
  el notebook.
- Produce **61 features nuevas** a partir de las columnas crudas, aplicando cada técnica solo
  cuando es estructuralmente compatible y no degenerada (p. ej. no se agrega `missing_flag` si la
  columna no tiene nulos).
- Técnicas aplicadas (agrupadas por tipo):
  - **Calidad/distribución**: `missing_flag`, `zero_flag`, `top_percentile_flag` (q=0.99),
    `log_feature` (log1p), `quantile_bins_dropna` (5 bins).
  - **Encoding categórico**: `frequency_encoding`, `rare_grouping`,
    `target_encoding_with_cross_val` (mean encoding con cross-validation para evitar leakage).
  - **Temporales** (sobre `fecha`): `hour`, `weekday`, `is_weekend_flag`,
    `hour_cyclic_sin_cos` (codificación seno/coseno de la hora).
  - **Reducción/combinación**: `pca_components` (1 componente para `d`+`m`, correlacionadas),
    `ratio` (`monto`/`l`).
- Exclusiones deliberadas y documentadas: `k` (ID 100% único, solo se usa como llave); `n`, `p`
  (binarias — su encoding sería una función biyectiva del valor crudo, no agrega información);
  `o` no recibe `missing_flag` porque su nulo (72.4%) ya es la señal más fuerte del dataset y se
  conserva como categoría explícita.
- Salida: `dataset/train_feat_engineer.csv`.

### `02_feature_engineer/01_statistical_analysis.ipynb`

- Cuantifica la relevancia individual de cada variable (original y derivada) frente al target,
  vía la clase `FeatureStatTests` (`src/feature_engineer_utils/statistical_tests.py`):
  - **Variables continuas**: Mann-Whitney U, Kolmogorov-Smirnov, correlación punto-biserial,
    ROC-AUC univariado, Cohen's d, test de Wald (regresión logística univariada).
  - **Variables categóricas**: Chi-cuadrado, V de Cramér, Information Value (IV), Weight of
    Evidence (WOE), test de razón de verosimilitud (LR test) logístico.
- Compara features originales vs. nuevas y produce un **ranking final** de todas las features
  frente al target (`stats_outputs/ranking_final_features.csv`, `resumen_final_features.csv`).

### `02_feature_engineer/02_aggregation_features.ipynb`

- **Velocity features**: agregaciones móviles (rolling) de `monto`, agrupadas por la entidad
  categórica `j`, sobre ventanas de **1h, 1d, 3d, 7d y 10d**.
- Estadísticos por ventana: suma, promedio, mediana, máximo, mínimo (5 estadísticos × 5 ventanas).
- **Diseño causal**: implementado con `closed="left"` (tanto en la versión de laboratorio,
  `src/feature_engineer_utils/aggregation_features.py`, como en la de producción,
  `_BaseWindowTransformer` en `src/transformers_utils/prod_transformers.py`), de forma que cada
  agregación usa únicamente información estrictamente anterior a la transacción actual — evita
  data leakage temporal.
- Consolida el resultado con `train_feat_engineer.csv` → `dataset/train_all_features.csv` /
  `train_agg_features.csv`.

### `03_model_selection/00_train_models.ipynb`

- Construye el `ColumnTransformer` de producción vía `build_preprocessor()` +
  `filter_config_by_features()` (`src/transformers_utils/preprocessing_pipeline.py`), aplicando
  `src/configs/preprocessing_config.json` filtrado a un conjunto curado de features (columnas
  passthrough + encodings categóricos + features engineered seleccionadas + 1 agregación
  temporal — **23 features en total** para el baseline, iguales para los 5 modelos).
- Entrena **5 modelos candidatos**: Logistic Regression, Random Forest, LightGBM, CatBoost,
  XGBoost.
- Cada modelo se persiste vía `save_model_artifact()` en su propia subcarpeta bajo
  `03_model_selection/artifacts/`: el `Pipeline` completo (preprocesamiento + `post_encode` +
  clasificador) serializado en `pipeline.joblib`, más el formato nativo del algoritmo cuando aplica
  (`model.txt` LightGBM, `model.json` XGBoost, `model.cbm` CatBoost) y `metadata.json`.

### `03_model_selection/01_evaluate_models.ipynb`

- Carga los 5 artifacts (`load_all_model_artifacts`) y construye la tabla comparativa vía
  `evaluate_fitted_models()` (`src/models/model_evaluation.py`): AUC, PR-AUC, Precision/Recall a
  **TH fijo (0.5)** y a **TH óptimo** (calibrado por `sweep_business_profit`, que maximiza
  $J$ sobre el test real), $J(\$)$ en ambos umbrales, y $J(\$)$ *block all* (referencia de piso:
  rechazar el 100% de las transacciones, **-\$428,970.04**, igual para los 5 modelos).
- Resultado final (ranking por $J$ óptimo):

  | Modelo | AUC | PR-AUC | J óptimo ($) | Mejora al calibrar TH |
  |---|---|---|---|---|
  | **LightGBM** | 0.8850 | 0.4354 | **-86,068** | +6.0% |
  | CatBoost | 0.8805 | 0.4368 | -87,887 | +3.4% |
  | XGBoost | 0.8681 | 0.4203 | -90,194 | +1.3% |
  | Random Forest | 0.8673 | 0.4049 | -93,752 | +11.0% |
  | Logistic Regression | 0.7525 | 0.1907 | -127,171 | +39.6% |

- Grafica precision-recall, ganancia de negocio vs. threshold, matriz de confusión y feature
  importance (normalizada a %) por modelo, y documenta fortalezas/limitaciones de cada uno
  (p. ej. XGBoost concentra 38.0% de su importancia en `o_freq_encoded` → mayor riesgo de concept
  drift si esa distribución cambia en producción).
- **Modelo elegido: LightGBM** — mejor $J$ óptimo (empatado con CatBoost, <1% de diferencia) y
  feature importance más distribuida entre los 5 modelos (menor dependencia de una sola señal).

### `03_model_selection/02_simulation_model_prod.ipynb`

- Simula el proceso de producción **solo con el modelo elegido (LightGBM)**, cargando
  únicamente `artifacts/LightGBM/pipeline.joblib` (sin reentrenar):
  1. **Reproducibilidad**: scorea `test.csv` como si fueran transacciones nuevas — AUC
     reproducido = 0.8850, idéntico al reportado en la evaluación.
  2. **Scoring transacción a transacción**: simula una sola transacción llegando como request de
     API.
  3. **Robustez ante categorías nunca vistas**: inyecta valores inexistentes en entrenamiento en
     `j`/`g` — el pipeline no falla, el score simplemente se ajusta.
- Valida la condición necesaria para un despliegue confiable: el pipeline serializado es
  autocontenido y reproducible fuera del entorno de entrenamiento.

### `05_monitoring/00_drift_monitoring.ipynb`

- Simula el monitoreo de drift descrito en la slide "Consideraciones adicionales" del resumen del
  proyecto: parte `dataset/train.csv` en **5 semanas** (bins de 7 días desde la fecha mínima) y
  compara cada semana contra la semana 0 como referencia/baseline, vía
  `src/monitoring_utils/drift_tests.py`:
  - **Variables continuas** (`b, c, d, e, f, h, l, m, monto, score`): estadístico **KS
    (Kolmogorov-Smirnov)** sobre los valores no nulos.
  - **Variables categóricas** (`a, g, j, n, o, p`): **Population Stability Index (PSI)**, con
    nulos tratados como categoría propia.
  - Tasa de nulos por semana, monitoreada aparte del KS (que solo ve los valores no nulos).
- **Hallazgo principal**: `b` y `c` tienen 47.8% de nulos en la semana 0 y caen a ~1% desde la
  semana 1 (con un patrón errático día a día dentro de la semana 0) — invisible para el KS,
  compatible con una inestabilidad puntual del feed de datos más que con un drift natural.
- Ninguna feature alcanza "drift significativo" (KS > 0.20 / PSI > 0.25) sobre los valores no
  nulos; `b`, `h` y `monto` muestran "drift leve" (KS ≈ 0.10-0.11) hacia las semanas 3-4,
  coincidiendo con el aumento de la tasa de fraude semanal en ese tramo.
- Tabla resumen final: feature, tipo, métrica usada, valor máximo observado, semana en que ocurre,
  y una etiqueta de alerta (sin drift / leve / significativo) con formato condicional.

## Diccionario de datos (columnas del dataset crudo)

Todas las columnas están anonimizadas; lo que se conoce de cada una viene del EDA y está
documentado en los comentarios de `src/configs/feature_engineer_config.yml`:

| Columna | Tipo | Notas |
|---|---|---|
| `k` | ID | 100% valores únicos — se usa solo como llave, nunca como feature |
| `a` | categórica | 4 niveles, sin nulos/ceros |
| `b` | numérica | rango [0,1], 12% nulos, pocos ceros |
| `c` | numérica | 12% nulos, skew alto, sin ceros |
| `d` | numérica | pocos nulos/ceros, correlacionada con `m` (0.59) |
| `e` | numérica | sin nulos, 44% ceros, skew muy alto |
| `f` | numérica | tiene negativos, pocos nulos, 17% ceros |
| `g` | categórica | 45 niveles, algo de nulos, muy concentrada en `"BR"` |
| `h` | numérica entera | sin nulos, 8.9% ceros |
| `j` | categórica | alta cardinalidad (7,558+ categorías tipo ID); mayor asociación con el target (V de Cramér ≈0.30) |
| `l` | numérica | casi sin nulos, 1.35% ceros |
| `m` | numérica | pocos nulos, 10.6% ceros, correlacionada con `d` |
| `n` | categórica binaria | sin transformaciones adicionales (encoding de una binaria no agrega información) |
| `o` | categórica | 72.4% nulo — el nulo es la señal más predictiva del dataset (2da mayor V de Cramér ≈0.29) |
| `p` | categórica binaria | sin transformaciones adicionales |
| `fecha` | timestamp | rango 2020-03-08 → 2020-04-21 |
| `monto` | numérica | monto de la transacción; sin nulos/ceros, ya fuertemente asociada al target |
| `score` | numérica | score de riesgo preexistente; sin nulos, 3.2% ceros |
| `fraude` | target | binario (0=legítima, 1=fraude); 5.0% de tasa de fraude global |

## Cómo ejecutar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> En macOS, LightGBM y XGBoost requieren `libomp` a nivel de sistema: `brew install libomp`
> (CatBoost no lo necesita, trae su propio runtime).

Luego correr los notebooks en el orden indicado arriba (`00_analisis_inicial` → `01_EDA` →
`02_feature_engineer` → `03_model_selection`), cada uno de principio a fin antes de pasar al
siguiente. `05_monitoring/00_drift_monitoring.ipynb` es independiente de esa cadena y puede
ejecutarse en cualquier momento (solo requiere `dataset/train.csv`).
