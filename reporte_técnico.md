# Reporte Técnico: Experimentos de Machine Learning & Trazabilidad con MLflow

---

## 1. Descripción del Dataset

### Fuente de Datos
Para este proyecto se utilizó el dataset público **California Housing**, disponible a través de `sklearn.datasets.fetch_california_housing`. Los datos provienen del censo de 1990 de California y contienen métricas demográficas y de vivienda por cada bloque censal (*block group*).

### Diccionario de Datos

| Variable | Tipo de Dato | Descripción | Unidad / Rango |
| :--- | :--- | :--- | :--- |
| `MedInc` | Numérico (Continuo) | Ingreso mediano en el bloque censal | Decenas de miles de USD ($10k) |
| `HouseAge` | Numérico (Continuo) | Antigüedad mediana de las viviendas | Años (1 a 52) |
| `AveRooms` | Numérico (Continuo) | Promedio de habitaciones por hogar | Razón numérica |
| `AveBedrms` | Numérico (Continuo) | Promedio de dormitorios por hogar | Razón numérica |
| `Population` | Numérico (Continuo) | Población total en el bloque | Conteo de personas |
| `AveOccup` | Numérico (Continuo) | Promedio de ocupantes por hogar | Personas / hogar |
| `Latitude` | Numérico (Continuo) | Latitud del centro del bloque | Grados decimales |
| `Longitude` | Numérico (Continuo) | Longitud del centro del bloque | Grados decimales |
| `MedHouseVal` (Target) | Numérico (Continuo) | Valor mediano de la vivienda | Cientos de miles de USD ($100k) |

### Supuestos Clave
1. **Representatividad Geográfica:** Las coordenadas (`Latitude`, `Longitude`) capturan variaciones espaciales críticas en los precios de los bienes raíces (e.g., cercanía a la costa o zonas urbanas).
2. **Relación Ingreso-Precio:** El ingreso mediano (`MedInc`) es la variable con mayor poder explicativo sobre el precio del inmueble.

---

## 2. Proceso de Limpieza y Estandarización

El pipeline de preparación de datos fue encapsulado en `fuentes/datos_prep.py` para asegurar consistencia y evitar fugas de datos (*data leakage*):

1. **Manejo de Valores Nulos:** Se verificó la integridad del conjunto de datos original (`dropna()`). En caso de presencia de nulos, se aplica imputación por mediana basada en la distribución regional.
2. **Reglas de Negocio y Filtrado de Outliers:**
   - Se eliminaron registros atípicos donde el promedio de habitaciones (`AveRooms`) superaba 10.
   - Se truncó la variable objetivo `MedHouseVal` eliminando valores mayores a $5.0$, debido al sesgo de censura en el límite superior original del dataset.
3. **Ingeniería de Características (*Feature Engineering*):**
   - `BedroomsPerRoom`: Ratio de dormitorios por habitación (`AveBedrms / AveRooms`).
   - `PopulationPerHouse`: Estimación de densidad poblacional por hogar (`Population / AveOccup`).
4. **Escalado y Normalización:** Se aplicó `StandardScaler` de Scikit-Learn a las covariables continuas dentro del pipeline de entrenamiento (`train.py`), garantizando que la transformación se ajuste únicamente sobre los pliegues de entrenamiento (*train folds*).
5. **Codificación de Variables Categóricas:** No aplicable directamente al dataset de California Housing, ya que todas las variables son numéricas continuas.

---

## 3. Estrategia de Versionamiento

Se implementó una estrategia de versionamiento dual para garantizar trazabilidad y reproducibilidad total:

* **Versionamiento de Código y Configuración (Git):** 
  - Control de versiones mediante commits semánticos y tags en el repositorio GitHub.
  - Registro claro del historial de cambios en el archivo `CHANGELOG.md`.
* **Versionamiento de Datos:**
  - Separación física de los datos originales no modificados (`datos/datos_ini/`) y los datos limpios/transformados (`datos/datos_limp/`).
* **Trazabilidad de Experimentos (MLflow):**
  - Registro automatizado de parámetros, métricas (MAE, RMSE, $R^2$), código ejecutor (`train.py`), artefactos gráficos y binarios de los modelos dentro de la base de datos `mlflow.db`.

---

## 4. Visualizaciones Relevantes

Durante la fase de Análisis Exploratorio de Datos (EDA) en `entrena.ipynb`, se identificaron los siguientes patrones y anomalías:

1. **Correlación de Pearson:**
   - `MedInc` muestra una fuerte correlación positiva con `MedHouseVal` ($r \approx 0.68$).
   - `AveRooms` presenta una correlación moderada con la variable objetivo ($r \approx 0.15$).
2. **Distribución Geográfica:** 
   - El mapeo de `Latitude` y `Longitude` contra `MedHouseVal` revela picos de precio significativos en zonas costeras (San Francisco y Los Ángeles).
3. **Identificación de Anomalías:**
   - Se detectó un valor tope (*top-coding*) artificial en $5.0$ para `MedHouseVal`, el cual distorsionaba la distribución original y justificó la regla de negocio aplicada en el preprocesamiento.

---

## 5. Descripción de los Modelos Entrenados

Se evaluaron dos algoritmos con diferentes niveles de complejidad analítica para resolver el problema de regresión:

1. **Ridge Regression (Baseline Lineal):**
   - Regresión lineal con regularización $L_2$. Permite penalizar coeficientes excesivamente grandes y controlar problemas de multicolinealidad entre variables numéricas.
2. **XGBoost Regressor (Modelo Basado en Árboles):**
   - Algoritmo de *Gradient Boosted Decision Trees* (GBDT). Diseñado para capturar relaciones no lineales y complejas interacciones entre covariables geográficas y demográficas.

---

## 6. Configuración de Hiperparámetros

Se utilizó la técnica de **Validación Cruzada de 5 Pliegues ($k=5$)** combinada con **GridSearchCV** para encontrar la combinación óptima de hiperparámetros.

* **Ridge Regression:**
  - Espacio de búsqueda: `model__alpha`: `[0.1, 1.0, 10.0, 100.0]`
  - **Mejor valor seleccionado:** `alpha = 10.0`

* **XGBoost Regressor:**
  - Espacio de búsqueda:
    - `n_estimators`: `[100, 200]`
    - `max_depth`: `[4, 6]`
    - `learning_rate`: `[0.05, 0.1]`
  - **Mejor combinación seleccionada:** `learning_rate = 0.1`, `max_depth = 6`, `n_estimators = 200`

---

## 7. Métricas Obtenidas

Las métricas estandarizadas de regresión calculadas sobre el conjunto de evaluación (*test set*) corresponden exactamente a los valores registrados en MLflow:

| Modelo / Ejecución | Run ID | MAE | RMSE | $R^2$ Score | Tiempo |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ridge_Regression_Baseline** | `0a9d9eefed46...` | 0.4485 | 0.6043 | 0.6098 | 8.7s |
| **XGBoost_Optimized ★** | `0d6a4bfda309...` | **0.2813** | **0.4217** | **0.8100** | 36.2s |

---

## 8. Comparación Estadística de Resultados

1. **Rendimiento General:**
   - **XGBoost** superó ampliamente al modelo baseline lineal (Ridge) en todas las métricas evaluadas.
   - El coeficiente de determinación ($R^2$) aumentó del **60.98%** (Ridge) al **81.00%** (XGBoost), representando una mejora relativa del **+32.8%** en la variabilidad explicada.
2. **Reducción del Error:**
   - El Error Cuadrático Medio (**RMSE**) disminuyó de $0.6043$ a $0.4217$ (reducción del **-30.2%**).
   - El Error Absoluto Medio (**MAE**) disminuyó de $0.4485$ a $0.2813$ (reducción del **-37.3%**).
3. **Conclusión Estadística:**
   - Las no linealidades inherentes a la localización geográfica (`Latitude`, `Longitude`) y el ingreso medio son capturadas de manera significativamente superior por los árboles de decisión iterativos de XGBoost frente a una frontera de decisión lineal.

---

## 9. Evidencia del Uso de MLflow

Los experimentos fueron registrados de manera centralizada en el servidor local de MLflow respaldado por una base de datos SQLite (`sqlite:///mlflow.db`):

* **Experimento:** `California_Housing_Benchmark`
* **Parámetros Registrados:** `learning_rate`, `max_depth`, `n_estimators`, `model__alpha`.
* **Métricas Registradas:** MAE, RMSE y $R^2$.
* **Artefactos Guardados:**
  - Gráficos de residuos (`residuos_ridge_regression.png`, `residuos_xgboost.png`).
  - Modelos serializados en carpetas `model_ridge` y `model_xgb` (`MLmodel`, `model.pkl`, `conda.yaml`).

---

## 10. Reflexión sobre Deuda Técnica y MLOps

### Deuda Técnica Identificada
1. **Monitoreo de Desviación de Datos (*Data Drift*):** Actualmente no hay mecanismos automatizados para detectar si la distribución de los ingresos o precios cambia en el tiempo (e.g., inflación o cambios en el mercado inmobiliario).
2. **Orquestación del Pipeline:** La ejecución actual depende de la invocación manual de scripts (`datos_prep.py` y `train.py`), en lugar de utilizar un orquestador de flujos de trabajo como *Prefect*, *Airflow* o *MLflow Pipelines*.

### Madurez MLOps y Próximos Pasos
- **Registro de Modelos (Model Registry):** Se recomienda promover el modelo `XGBoost_Optimized` al estado `Production` dentro del Registro de Modelos de MLflow.
- **Despliegue Continuo (CD):** Empaquetar la inferencia del modelo optimizado en un contenedor Docker con una API REST desarrollada en FastAPI.