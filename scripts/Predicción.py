# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 10:25:14 2025

@author: Admin
"""

import pandas as pd

# NumPy: operaciones numéricas (lo usaremos para algunas funciones)
import numpy as np

# Matplotlib y Seaborn: visualización básica (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Importas la clase OneHotEncoder, que sirve para convertir texto en números mediante One Hot Encoding.
from sklearn.preprocessing import OneHotEncoder

# StandardScaler: escalamiento de variables numéricas
from sklearn.preprocessing import StandardScaler

# train_test_split: separar datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Opcional: un modelo simple para demostrar uso (ej: Regresión Logística)
from sklearn.linear_model import LogisticRegression

import category_encoders as ce

# ===== IMPORTES ADICIONALES PARA MODELOS Y PIPELINES =====

# Modelos supervisados
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Modelo no supervisado
from sklearn.cluster import KMeans

# Métricas
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Herramientas avanzadas
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Guardar pipeline entrenado
#PIPELINES permite encadenar todos los pasos de tu proceso de Machine Learning en un solo flujo ordenado, desde el preprocesamiento hasta el modelo final.

import joblib

# Configuración estética opcional para los gráficos
plt.style.use("ggplot")

# Fijar semilla para reproducibilidad

np.random.seed(42)

# Carga de datos

df1 = pd.read_csv("base_unida_l.csv", sep=",")

df = df1.copy()

print(df.dtypes)

# SELECCIÓN DE VARIABLES (X, y)

NOMBRE_TARGET = "precio_promedio_por_litro"

if NOMBRE_TARGET not in df.columns:
    raise ValueError(
        f"ERROR: La columna objetivo '{NOMBRE_TARGET}' no existe en df_modelo.\n"
        f"Columnas disponibles: {list(df.columns)}"
    )

# y: lo que queremos predecir 
y = df[NOMBRE_TARGET]

# X: todas las demás columnas (features)
X = df.drop(NOMBRE_TARGET, axis=1)

print("Shape X:", X.shape)
print("Shape y:", y.shape)

print("\nDistribución de la variable objetivo (y):")
print(y.value_counts(normalize=True))

# Escalamiento de variables numéricas

columnas_num_X = X.columns
print("Columnas numéricas a escalar:", list(columnas_num_X))

# Creamos el StandardScaler
escalador = StandardScaler()

# Ajuste y transformación
X_escalado = X.copy()
X_escalado[columnas_num_X] = escalador.fit_transform(X[columnas_num_X])

print("\nPrimeras filas de X_escalado:")
print(X_escalado.head())


# DIVISIÓN EN TRAIN Y TEST
# ============================================================

print("\n===== 12. DIVISIÓN EN TRAIN / TEST =====")

X_train, X_test, y_train, y_test = train_test_split(
    X_escalado,
    y,
    test_size=0.2,
    random_state=42
)

print("Shape X_train:", X_train.shape)
print("Shape X_test :", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test :", y_test.shape)

y.value_counts()

# Diccionario de modelos a evaluar
modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

resultados_modelos = []

for nombre, modelo in modelos.items():
    print("\n--------------------------------------------------------")
    print(f"Entrenando modelo supervisado: {nombre}")
    
    
# Entrenamiento
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

modelo = RandomForestRegressor(random_state=42)
modelo.fit(X_train, y_train)

# Predicción
y_pred = modelo.predict(X_test)

# Metricas 

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


# ===== 1. Entrenar K-Means (modelo NO supervisado) =====
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_escalado)

# ===== 2. Etiquetas de cluster =====
labels_kmeans = kmeans.labels_

print("\nTamaño de cada cluster:")
valores, cuentas = np.unique(labels_kmeans, return_counts=True)
for cl, c in zip(valores, cuentas):
    print(f"Cluster {cl}: {c} registros")


# ===== 3. Comparar clusters con el Precio real =====
# Creamos una tabla cruzada: cluster vs rango de precios
df_temp = df.copy()
df_temp["cluster"] = labels_kmeans

# Crear categorías de precio (bajo, medio, alto) para comparar
df_temp["categoria_precio"] = pd.qcut(df_temp["precio_promedio_por_litro"], q=3, labels=["Bajo", "Medio", "Alto"])

print("\nTabla cruzada: Cluster vs Categoría de Precio")
print(pd.crosstab(df_temp["cluster"], df_temp["categoria_precio"]))

# PIPELINE + GRIDSEARCHCV (FLUJO COMPLETO)

#1. Separar X e y6+

# X_raw: dataset original sin la columna objetivo

y1 = df_temp["precio_promedio_por_litro"]
X1 = df_temp.drop("precio_promedio_por_litro", axis=1)

# 2. Dividir en train y test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Crear Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

pipeline_rf = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(random_state=42))
])

# 4. Definir hiperparámetros para GridSearchCV
param_grid = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth": [5, 10, 20, None],
    "rf__min_samples_split": [2, 5]
}

# 5. Ejecutar GridSearch
from sklearn.model_selection import GridSearchCV

grid_rf = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("Entrenando GridSearchCV... esto puede tardar un poco.")
grid_rf.fit(X_train, y_train)

# 6. Mejores hiperparámetros
print("Mejores hiperparámetros:")
print(grid_rf.best_params_)

print("\nMejor puntuación R² en validación cruzada:")
print(grid_rf.best_score_)

# 7. Evaluación final en el conjunto de prueba
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

best_model = grid_rf.best_estimator_

y_pred = best_model.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_pred)

print("\nDesempeño final del mejor modelo:")
print("MAE:", MAE)
print("MSE:", MSE)
print("RMSE:", RMSE)
print("R2:", R2)

# 8. Guardar el modelo entrenado
import joblib

joblib.dump(best_model, "pipeline_rf_precios.pkl")
print("\nModelo guardado como pipeline_rf_precios.pkl")

# PREDICCIONES FINALES Y EXPORTACIÓN

import joblib
import pandas as pd
# 1. Cargar el pipeline final ya entrenado
NOMBRE_PIPELINE = "pipeline_rf_precios.pkl"
modelo_final = joblib.load(NOMBRE_PIPELINE)

# 2. Hacer predicciones sobre el set de prueba
y_pred_final = modelo_final.predict(X_test)

# 3. Construir el DataFrame de resultados
df_resultados = pd.DataFrame({
    "precio_real": y_test,
    "precio_pred": y_pred_final
}, index=y_test.index)

# 4. Guardar resultados a un CSV
NOMBRE_RESULTADOS = "predicciones_precios_modelo.csv"
df_resultados.to_csv(NOMBRE_RESULTADOS, index=True)

print(f"✅ Archivo de predicciones exportado como: {NOMBRE_RESULTADOS}")

# Gráfica real vs predicho

import matplotlib.pyplot as plt

# Crear figura
plt.figure(figsize=(8, 6))

# Gráfica Real vs Predicho
plt.scatter(y_test, y_pred)

# Línea de referencia: y = x (predicción perfecta)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()])

plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("Gráfico Real vs Predicho")

plt.show()

