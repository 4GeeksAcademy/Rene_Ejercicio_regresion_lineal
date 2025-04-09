from utils import db_connect
engine = db_connect()


# ----------------------------------
# Proyecto: Predicción del coste de seguros con regresión lineal
# Alumno: Rene R.
# Fecha: 08/04/2025
# ----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

url = "https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv"
df = pd.read_csv(url)

df_encoded = pd.get_dummies(df, drop_first=True)

# ModeLo global:
X_full = df_encoded.drop('charges', axis=1)
y_full = df_encoded['charges']
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
modelo_full = LinearRegression()
modelo_full.fit(X_train_f, y_train_f)
y_pred_f = modelo_full.predict(X_test_f)
print("\n[Modelo Global - TODAS las variables]")
print(f"MAE: ${mean_absolute_error(y_test_f, y_pred_f):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_f, y_pred_f)):.2f}")
print(f"R^2: {r2_score(y_test_f, y_pred_f):.4f}")

#Modelo reducido (variables clave)
X_red = df_encoded[['age', 'bmi', 'smoker_yes']]
y_red = df_encoded['charges']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_red, y_red, test_size=0.2, random_state=42)
modelo_red = LinearRegression()
modelo_red.fit(X_train_r, y_train_r)
y_pred_r = modelo_red.predict(X_test_r)
print("\n[Modelo Reducido - age, bmi, smoker_yes]")
print(f"MAE: ${mean_absolute_error(y_test_r, y_pred_r):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_r)):.2f}")
print(f"R^2: {r2_score(y_test_r, y_pred_r):.4f}")

# Clustering con KMeans sobre age, bmi y smoker ---
cluster_vars = df_encoded[['age', 'bmi', 'smoker_yes']]
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_vars)
kmeans = KMeans(n_clusters=2, random_state=42)
df_encoded['cluster'] = kmeans.fit_predict(cluster_scaled)

# --- Paso 6: Visualización de clusters (comentado si se corre como script) ---
# sns.scatterplot(x='age', y='bmi', hue='cluster', data=df_encoded, palette='Set1')
# plt.title("Clusters: Edad vs BMI")
# plt.show()

# Entrenamiento de regresión lineal por cluster ---
def entrenar_por_cluster(df, cluster_id):
    subset = df[df['cluster'] == cluster_id]
    X = subset[['age', 'bmi', 'smoker_yes']]
    y = subset['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n\U0001F4E6 Resultados para Cluster {cluster_id}")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2:.4f}")

# Ejecutar para ambos clusters
entrenar_por_cluster(df_encoded, cluster_id=0)
entrenar_por_cluster(df_encoded, cluster_id=1)

# --- Fin del codigo ---
# #Notas:
# - Se probaron modelos lineales: global, reducido, y con log-transform (descartado)
# - Se aplicó clustering con KMeans sobre age, bmi, smoker
# - Se entrenaron modelos separados por cluster y se compararon métricas
# - Este archivo resume toda la exploración y experimentación realizada.

# --- Conclusiones ---
"""
Conclusiones del Proyecto

- El modelo de regresión lineal global explica aproximadamente el 78% de la variabilidad en el costo del seguro (`charges`) con buen rendimiento general.
- El modelo reducido con solo `age`, `bmi`, `smoker_yes` obtuvo resultados similares, demostrando que esas variables son las más influyentes.
- La transformación logarítmica (`log(charges)`) fue descartada por bajo rendimiento.
- El clustering con `KMeans` permitió segmentar a los clientes en dos grupos diferenciados.
- Los modelos entrenados por cluster mostraron métricas similares al modelo global, con mejor interpretabilidad por perfil.

Este enfoque es útil para personalizar estimaciones de primas y ajustar modelos por perfil de cliente.
"""