import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar datos
df = pd.read_csv("ads.csv")

# Procesamiento de la columna Time
df['Time'] = pd.to_datetime(df['Time'])
df['hour'] = df['Time'].dt.hour
df['dayofweek'] = df['Time'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# Definir variables
X = df[['hour', 'dayofweek', 'is_weekend']]
y = df['Ads']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo Árbol de Decisión
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# Modelo Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Métricas de evaluación
print("Árbol de Decisión:")
print("MSE:", mean_squared_error(y_test, dt_preds))
print("R2:", r2_score(y_test, dt_preds))

print("\nRandom Forest:")
print("MSE:", mean_squared_error(y_test, rf_preds))
print("R2:", r2_score(y_test, rf_preds))
