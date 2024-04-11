from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Se lee el archivo CSV de datos de Melbourne
melbourne_data = pd.read_csv("melb_data.csv")

# Se muestra la lista de columnas en los datos
melbourne_data.columns

# Se eliminan las filas que tienen valores faltantes (NaN)
melbourne_data = melbourne_data.dropna(axis=0)

# Variable dependiente: precios de las propiedades
y = melbourne_data.Price

# Características utilizadas para entrenar el modelo
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# Se divide el conjunto de datos en datos de entrenamiento y de validación
# tanto para las características como para el objetivo
# La división se realiza utilizando un generador de números aleatorios.
# Proporcionar un valor numérico al argumento random_state garantiza
# que obtengamos la misma división cada vez que ejecutamos este script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Se define el modelo de regresión de árbol de decisión
melbourne_model = DecisionTreeRegressor()

# Se ajusta el modelo utilizando los datos de entrenamiento
melbourne_model.fit(train_X, train_y)

# Se obtienen los precios predichos en los datos de validación
val_predictions = melbourne_model.predict(val_X)

# Se imprime el error medio absoluto entre las predicciones y los valores reales
print(mean_absolute_error(val_y, val_predictions))
