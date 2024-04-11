# Importa las funciones y clases necesarias
from sklearn.metrics import mean_absolute_error  # Importa la métrica de error absoluto medio
from sklearn.tree import DecisionTreeRegressor  # Importa el modelo de regresión de árbol de decisión
from sklearn.model_selection import train_test_split  # Importa la función para dividir los datos en conjuntos de entrenamiento y validación

# Define una función que calcula el error absoluto medio (MAE) para un determinado número máximo de nodos hoja
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    # Crea un modelo de regresión de árbol de decisión con el número máximo de nodos hoja especificado
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    # Entrena el modelo con los datos de entrenamiento
    model.fit(train_X, train_y)
    # Realiza predicciones en los datos de validación
    preds_val = model.predict(val_X)
    # Calcula el error absoluto medio entre las predicciones y los valores reales en los datos de validación
    mae = mean_absolute_error(val_y, preds_val)
    # Devuelve el error absoluto medio
    return(mae)

# Código para cargar datos que se ejecuta en este punto

# Importa la biblioteca pandas
import pandas as pd

# Carga los datos desde el archivo 'melb_data.csv'
melbourne_data = pd.read_csv('melb_data.csv')

# Filtra las filas con valores faltantes
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Elige la variable objetivo y las características
y = filtered_melbourne_data.Price  # Variable objetivo: Precio de la propiedad
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                      'YearBuilt', 'Lattitude', 'Longtitude']  # Características seleccionadas
X = filtered_melbourne_data[melbourne_features]  # Características del modelo

# Divide los datos en conjuntos de entrenamiento y validación
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Compara el error absoluto medio (MAE) con diferentes valores de max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    # Calcula el MAE para el número actual de nodos hoja
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    # Imprime el número de nodos hoja y el MAE correspondiente
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
