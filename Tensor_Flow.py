import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
print("Reconocimiento de números")
#Cargamos dataset
(x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = tf.keras.datasets.mnist.load_data()
#Imprimir el dato 8 de entrenamiento
print("Impresion dato de prueba")
plt.imshow(x_entrenamiento[8], cmap=plt.cm.binary)
print("Etiqueta", y_entrenamiento[8])
#Dimensiones de nuestro tensor
print("Dimensiones del tensor de entrenamiento" ,x_entrenamiento.ndim)
#Contenido del tensor
print("Contenido del tensor" ,x_entrenamiento.shape)
x_entrenamiento=x_entrenamiento.astype("float32")
y_entrenamiento=y_entrenamiento.astype("float32")
x_entrenamiento /=255
y_entrenamiento /=255
#Convertir matriz en un solo vector
x_entrenamiento =x_entrenamiento.reshape(60000, 784)
x_prueba = x_prueba.reshape(10000, 784)
y_entrenamiento = to_categorical(y_entrenamiento, num_classes=10)
y_prueba = to_categorical(y_prueba, num_classes=10)
modelo = tf.keras.Sequential()
#Crear la primera red neuronal con activacion "sigmoide"
modelo.add(tf.keras.layers.Dense(10, activation="sigmoid", input_shape=(784,)))
#Segunda capa con activacion softmax
modelo.add(tf.keras.layers.Dense(10, activation="softmax"))
print("Modelo propuesto")
print(modelo.summary())
#Especificar modelo  Funcion de coste /Optimizador /Metrica
modelo.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
#Entrenamiento del modelo para 5 epocas
modelo.fit(x_entrenamiento, y_entrenamiento, epochs=5)
#Probar el modelo con datos de prueba
prueba_perdida, prueba_precision = modelo.evaluate(x_prueba, y_prueba)
#Imprimir resultado
print("Precision", prueba_precision)
#Predecir el numero de la imagen 11
prediccion = modelo.predict(x_prueba)
np.argmax(prediccion[11])
print("Prediccion: ", prediccion[11])
