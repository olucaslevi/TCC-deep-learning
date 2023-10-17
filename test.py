import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Carregar os dados
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Separar as features e os rótulos
y_train = train['label'].values
x_train = train.drop(labels=['label'], axis=1).values
test = test.values

# Normalizar e remodelar as imagens
def reshape_and_normalize(images):
    images = images / 255.0  # Normalizar os valores de pixel
    images = images.reshape(-1, 28, 28, 1)
    return images

X_train = reshape_and_normalize(x_train)
X_test = reshape_and_normalize(test)

# Dividir os dados de treinamento
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2)

# Definir a rede neural convolucional (CNN)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Salvar o modelo treinado em um arquivo
model.save('mnist_cnn_model.h5')
