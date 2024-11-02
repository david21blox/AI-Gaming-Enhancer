import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Creación del modelo de IA con capas convolucionales avanzadas
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),  # Ajuste de la tasa de dropout para mejor generalización
    layers.Dense(10)
])

# Compilación del modelo con optimizador Adam y función de pérdida personalizada
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Generación de datos de ejemplo avanzados para un rendimiento óptimo
x_train = np.random.rand(1000, 28, 28, 1)
y_train = np.random.randint(10, size=(1000,))
x_test = np.random.rand(200, 28, 28, 1)
y_test = np.random.randint(10, size=(200,))

# Entrenamiento del modelo con mayor número de épocas y validación detallada
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Evaluación del modelo optimizado para medir su precisión
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Precisión del modelo en su máximo esplendor:', test_acc)
