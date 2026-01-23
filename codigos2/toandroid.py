import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model("cnn2025.h5")

# Crear un conversor TFLite a partir del modelo de Keras
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Opcional) Ajustar optimizaciones, cuantización, etc.
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertir a TFLite
tflite_model = converter.convert()

# Guardar el modelo .tflite
with open("cnn2025.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo TFLite generado: cnn2025.tflite")
