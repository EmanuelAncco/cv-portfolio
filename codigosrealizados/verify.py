import tensorflow as tf

print("Versión de TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detectadas:", gpus)

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✔ GPU habilitada correctamente.")
else:
    print("✘ No se detectó GPU. Revisa CUDA/cuDNN.")
