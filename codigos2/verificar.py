import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices('GPU'))
print("¿Construido con soporte CUDA?:", tf.test.is_built_with_cuda())
print("¿GPU disponible (método antiguo)?:", tf.test.is_gpu_available())
