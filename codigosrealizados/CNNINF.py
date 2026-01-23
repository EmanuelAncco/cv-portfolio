#!/usr/bin/env python3
"""
potato_cnn_inference_gui.py – Aplicación GUI para Inferencia con CNN

Permite seleccionar MULTIPLES imágenes, realizar inferencia con el modelo CNN entrenado
y mostrar los resultados en una interfaz gráfica con historial y sugerencias
de solución para el contexto de Perú, incluyendo el porcentaje de confianza.

Este script construye la arquitectura del modelo CNN SIN aumento de datos
y carga solo los pesos guardados por el script de entrenamiento CNN.

Requiere:
- Python 3.10+
- TensorFlow 2.10+
- Pillow (pip install Pillow)
- El archivo de pesos entrenado guardado ('modeloGAIA_cnn_best_weights.h5' por defecto)
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16 # Importar VGG16 para reconstruir la arquitectura
import os
import time # Importar time para pausas (opcional, para visualización)


# ─────────────────────────── Configuración del Modelo ────────────────────────────
# Ruta donde se guardaron los pesos del modelo entrenado
# ¡ASEGÚRATE DE QUE ESTA RUTA APUNTE AL ARCHIVO 'modeloGAIA_cnn_best_weights.h5'!
MODEL_WEIGHTS_PATH = "modeloGAIA_cnn_best_weights.h5" # Cargar solo pesos

# Parámetros de preprocesamiento de imagen (deben coincidir con el entrenamiento)
IMG_SIZE = (224, 224)
# Rango de reescalado utilizado durante el entrenamiento (ej: Rescaling(1./255))
RESCALE_RANGE = 1.0 / 255.0

# Nombres de las clases en el orden que el modelo las predice (deben coincidir con el orden alfabético de las carpetas)
# Orden alfabético de las carpetas: ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']
CLASS_NAMES = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']

# Información de soluciones y costos estimados para Perú (esto es una simplificación)
# Los costos son estimados y pueden variar significativamente según la región, proveedor, escala, etc.
SOLUTIONS_INFO = {
    'Potato___Early_blight': {
        'nombre_comun': 'Tizón Temprano',
        'soluciones': [
            "- Aplicación de fungicidas protectores (ej: Clorotalonil, Mancozeb) de forma preventiva.",
            "- Rotación de cultivos para reducir inóculo en el suelo.",
            "- Uso de variedades de papa con mayor tolerancia.",
            "- Manejo adecuado del riego para evitar humedad prolongada en las hojas.",
            "- Eliminación de restos de cultivo infectados."
        ],
        'costo_estimado': "Medio a Alto (dependiendo de la frecuencia y tipo de fungicida)"
    },
    'Potato___Late_blight': {
        'nombre_comun': 'Tizón Tardío',
        'soluciones': [
            "- Aplicación **urgente** y regular de fungicidas sistémicos y de contacto (ej: Metalaxil, Cymoxanil, Propamocarb, Fosetil-Al).",
            "- Monitoreo constante para detección temprana.",
            "- Eliminación y destrucción de plantas y tubérculos infectados.",
            "- Evitar riegos por aspersión, preferir riego por surcos o goteo.",
            "- Uso de variedades resistentes (si están disponibles y adaptadas a la zona)."
        ],
        'costo_estimado': "**Alto** (requiere aplicaciones frecuentes y fungicidas específicos)"
    },
    'Potato___healthy': {
        'nombre_comun': 'Papa Saludable',
        'soluciones': [
            "- Continuar con buenas prácticas agrícolas.",
            "- Monitoreo regular para detectar signos tempranos de enfermedades o plagas.",
            "- Mantener un suelo sano y nutrición adecuada de la planta."
        ],
        'costo_estimado': "Bajo (costos de mantenimiento y monitoreo preventivo)"
    }
}

# Parámetros de la arquitectura CNN (deben coincidir con el entrenamiento)
# No necesitamos todos los parámetros de ViT aquí, solo los relevantes para la arquitectura CNN
# IMG_SIZE y las capas clasificadoras
# NUM_CLASSES lo obtendremos de la lista CLASS_NAMES

# ─────────────────────────── Modelo CNN (para reconstrucción) ────────────────────────────
# Copiar la función create_cnn_classifier del script de entrenamiento, sin include_augmentation

def create_cnn_classifier(num_classes, input_shape=IMG_SIZE + (3,)): # Eliminamos include_augmentation
    """
    Construye la arquitectura del modelo CNN (VGG16 pre-entrenado) sin aumento de datos.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs # Directamente desde los inputs, sin aumento

    # Cargar el modelo base VGG16 pre-entrenado en ImageNet
    # include_top=False elimina la capa clasificadora final de VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congelar las capas del modelo base (no entrenamos en inferencia)
    base_model.trainable = False

    # Conectar el modelo base a la pipeline
    # VGG16 espera imágenes en el rango 0-255, no 0-1.
    # Sin embargo, la capa Rescaling(1./255) en el preprocesamiento de datasets
    # ya escala a 0-1. VGG16 tiene su propio preprocesamiento interno si se usa
    # con include_top=True o si se usa preprocess_input.
    # Para simplicidad y consistencia con el preprocesamiento de datasets,
    # mantendremos la entrada 0-1 y VGG16 debería manejarlo (aunque idealmente
    # usaríamos preprocess_input de VGG16).
    # Si experimentas problemas, considera añadir una capa preprocess_input de VGG16
    # o ajustar el Rescaling inicial. Por ahora, mantenemos la simplicidad.

    x = base_model(x, training=False) # training=False es crucial para inferencia

    # Añadir capas clasificadoras personalizadas (deben coincidir con el entrenamiento)
    x = layers.GlobalAveragePooling2D()(x) # Pooling global para reducir dimensiones
    x = layers.Dropout(0.5)(x) # Dropout (aunque inactivo en training=False)
    # Capa densa con activación ReLU
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x) # Otro dropout
    outputs = layers.Dense(num_classes, activation="softmax")(x) # Capa de salida

    # Crear el modelo completo
    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_classifier")
    return model


# ─────────────────────────── Carga del Modelo (Reconstrucción y Carga de Pesos) ────────────────────────────

model = None # Inicializa el modelo como None
num_classes_inferred = len(CLASS_NAMES) # Obtener número de clases de la lista CLASS_NAMES

try:
    print(f"Intentando construir la arquitectura del modelo CNN y cargar pesos desde: {MODEL_WEIGHTS_PATH}")
    # *** Construir la arquitectura del modelo SIN aumento de datos ***
    # Pasamos include_augmentation=False (aunque la función create_cnn_classifier ya no tiene ese flag)
    model = create_cnn_classifier(num_classes=num_classes_inferred)

    # *** Cargar solo los pesos en la arquitectura construida ***
    # No necesitamos custom_objects para cargar pesos .h5 si las capas son estándar o definidas.
    model.load_weights(MODEL_WEIGHTS_PATH)

    print("Arquitectura CNN construida y pesos cargados exitosamente.")
    # Opcional: Imprimir un resumen del modelo cargado para verificar
    print("\n--- Resumen del modelo cargado para inferencia ---")
    model.summary()
    print("-------------------------------------------------")

except Exception as e:
    print(f"Error al construir la arquitectura CNN o cargar los pesos: {e}")
    model = None # Asegúrate de que el modelo sea None si falla la carga
    # Muestra un cuadro de diálogo de error si la GUI se va a iniciar
    if __name__ == "__main__":
         messagebox.showerror("Error de Carga del Modelo",
                              f"No se pudo construir la arquitectura del modelo CNN o cargar los pesos desde {MODEL_WEIGHTS_PATH}.\n"
                              f"Detalle del error: {e}\n"
                              "La aplicación no puede iniciar sin el modelo.")


# ─────────────────────────── Clase de la Aplicación GUI ────────────────────────────

class InferenceApp:
    def __init__(self, root):
        self.root = root
        root.title("Clasificador de Tizón de la Papa con CNN") # Título de la ventana actualizado

        # Aplica un tema para una mejor apariencia visual
        style = ttk.Style()
        style.theme_use('clam') # Prueba con diferentes temas: 'clam', 'alt', 'default', 'classic'

        # --- Elementos de la GUI ---
        self.main_frame = ttk.Frame(root, padding="15")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Área para mostrar la imagen seleccionada
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Imagen Seleccionada", padding="15") # Texto en español
        self.image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Label para mostrar la imagen. Inicialmente vacío.
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Área para mostrar el resultado de la predicción
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Resultado de Clasificación", padding="15") # Texto en español
        self.result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N), padx=10, pady=10)

        # Label para mostrar el nombre de la clase predicha y la confianza
        # Aumentamos el wraplength para acomodar el porcentaje
        self.prediction_label = ttk.Label(self.result_frame, text="Clasificación: N/A\nConfianza: N/A", font=('Arial', 14, 'bold'), wraplength=350) # Texto en español
        self.prediction_label.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Área para mostrar los métodos de solución y costos
        self.solutions_frame = ttk.LabelFrame(self.main_frame, text="Métodos de Solución y Costos (Perú)", padding="15") # Texto en español
        self.solutions_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10) # Nueva columna

        # Widget de texto con scroll para mostrar las soluciones
        self.solutions_text = scrolledtext.ScrolledText(self.solutions_frame, wrap=tk.WORD, width=40, height=10, font=('Arial', 10))
        self.solutions_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.solutions_text.insert(tk.END, "Seleccione una imagen para ver las sugerencias de solución.")
        self.solutions_text.configure(state='disabled') # Deshabilitar edición

        # Botón para seleccionar imagen
        # Cambiamos el texto para indicar que se pueden seleccionar múltiples archivos
        self.select_button = ttk.Button(self.main_frame, text="Seleccionar Imágenes para Clasificar", command=self.select_images) # Texto en español, plural
        self.select_button.grid(row=1, column=0, columnspan=3, pady=15) # columnspan 3 para centrar en las 3 columnas

        # Tabla para el historial de clasificaciones
        self.history_frame = ttk.LabelFrame(self.main_frame, text="Historial de Clasificaciones", padding="15") # Texto en español
        self.history_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10) # columnspan 3

        # Scrollbar para la tabla
        self.history_scrollbar = ttk.Scrollbar(self.history_frame)
        self.history_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Treeview para mostrar el historial (actúa como tabla)
        # Añadimos una nueva columna para la confianza
        self.history_tree = ttk.Treeview(self.history_frame, columns=("Filename", "Classification", "Confidence"), show="headings", yscrollcommand=self.history_scrollbar.set)
        self.history_tree.heading("Filename", text="Nombre del Archivo") # Texto en español
        self.history_tree.heading("Classification", text="Clasificación Predicha") # Texto en español
        self.history_tree.heading("Confidence", text="Confianza (%)") # Nuevo encabezado en español

        # Ajustar el ancho de las columnas (opcional, se ajustarán automáticamente)
        self.history_tree.column("Filename", width=200, anchor=tk.W)
        self.history_tree.column("Classification", width=200, anchor=tk.W)
        self.history_tree.column("Confidence", width=100, anchor=tk.CENTER) # Columna centrada para el porcentaje

        self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configurar el scrollbar para controlar la tabla
        self.history_scrollbar.config(command=self.history_tree.yview)

        # --- Configurar pesos para redimensionamiento ---
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1) # Columna de imagen
        self.main_frame.columnconfigure(1, weight=1) # Columna de resultado
        self.main_frame.columnconfigure(2, weight=1) # Nueva columna de soluciones
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(0, weight=1)
        self.solutions_frame.columnconfigure(0, weight=1) # Permitir que el contenido de soluciones se expanda
        self.solutions_frame.rowconfigure(0, weight=1)
        self.history_frame.columnconfigure(0, weight=1)
        self.history_frame.rowconfigure(0, weight=1)


    def select_images(self):
        """
        Abre un diálogo para seleccionar MULTIPLES archivos de imagen,
        carga cada imagen, realiza la inferencia y actualiza la GUI.
        """
        # Verifica si el modelo se cargó correctamente
        if model is None:
            messagebox.showerror("Error", "El modelo no está disponible. No se puede realizar la inferencia.")
            return

        # Abre el diálogo de selección de archivos (permite seleccionar múltiples)
        file_paths = filedialog.askopenfilenames( # *** CAMBIO A askopenfilenames ***
            initialdir=".", # Directorio inicial (puede ser "." para el directorio actual)
            title="Seleccionar archivos de imagen", # Texto en español, plural
            filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.gif"), # Tipos de archivo permitidos
                       ("Todos los archivos", "*.*")) # Texto en español
        )

        # Si el usuario cancela el diálogo, no hacemos nada
        if not file_paths:
            return

        # Limpiar el historial y los resultados anteriores antes de procesar un nuevo lote
        # self.history_tree.delete(*self.history_tree.get_children()) # Opcional: limpiar historial
        self.prediction_label.configure(text="Clasificación: Procesando...\nConfianza: N/A")
        self.solutions_text.configure(state='normal')
        self.solutions_text.delete(1.0, tk.END)
        self.solutions_text.insert(tk.END, "Procesando imágenes...")
        self.solutions_text.configure(state='disabled')
        self.root.update() # Actualizar la GUI para mostrar el mensaje de procesamiento

        # Iterar sobre cada archivo seleccionado
        for file_path in file_paths:
            try:
                # --- Cargar y mostrar la imagen actual ---
                img = Image.open(file_path)
                img_display = img.copy()
                img_display.thumbnail((300, 300))
                img_tk = ImageTk.PhotoImage(img_display)
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk # Mantener referencia
                self.root.update_idletasks() # Actualizar la GUI para mostrar la imagen actual

                # --- Preprocesar la imagen para la inferencia ---
                img_inference = img.resize(IMG_SIZE)
                img_array = np.array(img_inference).astype(np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                # Aplicar el mismo reescalado que en el entrenamiento
                img_array = img_array * RESCALE_RANGE

                # --- Depuración: Inspeccionar el array de imagen antes de la predicción ---
                print(f"\n--- Procesando archivo: {os.path.basename(file_path)} ---")
                print(f"Forma del array de imagen: {img_array.shape}")
                print(f"Tipo de dato del array de imagen: {img_array.dtype}")
                print(f"Rango de valores del array de imagen: {np.min(img_array)} - {np.max(img_array)}")
                # ---------------------------------------------------------------------

                # --- Realizar la inferencia ---
                # Asegurarse de que el modelo esté en modo de inferencia (las capas de dropout, etc. inactivas)
                # Al cargar solo pesos en una arquitectura sin aumento, esto ya debería ser el caso,
                # pero predict por defecto también maneja esto.
                predictions = model.predict(img_array, verbose=0) # verbose=0 para no imprimir progreso por cada archivo

                # --- Depuración: Imprimir las predicciones crudas ---
                print(f"Predicciones crudas (probabilidades): {predictions}")
                # ---------------------------------------------------

                # Obtiene el índice de la clase con la mayor probabilidad
                predicted_class_index = np.argmax(predictions, axis=1)[0]
                # Obtiene el nombre de la clase predicha usando el índice
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                # Obtiene la probabilidad (confianza) de la clase predicha
                confidence = predictions[0][predicted_class_index] * 100 # Convertir a porcentaje

                # --- Actualizar la GUI con el resultado actual ---
                self.prediction_label.configure(text=f"Clasificación: {predicted_class_name}\nConfianza: {confidence:.2f}%") # Mostrar 2 decimales
                self.root.update_idletasks() # Actualizar la GUI

                # --- Mostrar métodos de solución y costos para la imagen actual ---
                self.solutions_text.configure(state='normal') # Habilitar edición temporalmente
                self.solutions_text.delete(1.0, tk.END) # Limpiar contenido anterior

                if predicted_class_name in SOLUTIONS_INFO:
                    info = SOLUTIONS_INFO[predicted_class_name]
                    self.solutions_text.insert(tk.END, f"Enfermedad: {info['nombre_comun']}\n\n")
                    self.solutions_text.insert(tk.END, "Métodos de Solución:\n")
                    for sol in info['soluciones']:
                        self.solutions_text.insert(tk.END, f"{sol}\n")
                    self.solutions_text.insert(tk.END, f"\nCosto Estimado en Perú: {info['costo_estimado']}\n")
                    self.solutions_text.insert(tk.END, "\nNota: Estos costos son estimados y pueden variar.")
                else:
                    self.solutions_text.insert(tk.END, "Información de solución no disponible para esta clasificación.")

                self.solutions_text.configure(state='disabled') # Deshabilitar edición nuevamente
                self.root.update_idletasks() # Actualizar la GUI


                # --- Añadir el resultado al historial ---
                filename = os.path.basename(file_path)
                self.history_tree.insert("", tk.END, values=(filename, predicted_class_name, f"{confidence:.2f}%")) # Añadir confianza a la fila
                self.root.update_idletasks() # Actualizar la GUI

                # Opcional: Pausa breve para ver cada imagen procesada
                # time.sleep(0.5)


            except Exception as e:
                # Muestra un cuadro de diálogo de error para el archivo actual
                messagebox.showerror("Error de procesamiento", f"No se pudo procesar el archivo {os.path.basename(file_path)}: {e}")
                # Puedes añadir una fila al historial indicando el error si lo deseas
                self.history_tree.insert("", tk.END, values=(os.path.basename(file_path), "ERROR", "N/A"))
                self.root.update_idletasks()

        # Mensaje final al terminar de procesar todos los archivos
        self.solutions_text.configure(state='normal')
        self.solutions_text.delete(1.0, tk.END)
        self.solutions_text.insert(tk.END, "Procesamiento de lote completado.")
        self.solutions_text.configure(state='disabled')
        self.prediction_label.configure(text="Clasificación: Lote Procesado\nConfianza: N/A")
        self.root.update()


# ─────────────────────────── Ejecución Principal ────────────────────────────

if __name__ == "__main__":
    # Solo inicia la aplicación GUI si el modelo se cargó exitosamente
    if model is not None:
        # Crea la ventana principal de Tkinter
        root = tk.Tk()
        # Crea una instancia de nuestra aplicación
        app = InferenceApp(root)
        # Inicia el bucle principal de eventos de Tkinter
        root.mainloop()
    else:
        # Si el modelo no se cargó, el mensaje de error ya se mostró durante la carga.
        # No es necesario hacer nada más aquí, el script terminará.
        pass
