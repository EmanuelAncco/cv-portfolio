#!/usr/bin/env python3
"""
potato_vit_inference_gui.py – Aplicación GUI para Inferencia con Vision Transformer

Permite seleccionar MULTIPLES imágenes, realizar inferencia con el modelo ViT entrenado
y mostrar los resultados en una interfaz gráfica con historial y sugerencias
de solución para el contexto de Perú, incluyendo el porcentaje de confianza.

Requiere:
- Python 3.10+
- TensorFlow 2.10+
- Pillow (pip install Pillow)
- El modelo entrenado guardado en formato SavedModel ('modeloGAIA_best' por defecto)
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # Importar layers para las clases personalizadas
import os
import time # Importar time para pausas (opcional, para visualización)


# ─────────────────────────── Configuración del Modelo ────────────────────────────
# Ruta donde se guardó el modelo en formato SavedModel
# ¡ASEGÚRATE DE QUE ESTA RUTA APUNTE A LA CARPETA 'modeloGAIA_best'!
MODEL_PATH = "modeloGAIA_best"

# Parámetros de preprocesamiento de imagen (deben coincidir con el entrenamiento)
IMG_SIZE = (224, 224)
# Rango de reescalado utilizado durante el entrenamiento (ej: Rescaling(1./255))
RESCALE_RANGE = 1.0 / 255.0

# Nombres de las clases en el orden que el modelo las predice (deben coincidir con el orden alfabético de las carpetas)
# Orden alfabético de las carpetas: ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']
CLASS_NAMES = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight'] # *** CORREGIDO EL ORDEN ***

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


# ─────────────────────────── Carga del Modelo ────────────────────────────
# Para cargar un SavedModel que contiene capas personalizadas,
# necesitamos definir esas clases personalizadas aquí.
# Estas definiciones deben coincidir exactamente con las utilizadas al construir el modelo.

# --- Definiciones de Capas Personalizadas (Copiadas del script de entrenamiento) ---
class Patches(layers.Layer):
    """Divide las imágenes en parches."""
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    """Codifica los parches y añade embeddings posicionales."""
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # Proyectar los parches y añadir el embedding posicional
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
            })
        return config
# --- Fin Definiciones de Capas Personalizadas ---

# Diccionario de objetos personalizados para cargar el modelo
custom_objects = {
    'Patches': Patches,
    'PatchEncoder': PatchEncoder,
    # Si la capa de aumento de datos fue una Sequential nombrada y necesitas cargarla
    # como un objeto personalizado, podrías necesitar algo como:
    # 'data_augmentation': keras.Sequential # O la clase específica si la definiste
}

# Intenta cargar el modelo
model = None # Inicializa el modelo como None
try:
    print(f"Intentando cargar el modelo desde: {MODEL_PATH}")
    # Cargar el modelo SavedModel, pasando los objetos personalizados
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("Modelo cargado exitosamente.")
    # --- Depuración: Imprimir resumen del modelo cargado al inicio ---
    print("\n--- Resumen del modelo cargado al inicio ---")
    model.summary()
    print("------------------------------------------")
    # -----------------------------------------------------------------
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None # Asegúrate de que el modelo sea None si falla la carga
    # Muestra un cuadro de diálogo de error si la GUI se va a iniciar
    if __name__ == "__main__":
         messagebox.showerror("Error de Carga del Modelo",
                              f"No se pudo cargar el modelo desde {MODEL_PATH}.\n"
                              f"Detalle del error: {e}\n"
                              "La aplicación no puede iniciar sin el modelo.")


# ─────────────────────────── Clase de la Aplicación GUI ────────────────────────────

class InferenceApp:
    def __init__(self, root):
        self.root = root
        root.title("Clasificador de Tizón de la Papa con ViT") # Título de la ventana

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
                img_array = img_array * RESCALE_RANGE

                # --- Depuración: Inspeccionar el array de imagen antes de la predicción ---
                print(f"\n--- Procesando archivo: {os.path.basename(file_path)} ---")
                print(f"Forma del array de imagen: {img_array.shape}")
                print(f"Tipo de dato del array de imagen: {img_array.dtype}")
                print(f"Rango de valores del array de imagen: {np.min(img_array)} - {np.max(img_array)}")
                # ---------------------------------------------------------------------

                # --- Realizar la inferencia ---
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
