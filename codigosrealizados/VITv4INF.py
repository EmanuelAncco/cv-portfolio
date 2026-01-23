#!/usr/bin/env python3
"""
potato_gui_predictor.py – Interfaz gráfica para clasificar imágenes de papa.

Permite seleccionar múltiples imágenes y muestra la predicción y confianza
para cada una usando el modelo ViT pre-entrenado.
v3: Añade model.compile() después de cargar pesos para mayor consistencia.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image # Para cargar y preprocesar la imagen

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk # Para widgets mejorados como el Separator y Progressbar
from tkinter import scrolledtext # Para el área de resultados con scroll
from tkinter import messagebox # Para mostrar errores

# --- Configuración del Modelo (Igual que en el script de inferencia) ---
IMG_SIZE         = (224, 224)
PATCH_SIZE       = 16
NUM_PATCHES      = (IMG_SIZE[0] // PATCH_SIZE) ** 2
PROJECTION_DIM   = 64
NUM_HEADS        = 4
TRANSFORMER_LAYERS = 8
MLP_UNITS        = [PROJECTION_DIM * 2, PROJECTION_DIM]
WEIGHT_DECAY_L2  = 1e-4
CLASS_NAMES = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']
NUM_CLASSES = len(CLASS_NAMES)
BEST_WEIGHTS_PATH = "modeloGAIA_vit_optimized_best_weights.h5" # Asegúrate que esté accesible

# --- Variable Global para el Modelo (para no recargarlo) ---
loaded_model = None

# --- Definiciones del Modelo ViT (Exactamente como en entrenamiento/inferencia) ---
# (Se incluyen las clases Patches, PatchEncoder, mlp y create_vit_classifier aquí)

class Patches(layers.Layer):
    """Divide las imágenes en parches."""
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    # ... (resto del código de la clase Patches igual que antes) ...
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

    # ... (resto del código de la clase PatchEncoder igual que antes) ...
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
            })
        return config


def mlp(x, hidden_units, dropout_rate, kernel_regularizer=None):
    """Bloque MLP (Multi-Layer Perceptron)."""
    # ... (código de mlp igual que antes) ...
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu, kernel_regularizer=kernel_regularizer)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_vit_classifier(num_classes, input_shape=IMG_SIZE + (3,),
                          patch_size=PATCH_SIZE, projection_dim=PROJECTION_DIM,
                          num_heads=NUM_HEADS, transformer_layers=TRANSFORMER_LAYERS,
                          mlp_units=MLP_UNITS, include_augmentation=False, # Siempre False para inferencia
                          l2_regularization_factor=WEIGHT_DECAY_L2):
    """Construye el modelo Vision Transformer para inferencia."""
    # ... (código de create_vit_classifier igual que antes) ...
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Crear parches
    patches = Patches(patch_size)(x)
    # Codificar parches y añadir embedding posicional
    encoded_patches = PatchEncoder(NUM_PATCHES, projection_dim)(patches)

    # Crear múltiples capas del Transformer Encoder
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=mlp_units, dropout_rate=0.1,
                 kernel_regularizer=tf.keras.regularizers.l2(l2_regularization_factor))
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=tf.keras.regularizers.l2(l2_regularization_factor))(representation)

    model = keras.Model(inputs=inputs, outputs=outputs, name="vit_classifier_inference")
    return model


# --- Funciones de Inferencia ---

def load_model_lazily():
    """Carga el modelo, los pesos y lo COMPILA si aún no se han cargado."""
    global loaded_model
    if loaded_model is None:
        if not os.path.exists(BEST_WEIGHTS_PATH):
            print(f"Error: No se encontró el archivo de pesos: {BEST_WEIGHTS_PATH}")
            messagebox.showerror("Error", f"No se encontró el archivo de pesos:\n{BEST_WEIGHTS_PATH}")
            return None
        try:
            print("Cargando modelo y pesos...")
            model_instance = create_vit_classifier(
                num_classes=NUM_CLASSES,
                include_augmentation=False,
                l2_regularization_factor=WEIGHT_DECAY_L2
            )
            model_instance.load_weights(BEST_WEIGHTS_PATH)
            print("Pesos cargados. Compilando modelo...")

            # --- ¡NUEVO PASO: Compilar el modelo! ---
            # Usamos un optimizador simple, ya que no se usa para la predicción
            # pero sí es necesario para que compile. Los parámetros deben ser
            # consistentes con los usados en la evaluación del script de entrenamiento.
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # Usar un LR base
            model_instance.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy", # Debe coincidir con el entrenamiento/evaluación
                metrics=["accuracy"] # Debe coincidir con el entrenamiento/evaluación
            )
            print("Modelo compilado exitosamente.")
            loaded_model = model_instance # Asignar a la variable global solo si todo fue exitoso
        except Exception as e:
            print(f"Error al cargar o compilar el modelo: {e}")
            messagebox.showerror("Error", f"Error al cargar o compilar el modelo:\n{e}")
            loaded_model = None # Asegurarse de que sigue siendo None si falla
            return None
    return loaded_model

def load_and_preprocess_image(img_path, target_size):
    """Carga y preprocesa imagen (sin reescalar, interpolación bilinear)."""
    # ... (código igual que antes) ...
    try:
        img = image.load_img(img_path, target_size=target_size, interpolation='bilinear')
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch
    except Exception as e:
        print(f"Error al cargar/preprocesar {os.path.basename(img_path)}: {e}")
        return None


def predict_single_image(model, img_path):
    """Realiza la predicción para una sola imagen."""
    # ... (código igual que antes) ...
    preprocessed_image = load_and_preprocess_image(img_path, IMG_SIZE)
    if preprocessed_image is None:
        return "Error de carga", 0.0

    try:
        # Asegurarse de que la entrada sea float32
        predictions = model.predict(tf.cast(preprocessed_image, tf.float32))
        prediction_scores = predictions[0]
        predicted_index = np.argmax(prediction_scores)
        predicted_class_name = CLASS_NAMES[predicted_index]
        predicted_confidence = prediction_scores[predicted_index]
        return predicted_class_name, predicted_confidence
    except Exception as e:
        print(f"Error durante la predicción para {os.path.basename(img_path)}: {e}")
        return "Error de predicción", 0.0


# --- Lógica de la GUI ---

class App:
    # ... (El __init__ y los métodos select_files, update_file_list, update_results son iguales) ...
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Tizón de Papa")
        self.root.geometry("700x550") # Tamaño inicial de la ventana

        self.selected_files = []
        self.model = None # Se cargará lazily

        # --- Frame principal ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Sección de Selección de Archivos ---
        selection_frame = ttk.LabelFrame(main_frame, text="Selección de Imágenes", padding="10")
        selection_frame.pack(fill=tk.X, pady=(0, 10))

        self.select_button = ttk.Button(selection_frame, text="Seleccionar Imágenes...", command=self.select_files)
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))

        self.run_button = ttk.Button(selection_frame, text="Realizar Predicciones", command=self.run_predictions, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT)

        # --- Lista de Archivos Seleccionados ---
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.X, pady=(0, 10))

        list_label = ttk.Label(list_frame, text="Archivos seleccionados:")
        list_label.pack(anchor=tk.W)

        self.file_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.SINGLE) # Altura inicial
        self.file_listbox.pack(fill=tk.X, expand=True)

        # --- Sección de Resultados ---
        results_frame = ttk.LabelFrame(main_frame, text="Resultados de Predicción", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=15, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # --- Barra de Progreso (opcional) ---
        self.progress_bar = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))


    def select_files(self):
        """Abre el diálogo para seleccionar archivos y actualiza la lista."""
        # Tipos de archivo permitidos
        filetypes = (
            ('Imágenes', '*.jpg *.jpeg *.png *.bmp *.gif'),
            ('Todos los archivos', '*.*')
        )
        # Abrir diálogo para seleccionar MÚLTIPLES archivos
        filenames = filedialog.askopenfilenames(
            title='Selecciona imágenes para clasificar',
            filetypes=filetypes
        )
        if filenames:
            self.selected_files = list(filenames) # Guardar la tupla como lista
            self.update_file_list()
            self.run_button.config(state=tk.NORMAL) # Habilitar botón de predicción
        else:
            # Si no se seleccionan archivos, mantener o deshabilitar el botón
            if not self.selected_files:
                 self.run_button.config(state=tk.DISABLED)


    def update_file_list(self):
        """Actualiza el Listbox con los nombres de archivo seleccionados."""
        self.file_listbox.delete(0, tk.END) # Limpiar lista anterior
        if self.selected_files:
            for filepath in self.selected_files:
                self.file_listbox.insert(tk.END, os.path.basename(filepath))
        else:
            self.file_listbox.insert(tk.END, "(Ningún archivo seleccionado)")


    def run_predictions(self):
        """Carga el modelo (si es necesario) y ejecuta la predicción en los archivos."""
        if not self.selected_files:
            self.update_results("Por favor, selecciona imágenes primero.")
            return

        # Intentar cargar Y COMPILAR el modelo si no está cargado
        self.model = load_model_lazily() # Ahora llama a la función actualizada
        if self.model is None:
            # El error ya se mostró en un messagebox dentro de load_model_lazily
            self.update_results("Error: No se pudo cargar/compilar el modelo. Revisa los logs o mensajes de error.")
            return

        self.update_results("Iniciando predicciones...\n" + "="*30 + "\n")
        self.progress_bar['value'] = 0 # Resetear barra de progreso
        self.progress_bar['maximum'] = len(self.selected_files)
        self.root.update_idletasks() # Actualizar GUI

        # Deshabilitar botones mientras se procesa
        self.select_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.DISABLED)

        results_str = ""
        for i, filepath in enumerate(self.selected_files):
            filename = os.path.basename(filepath)
            # Actualizar mensaje en GUI (opcional, puede ralentizar un poco)
            # self.update_results(f"Procesando: {filename} ({i+1}/{len(self.selected_files)})...", append=True)
            # self.root.update_idletasks() # Actualizar para mostrar mensaje

            pred_class, pred_conf = predict_single_image(self.model, filepath)

            results_str += f"Archivo: {filename}\n"
            results_str += f"  Clase Predicha: {pred_class}\n"
            results_str += f"  Confianza: {pred_conf*100:.2f}%\n" # Mostrar como porcentaje
            results_str += "-"*20 + "\n"

            # Actualizar barra de progreso
            self.progress_bar['value'] = i + 1
            self.root.update_idletasks()

        self.update_results(results_str, append=True) # Añadir todos los resultados al final
        self.update_results("\n" + "="*30 + "\nPredicciones completadas.", append=True)

        # Rehabilitar botones
        self.select_button.config(state=tk.NORMAL)
        self.run_button.config(state=tk.NORMAL)


    def update_results(self, text, append=False):
        """Actualiza el área de texto de resultados."""
        self.results_text.config(state=tk.NORMAL) # Habilitar para escribir
        if not append:
            self.results_text.delete('1.0', tk.END) # Borrar contenido anterior
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.see(tk.END) # Hacer scroll hacia el final
        self.results_text.config(state=tk.DISABLED) # Deshabilitar para evitar edición


# --- Ejecución Principal de la GUI ---
if __name__ == "__main__":
    # Configurar GPU (opcional pero recomendado si está disponible)
    # ... (código igual que antes) ...
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✔ Usando GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(f"⚠️ Error configurando GPU: {e}. Usando CPU.")
    else:
        print("ℹ️ No se detectó GPU, usando CPU.")


    # Crear la ventana principal y la aplicación
    root = tk.Tk()
    app = App(root)
    root.mainloop() # Iniciar el bucle de eventos de la GUI

