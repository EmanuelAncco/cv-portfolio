#!/usr/bin/env python3
"""
potato_gui_predictor_resnet.py – Interfaz gráfica para clasificar imágenes de papa.

Permite seleccionar múltiples imágenes y muestra la predicción y confianza
para cada una usando el modelo ResNet50V2 pre-entrenado y afinado.
v3: Corrige el orden de CLASS_NAMES para coincidir con la predicción observada.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image # Para cargar y preprocesar la imagen
from tensorflow.keras.applications import ResNet50V2 # Importar ResNet50V2
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense # Capas necesarias

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox

# --- Configuración del Modelo (Alineada con el entrenamiento ResNet) ---
IMG_SIZE         = (224, 224)

# --- ¡CORRECCIÓN! ---
# Clases ajustadas para reflejar la inversión observada en la predicción.
# El modelo parece haber aprendido: Índice 1 -> Late Blight, Índice 2 -> Healthy
# Orden original alfabético era: ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'] # <-- ¡CAMBIO AQUÍ!
NUM_CLASSES = len(CLASS_NAMES)

# Ruta al archivo de pesos del modelo ResNet entrenado
BEST_WEIGHTS_PATH = "resnet_transfer_best_weights.h5"

# --- Variable Global para el Modelo ---
loaded_model = None

# --- Definición del Modelo ResNet50V2 Transfer Learning ---
def create_transfer_model(num_classes, input_shape=IMG_SIZE + (3,),
                          include_augmentation=False): # False para inferencia/evaluación
    """
    Construye el modelo usando ResNet50V2 pre-entrenado.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.Rescaling(1./255)(x) # Capa de reescalado incluida en el modelo
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = False # Congelar base
    x = base_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet50V2_Transfer_Inference")
    return model

# --- Funciones de Inferencia ---

def load_model_lazily():
    """Carga el modelo ResNet, los pesos y lo COMPILA si aún no se han cargado."""
    global loaded_model
    if loaded_model is None:
        if not os.path.exists(BEST_WEIGHTS_PATH):
            print(f"Error: No se encontró el archivo de pesos: {BEST_WEIGHTS_PATH}")
            messagebox.showerror("Error", f"No se encontró el archivo de pesos:\n{BEST_WEIGHTS_PATH}")
            return None
        try:
            print("Cargando modelo ResNet y pesos...")
            model_instance = create_transfer_model(
                num_classes=NUM_CLASSES,
                include_augmentation=False
            )
            model_instance.load_weights(BEST_WEIGHTS_PATH)
            print("Pesos cargados. Compilando modelo...")
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            model_instance.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            print("Modelo ResNet compilado exitosamente.")
            loaded_model = model_instance
        except Exception as e:
            print(f"Error al cargar o compilar el modelo ResNet: {e}")
            messagebox.showerror("Error", f"Error al cargar o compilar el modelo ResNet:\n{e}")
            loaded_model = None
            return None
    return loaded_model

def load_and_preprocess_image(img_path, target_size):
    """
    Carga y preprocesa imagen. SIN REESCALADO MANUAL. Usa interpolación 'bilinear'.
    """
    try:
        img = image.load_img(img_path, target_size=target_size, interpolation='bilinear')
        img_array = image.img_to_array(img) # Rango [0, 255]
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch
    except Exception as e:
        print(f"Error al cargar/preprocesar {os.path.basename(img_path)}: {e}")
        return None

def predict_single_image(model, img_path):
    """Realiza la predicción y devuelve clase (corregida), confianza y puntuaciones crudas."""
    preprocessed_image = load_and_preprocess_image(img_path, IMG_SIZE)
    default_scores = np.array([0.0] * NUM_CLASSES)

    if preprocessed_image is None:
        return "Error de carga", 0.0, default_scores

    try:
        predictions = model.predict(tf.cast(preprocessed_image, tf.float32))
        prediction_scores = predictions[0]
        if len(prediction_scores) != NUM_CLASSES:
             print(f"Advertencia: El número de puntuaciones ({len(prediction_scores)}) no coincide con NUM_CLASSES ({NUM_CLASSES}).")
             return "Error de dimensión", 0.0, prediction_scores

        predicted_index = np.argmax(prediction_scores)
        if predicted_index < 0 or predicted_index >= len(CLASS_NAMES):
            print(f"Error: Índice predicho ({predicted_index}) fuera de rango para CLASS_NAMES.")
            return "Error de índice", 0.0, prediction_scores

        # Usar la lista CLASS_NAMES CORREGIDA para la interpretación
        predicted_class_name = CLASS_NAMES[predicted_index]
        predicted_confidence = prediction_scores[predicted_index]

        return predicted_class_name, predicted_confidence, prediction_scores
    except Exception as e:
        print(f"Error durante la predicción para {os.path.basename(img_path)}: {e}")
        return "Error de predicción", 0.0, default_scores

# --- Lógica de la GUI ---

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Tizón de Papa (ResNet50V2 - Corregido)") # Título actualizado
        self.root.geometry("700x600")

        self.selected_files = []
        self.model = None

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        selection_frame = ttk.LabelFrame(main_frame, text="Selección de Imágenes", padding="10")
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        self.select_button = ttk.Button(selection_frame, text="Seleccionar Imágenes...", command=self.select_files)
        self.select_button.pack(side=tk.LEFT, padx=(0, 10))
        self.run_button = ttk.Button(selection_frame, text="Realizar Predicciones", command=self.run_predictions, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT)

        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.X, pady=(0, 10))
        list_label = ttk.Label(list_frame, text="Archivos seleccionados:")
        list_label.pack(anchor=tk.W)
        self.file_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.SINGLE)
        self.file_listbox.pack(fill=tk.X, expand=True)

        results_frame = ttk.LabelFrame(main_frame, text="Resultados de Predicción", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=18, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        self.progress_bar = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))


    def select_files(self):
        filetypes = (
            ('Imágenes', '*.jpg *.jpeg *.png *.bmp *.gif'),
            ('Todos los archivos', '*.*')
        )
        filenames = filedialog.askopenfilenames(
            title='Selecciona imágenes para clasificar',
            filetypes=filetypes
        )
        if filenames:
            self.selected_files = list(filenames)
            self.update_file_list()
            self.run_button.config(state=tk.NORMAL)
        else:
            if not self.selected_files:
                 self.run_button.config(state=tk.DISABLED)

    def update_file_list(self):
        self.file_listbox.delete(0, tk.END)
        if self.selected_files:
            for filepath in self.selected_files:
                self.file_listbox.insert(tk.END, os.path.basename(filepath))
        else:
            self.file_listbox.insert(tk.END, "(Ningún archivo seleccionado)")


    def run_predictions(self):
        """Carga modelo y ejecuta predicción, mostrando puntuaciones y clase corregida."""
        if not self.selected_files:
            self.update_results("Por favor, selecciona imágenes primero.")
            return

        self.model = load_model_lazily()
        if self.model is None:
            self.update_results("Error: No se pudo cargar/compilar el modelo ResNet. Revisa los logs o mensajes de error.")
            return

        self.update_results("Iniciando predicciones (ResNet50V2 - Corregido)...\n" + "="*40 + "\n")
        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = len(self.selected_files)
        self.root.update_idletasks()

        self.select_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.DISABLED)

        results_str = ""
        for i, filepath in enumerate(self.selected_files):
            filename = os.path.basename(filepath)
            pred_class, pred_conf, raw_scores = predict_single_image(self.model, filepath)

            # Formatear puntuaciones con la NUEVA interpretación de CLASS_NAMES
            # CLASS_NAMES = ['Early', 'Late', 'Healthy'] (Indices 0, 1, 2)
            scores_str = f"[{raw_scores[0]:.4f}, {raw_scores[1]:.4f}, {raw_scores[2]:.4f}]"

            results_str += f"Archivo: {filename}\n"
            # Añadir comentario sobre el orden de las puntuaciones para claridad
            results_str += f"  Puntuaciones (E, L, H): {scores_str}\n" # <-- Orden corregido E, L, H
            results_str += f"  Clase Predicha: {pred_class}\n"
            results_str += f"  Confianza: {pred_conf*100:.2f}%\n"
            results_str += "-"*25 + "\n"

            self.progress_bar['value'] = i + 1
            self.root.update_idletasks()

        self.update_results(results_str, append=True)
        self.update_results("\n" + "="*40 + "\nPredicciones completadas.", append=True)

        self.select_button.config(state=tk.NORMAL)
        self.run_button.config(state=tk.NORMAL)

    def update_results(self, text, append=False):
        self.results_text.config(state=tk.NORMAL)
        if not append:
            self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)

# --- Ejecución Principal de la GUI ---
if __name__ == "__main__":
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

    root = tk.Tk()
    app = App(root)
    root.mainloop()
