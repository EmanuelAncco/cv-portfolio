#!/usr/bin/env python3
"""
potato_gui_predictor_resnet_enhanced.py – Interfaz gráfica mejorada.

Muestra la imagen analizada y posibles soluciones para la clasificación
de enfermedades de la papa usando el modelo ResNet50V2.
v4: Layout mejorado, visualización de imagen, recomendaciones.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense

import tkinter as tk
from tkinter import filedialog, font
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
from PIL import Image, ImageTk # Necesario para mostrar imágenes

# --- Configuración del Modelo ---
IMG_SIZE         = (224, 224)
# Orden corregido según lo aprendido por el modelo
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
NUM_CLASSES = len(CLASS_NAMES)
BEST_WEIGHTS_PATH = "resnet_transfer_best_weights.h5"

# --- Variable Global para el Modelo ---
loaded_model = None

# --- SOLUCIONES / RECOMENDACIONES (Ejemplos) ---
# Puedes expandir y detallar estas recomendaciones
SOLUTIONS = {
    "Potato___Early_blight": (
        "Tizón Temprano (Alternaria solani):\n"
        "---------------------------------\n"
        "- Rotación de cultivos (evitar plantar papas o tomates en el mismo lugar por 2-3 años).\n"
        "- Usar variedades resistentes si están disponibles.\n"
        "- Eliminar y destruir restos de plantas infectadas al final de la temporada.\n"
        "- Aplicar fungicidas preventivos (ej. mancozeb, clorotalonil) siguiendo las recomendaciones locales, especialmente en condiciones húmedas.\n"
        "- Asegurar buena ventilación y espaciado entre plantas.\n"
        "- Controlar la humedad del suelo, evitar riego por aspersión si es posible."
    ),
    "Potato___Late_blight": (
        "Tizón Tardío (Phytophthora infestans):\n"
        "----------------------------------\n"
        "- ¡Enfermedad muy destructiva! Actuar rápido.\n"
        "- Usar tubérculos certificados y libres de enfermedad.\n"
        "- Eliminar plantas voluntarias de papa y tomate.\n"
        "- Monitoreo constante, especialmente en clima fresco y húmedo.\n"
        "- Aplicación de fungicidas específicos (ej. metalaxyl, cimoxanil, propamocarb, mancozeb) de forma preventiva y siguiendo un calendario estricto según las condiciones climáticas y las alertas locales.\n"
        "- Destruir plantas severamente infectadas para reducir la fuente de inóculo.\n"
        "- Eliminar restos de cosecha."
    ),
    "Potato___healthy": (
        "Hoja Sana:\n"
        "----------\n"
        "- ¡Excelente! La planta parece no tener signos de Tizón Temprano o Tardío.\n"
        "- Continuar con buenas prácticas agrícolas:\n"
        "  * Monitoreo regular.\n"
        "  * Nutrición balanceada.\n"
        "  * Manejo adecuado del riego.\n"
        "  * Medidas preventivas si las condiciones climáticas son favorables para enfermedades."
    )
}

# --- Definición del Modelo ResNet50V2 (igual que antes) ---
def create_transfer_model(num_classes, input_shape=IMG_SIZE + (3,),
                          include_augmentation=False):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.Rescaling(1./255)(x)
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet50V2_Transfer_Inference")
    return model

# --- Funciones de Inferencia (igual que antes) ---
def load_model_lazily():
    global loaded_model
    if loaded_model is None:
        if not os.path.exists(BEST_WEIGHTS_PATH):
            messagebox.showerror("Error", f"No se encontró el archivo de pesos:\n{BEST_WEIGHTS_PATH}")
            return None
        try:
            print("Cargando modelo ResNet y pesos...")
            model_instance = create_transfer_model(num_classes=NUM_CLASSES, include_augmentation=False)
            model_instance.load_weights(BEST_WEIGHTS_PATH)
            print("Pesos cargados. Compilando modelo...")
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            model_instance.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
            print("Modelo ResNet compilado exitosamente.")
            loaded_model = model_instance
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar o compilar el modelo ResNet:\n{e}")
            loaded_model = None
            return None
    return loaded_model

def load_and_preprocess_image(img_path, target_size):
    try:
        img = image.load_img(img_path, target_size=target_size, interpolation='bilinear')
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch
    except Exception as e:
        print(f"Error al cargar/preprocesar {os.path.basename(img_path)}: {e}")
        return None

def predict_single_image(model, img_path):
    preprocessed_image = load_and_preprocess_image(img_path, IMG_SIZE)
    default_scores = np.array([0.0] * NUM_CLASSES)
    if preprocessed_image is None:
        return "Error de carga", 0.0, default_scores
    try:
        predictions = model.predict(tf.cast(preprocessed_image, tf.float32))
        prediction_scores = predictions[0]
        if len(prediction_scores) != NUM_CLASSES:
             return "Error de dimensión", 0.0, prediction_scores
        predicted_index = np.argmax(prediction_scores)
        if predicted_index < 0 or predicted_index >= len(CLASS_NAMES):
            return "Error de índice", 0.0, prediction_scores
        predicted_class_name = CLASS_NAMES[predicted_index]
        predicted_confidence = prediction_scores[predicted_index]
        return predicted_class_name, predicted_confidence, prediction_scores
    except Exception as e:
        return "Error de predicción", 0.0, default_scores

# --- Lógica de la GUI Mejorada ---

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Enfermedades de Papa v2.0")
        self.root.geometry("950x650") # Ventana más grande
        self.root.configure(bg='#eafaf1') # Color de fondo general

        # Estilo ttk
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam') # Probar temas: 'clam', 'alt', 'default', 'classic'

        # Configurar colores y fuentes
        self.style.configure("TFrame", background='#eafaf1')
        self.style.configure("TLabel", background='#eafaf1', font=('Helvetica', 10))
        self.style.configure("TButton", padding=6, relief="flat", font=('Helvetica', 10, 'bold'), background="#2ecc71", foreground="white")
        self.style.map("TButton", background=[('active', '#27ae60')])
        self.style.configure("TLabelFrame", background='#eafaf1', borderwidth=1, relief="groove")
        self.style.configure("TLabelFrame.Label", background='#eafaf1', font=('Helvetica', 11, 'bold'), foreground="#34495e")
        self.style.configure("TProgressbar", thickness=15, troughcolor='#bdc3c7', background='#2ecc71')

        self.selected_files = []
        self.current_image_tk = None # Para mantener referencia a la imagen
        self.model = None

        # --- Frame Principal ---
        main_frame = ttk.Frame(root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configurar columnas para el layout
        main_frame.columnconfigure(0, weight=1, minsize=200) # Columna izquierda (Selección)
        main_frame.columnconfigure(1, weight=2, minsize=350) # Columna central (Imagen)
        main_frame.columnconfigure(2, weight=2, minsize=350) # Columna derecha (Resultados/Soluciones)
        main_frame.rowconfigure(0, weight=1) # Fila principal se expande

        # --- Columna Izquierda: Selección ---
        left_frame = ttk.LabelFrame(main_frame, text="1. Selección", padding="10")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)
        left_frame.rowconfigure(1, weight=1) # Hacer que el listbox se expanda

        self.select_button = ttk.Button(left_frame, text="Seleccionar Imágenes...", command=self.select_files)
        self.select_button.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 10))

        self.file_listbox = tk.Listbox(left_frame, height=15, selectmode=tk.SINGLE, bg="white", fg="#2c3e50", font=('Courier', 9))
        self.file_listbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        # Scrollbar para listbox
        list_scrollbar = ttk.Scrollbar(left_frame, orient='vertical', command=self.file_listbox.yview)
        list_scrollbar.grid(row=1, column=1, sticky='ns')
        self.file_listbox['yscrollcommand'] = list_scrollbar.set

        # --- Columna Central: Imagen y Control ---
        center_frame = ttk.LabelFrame(main_frame, text="2. Imagen y Predicción", padding="10")
        center_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        center_frame.rowconfigure(0, weight=1) # Imagen se expande
        center_frame.columnconfigure(0, weight=1)

        # Placeholder para la imagen
        self.image_label = tk.Label(center_frame, bg="#ecf0f1", text="La imagen aparecerá aquí", font=('Helvetica', 10), relief="groove")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.img_display_size = (350, 350) # Tamaño deseado para mostrar la imagen

        self.run_button = ttk.Button(center_frame, text="Realizar Predicciones", command=self.run_predictions, state=tk.DISABLED)
        self.run_button.grid(row=1, column=0, sticky="ew", padx=5, pady=(10, 5))

        self.progress_bar = ttk.Progressbar(center_frame, orient='horizontal', mode='determinate')
        self.progress_bar.grid(row=2, column=0, sticky="ew", padx=5, pady=(5, 5))

        # --- Columna Derecha: Resultados y Soluciones ---
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 0), pady=5)
        right_frame.rowconfigure(0, weight=1) # Resultados
        right_frame.rowconfigure(1, weight=1) # Soluciones
        right_frame.columnconfigure(0, weight=1)

        results_frame = ttk.LabelFrame(right_frame, text="3. Resultados", padding="10")
        results_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=10, state=tk.DISABLED, font=('Consolas', 9), bg="#fdfefe", fg="#34495e")
        self.results_text.grid(row=0, column=0, sticky="nsew")

        solutions_frame = ttk.LabelFrame(right_frame, text="4. Recomendaciones", padding="10")
        solutions_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        solutions_frame.rowconfigure(0, weight=1)
        solutions_frame.columnconfigure(0, weight=1)
        self.solutions_text = scrolledtext.ScrolledText(solutions_frame, wrap=tk.WORD, height=10, state=tk.DISABLED, font=('Helvetica', 9), bg="#fdfefe", fg="#34495e")
        self.solutions_text.grid(row=0, column=0, sticky="nsew")

    def select_files(self):
        filetypes = (('Imágenes', '*.jpg *.jpeg *.png *.bmp *.gif'), ('Todos los archivos', '*.*'))
        filenames = filedialog.askopenfilenames(title='Selecciona imágenes', filetypes=filetypes)
        if filenames:
            self.selected_files = list(filenames)
            self.update_file_list()
            self.run_button.config(state=tk.NORMAL)
            self.clear_displays() # Limpiar imagen y textos anteriores
            if self.selected_files: # Mostrar la primera imagen seleccionada
                 self.display_image(self.selected_files[0])
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

    def display_image(self, filepath):
        """Muestra la imagen en el Label central."""
        try:
            img = Image.open(filepath)
            img.thumbnail(self.img_display_size) # Redimensionar manteniendo aspecto
            self.current_image_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.current_image_tk, text="") # Mostrar imagen, quitar texto
        except Exception as e:
            print(f"Error al mostrar imagen {filepath}: {e}")
            self.image_label.config(image=None, text=f"Error al cargar\n{os.path.basename(filepath)}")
            self.current_image_tk = None # Limpiar referencia

    def display_solution(self, predicted_class):
        """Muestra las recomendaciones basadas en la clase predicha."""
        solution_text = SOLUTIONS.get(predicted_class, "No hay recomendaciones específicas para esta predicción.")
        self.update_solutions_text(solution_text)

    def clear_displays(self):
         """Limpia la imagen, resultados y soluciones."""
         self.image_label.config(image=None, text="Selecciona imágenes...")
         self.current_image_tk = None
         self.update_results("", append=False)
         self.update_solutions_text("", append=False)


    def run_predictions(self):
        if not self.selected_files:
            messagebox.showwarning("Sin Selección", "Por favor, selecciona imágenes primero.")
            return

        self.model = load_model_lazily()
        if self.model is None:
            self.update_results("Error: No se pudo cargar/compilar el modelo ResNet.")
            return

        # Limpiar áreas de texto antes de empezar
        self.update_results("Iniciando predicciones (ResNet50V2)...\n" + "="*40 + "\n", append=False)
        self.update_solutions_text("", append=False) # Limpiar soluciones

        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = len(self.selected_files)
        self.root.update_idletasks()

        self.select_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.DISABLED)

        results_buffer = "" # Acumular resultados para actualizar menos veces

        for i, filepath in enumerate(self.selected_files):
            filename = os.path.basename(filepath)
            print(f"Procesando: {filename}") # Log en consola

            # Mostrar imagen actual
            self.display_image(filepath)
            self.root.update_idletasks() # Asegurar que la imagen se muestre

            # Realizar predicción
            pred_class, pred_conf, raw_scores = predict_single_image(self.model, filepath)

            # Formatear y acumular resultados
            scores_str = f"[{raw_scores[0]:.4f}, {raw_scores[1]:.4f}, {raw_scores[2]:.4f}]"
            results_buffer += f"Archivo: {filename}\n"
            results_buffer += f"  Puntuaciones (E, L, H): {scores_str}\n"
            results_buffer += f"  Clase Predicha: {pred_class}\n"
            results_buffer += f"  Confianza: {pred_conf*100:.2f}%\n"
            results_buffer += "-"*25 + "\n"

            # Actualizar resultados y soluciones en la GUI (quizás no en cada iteración si es lento)
            self.update_results(results_buffer, append=False) # Sobrescribir con acumulado
            self.display_solution(pred_class) # Mostrar solución para la imagen actual

            # Actualizar progreso
            self.progress_bar['value'] = i + 1
            self.root.update_idletasks() # Actualizar GUI

        # Mensaje final
        final_results_text = self.results_text.get("1.0", tk.END) # Obtener texto actual
        self.update_results(final_results_text + "\n" + "="*40 + "\nPredicciones completadas.", append=False)

        self.select_button.config(state=tk.NORMAL)
        self.run_button.config(state=tk.NORMAL)

    def update_results(self, text, append=False):
        self.results_text.config(state=tk.NORMAL)
        if not append:
            self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, text) # No añadir newline extra aquí
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)

    def update_solutions_text(self, text):
        """Actualiza el área de texto de soluciones."""
        self.solutions_text.config(state=tk.NORMAL)
        self.solutions_text.delete('1.0', tk.END)
        self.solutions_text.insert(tk.END, text)
        self.solutions_text.config(state=tk.DISABLED)

# --- Ejecución Principal ---
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
