#!/usr/bin/env python3
"""
potato_gui_predictor_resnet_category_grids.py – Interfaz gráfica avanzada.

Muestra miniaturas numeradas globalmente por categoría, historial numerado,
recomendaciones activadas por botones y botón para detener el proceso.
Usa el modelo ResNet50V2.
v11: Numeración global en grids, botón Detener.
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
from PIL import Image, ImageTk

# --- Configuración del Modelo ---
IMG_SIZE         = (224, 224)
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
NUM_CLASSES = len(CLASS_NAMES)
BEST_WEIGHTS_PATH = "resnet_transfer_best_weights.h5"

# --- Variable Global para el Modelo ---
loaded_model = None

# --- SOLUCIONES / RECOMENDACIONES ---
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

# --- Definición del Modelo ResNet50V2 ---
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

# --- Funciones de Inferencia ---
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
        return "Error de carga", 0.0, default_scores, None
    try:
        img_for_thumb = Image.open(img_path)
        predictions = model.predict(tf.cast(preprocessed_image, tf.float32))
        prediction_scores = predictions[0]
        if len(prediction_scores) != NUM_CLASSES:
             return "Error de dimensión", 0.0, prediction_scores, None
        predicted_index = np.argmax(prediction_scores)
        if predicted_index < 0 or predicted_index >= len(CLASS_NAMES):
            return "Error de índice", 0.0, prediction_scores, None
        predicted_class_name = CLASS_NAMES[predicted_index]
        predicted_confidence = prediction_scores[predicted_index]
        return predicted_class_name, predicted_confidence, prediction_scores, img_for_thumb
    except Exception as e:
        print(f"Error durante la predicción para {os.path.basename(img_path)}: {e}")
        return "Error de predicción", 0.0, default_scores, None

# --- Lógica de la GUI con Grids por Categoría, Numeración y Botón Detener ---

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Enfermedades de Papa v6.0")
        self.root.geometry("1300x800")
        self.BG_COLOR = '#f0f4f7'
        self.root.configure(bg=self.BG_COLOR)

        # Estilo ttk
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')
        self.FG_COLOR = '#2c3e50'
        BTN_COLOR = '#3498db'
        BTN_ACTIVE_COLOR = '#2980b9'
        ACCENT_COLOR = '#1abc9c'
        BTN_REC_COLOR = '#5dade2'
        BTN_REC_ACTIVE_COLOR = '#4a90e2'
        BTN_STOP_COLOR = '#e74c3c' # Rojo para detener
        BTN_STOP_ACTIVE_COLOR = '#c0392b'

        self.style.configure("TFrame", background=self.BG_COLOR)
        self.style.configure("TLabel", background=self.BG_COLOR, foreground=self.FG_COLOR, font=('Segoe UI', 10))
        # Estilo Botón Principal
        self.style.configure("TButton", padding=8, relief="flat", font=('Segoe UI', 10, 'bold'), background=BTN_COLOR, foreground="white")
        self.style.map("TButton", background=[('active', BTN_ACTIVE_COLOR)])
        # Estilo Botones Recomendación
        self.style.configure("Rec.TButton", padding=6, relief="flat", font=('Segoe UI', 9, 'bold'), background=BTN_REC_COLOR, foreground="white")
        self.style.map("Rec.TButton", background=[('active', BTN_REC_ACTIVE_COLOR)])
        # Estilo Botón Detener
        self.style.configure("Stop.TButton", padding=8, relief="flat", font=('Segoe UI', 10, 'bold'), background=BTN_STOP_COLOR, foreground="white")
        self.style.map("Stop.TButton", background=[('active', BTN_STOP_ACTIVE_COLOR)])

        self.style.configure("TLabelFrame", background=self.BG_COLOR, borderwidth=1, relief="groove")
        self.style.configure("TLabelFrame.Label", background=self.BG_COLOR, font=('Segoe UI', 11, 'bold'), foreground=self.FG_COLOR)
        self.style.configure("Treeview.Heading", font=('Segoe UI', 10, 'bold'), background=BTN_COLOR, foreground="white", relief="flat")
        self.style.map("Treeview.Heading", background=[('active', BTN_ACTIVE_COLOR)])
        self.style.configure("Treeview", rowheight=25, fieldbackground="#ecf0f1", background="#ecf0f1", foreground=self.FG_COLOR)
        self.style.map("Treeview", background=[('selected', ACCENT_COLOR)], foreground=[('selected', 'white')])
        self.style.configure("Vertical.TScrollbar", background=BTN_COLOR, troughcolor=self.BG_COLOR, bordercolor=self.BG_COLOR, arrowcolor='white')
        self.style.configure("TProgressbar", thickness=15, troughcolor='#bdc3c7', background=ACCENT_COLOR)

        self.selected_files = []
        self.image_references = {CLASS_NAMES[0]: [], CLASS_NAMES[1]: [], CLASS_NAMES[2]: []}
        self.model = None
        self.thumb_size = (100, 100)
        self.images_per_row = 4
        self.processed_counter = 0
        self.stop_processing = False # Flag para detener el proceso

        # --- Frame Principal ---
        main_frame = ttk.Frame(root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configurar columnas
        main_frame.columnconfigure(0, weight=1, minsize=180) # Selección
        main_frame.columnconfigure(1, weight=3, minsize=550) # Grids Categorías
        main_frame.columnconfigure(2, weight=2, minsize=400) # Recomendaciones + Historial
        main_frame.rowconfigure(0, weight=1)

        # --- Columna 0: Selección ---
        left_frame = ttk.LabelFrame(main_frame, text="1. Selección", padding="10")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)
        left_frame.rowconfigure(1, weight=1)

        self.select_button = ttk.Button(left_frame, text="Seleccionar Imágenes...", command=self.select_files)
        self.select_button.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 10))

        self.file_listbox = tk.Listbox(left_frame, height=15, selectmode=tk.SINGLE, bg="white", fg=self.FG_COLOR, font=('Courier', 9), relief="flat", borderwidth=1)
        self.file_listbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        list_scrollbar = ttk.Scrollbar(left_frame, orient='vertical', command=self.file_listbox.yview)
        list_scrollbar.grid(row=1, column=1, sticky='ns')
        self.file_listbox['yscrollcommand'] = list_scrollbar.set

        # Frame para botones de acción
        action_button_frame = ttk.Frame(left_frame)
        action_button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 5))
        action_button_frame.columnconfigure(0, weight=1)
        action_button_frame.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(action_button_frame, text="Realizar Predicciones", command=self.run_predictions, state=tk.DISABLED)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))

        self.stop_button = ttk.Button(action_button_frame, text="Detener", command=self.request_stop, state=tk.DISABLED, style="Stop.TButton")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(2, 0))

        self.progress_bar = ttk.Progressbar(left_frame, orient='horizontal', mode='determinate')
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 5))

        # --- Columna 1: Grids por Categoría ---
        center_frame = ttk.Frame(main_frame)
        center_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        center_frame.rowconfigure(0, weight=1); center_frame.rowconfigure(1, weight=1); center_frame.rowconfigure(2, weight=1)
        center_frame.columnconfigure(0, weight=1)

        self.category_canvases = {}
        self.category_inner_frames = {}
        category_titles = {
            CLASS_NAMES[0]: "Tizón Temprano (Predicho)",
            CLASS_NAMES[1]: "Tizón Tardío (Predicho)",
            CLASS_NAMES[2]: "Sana (Predicho)"
        }

        for i, class_name in enumerate(CLASS_NAMES):
            cat_frame = ttk.LabelFrame(center_frame, text=category_titles[class_name], padding="5")
            cat_frame.grid(row=i, column=0, sticky="nsew", pady=(0, 5))
            cat_frame.rowconfigure(0, weight=1); cat_frame.columnconfigure(0, weight=1)
            canvas = tk.Canvas(cat_frame, bg=self.BG_COLOR, highlightthickness=0)
            canvas.grid(row=0, column=0, sticky='nsew')
            scrollbar_y = ttk.Scrollbar(cat_frame, orient='vertical', command=canvas.yview)
            scrollbar_y.grid(row=0, column=1, sticky='ns')
            scrollbar_x = ttk.Scrollbar(cat_frame, orient='horizontal', command=canvas.xview)
            scrollbar_x.grid(row=1, column=0, sticky='ew')
            canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
            inner_frame = ttk.Frame(canvas)
            canvas.create_window((0, 0), window=inner_frame, anchor='nw')
            inner_frame.bind('<Configure>', lambda e, c=canvas: c.configure(scrollregion=c.bbox('all')))
            self.category_canvases[class_name] = canvas
            self.category_inner_frames[class_name] = inner_frame


        # --- Columna 2: Recomendaciones e Historial ---
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 0), pady=5)
        right_frame.rowconfigure(1, weight=1) # Área de texto de recomendaciones
        right_frame.rowconfigure(2, weight=2) # Historial (más grande)
        right_frame.columnconfigure(0, weight=1)

        rec_button_frame = ttk.LabelFrame(right_frame, text="3. Ver Recomendaciones", padding="10")
        rec_button_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        rec_button_frame.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(rec_button_frame, text="T. Temprano", style="Rec.TButton", command=lambda: self.show_recommendations(CLASS_NAMES[0])).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(rec_button_frame, text="T. Tardío", style="Rec.TButton", command=lambda: self.show_recommendations(CLASS_NAMES[1])).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(rec_button_frame, text="Sana", style="Rec.TButton", command=lambda: self.show_recommendations(CLASS_NAMES[2])).grid(row=0, column=2, sticky="ew", padx=2)

        self.solutions_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=10, state=tk.DISABLED, font=('Segoe UI', 9), bg="white", fg=self.FG_COLOR, relief="flat", borderwidth=1)
        self.solutions_text.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

        history_frame = ttk.LabelFrame(right_frame, text="4. Historial", padding="10")
        history_frame.grid(row=2, column=0, sticky="nsew")
        history_frame.rowconfigure(0, weight=1); history_frame.columnconfigure(0, weight=1)
        self.history_tree = ttk.Treeview(history_frame, columns=('#', 'Archivo', 'Clase', 'Confianza'), show='headings')
        self.history_tree.heading('#', text='#'); self.history_tree.heading('Archivo', text='Archivo'); self.history_tree.heading('Clase', text='Clase Predicha'); self.history_tree.heading('Confianza', text='Confianza (%)')
        self.history_tree.column('#', anchor=tk.CENTER, width=40, stretch=tk.NO); self.history_tree.column('Archivo', anchor=tk.W, width=130, stretch=tk.YES); self.history_tree.column('Clase', anchor=tk.W, width=100); self.history_tree.column('Confianza', anchor=tk.E, width=80, stretch=tk.NO)
        self.history_tree.grid(row=0, column=0, sticky='nsew')
        hist_scrollbar_y = ttk.Scrollbar(history_frame, orient='vertical', command=self.history_tree.yview); hist_scrollbar_y.grid(row=0, column=1, sticky='ns')
        hist_scrollbar_x = ttk.Scrollbar(history_frame, orient='horizontal', command=self.history_tree.xview); hist_scrollbar_x.grid(row=1, column=0, columnspan=2, sticky='ew')
        self.history_tree.configure(yscrollcommand=hist_scrollbar_y.set, xscrollcommand=hist_scrollbar_x.set)


    def select_files(self):
        filetypes = (('Imágenes', '*.jpg *.jpeg *.png *.bmp *.gif'), ('Todos los archivos', '*.*'))
        filenames = filedialog.askopenfilenames(title='Selecciona imágenes', filetypes=filetypes)
        if filenames:
            self.selected_files = list(filenames)
            self.update_file_list()
            self.run_button.config(state=tk.NORMAL)
            self.clear_displays()
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

    def add_image_to_category_grid(self, img_pil, category, global_number): # Recibe número global
        """Añade una miniatura numerada (global) a la cuadrícula de su categoría."""
        if category not in self.category_inner_frames:
            print(f"Error: Categoría '{category}' no encontrada en los frames.")
            return

        inner_frame = self.category_inner_frames[category]
        canvas = self.category_canvases[category]
        ref_list = self.image_references[category] # Aún necesitamos la lista para guardar refs

        try:
            img_pil_copy = img_pil.copy()
            img_pil_copy.thumbnail(self.thumb_size)
            img_tk = ImageTk.PhotoImage(img_pil_copy)
            ref_list.append(img_tk) # Guardar referencia

            item_frame = ttk.Frame(inner_frame, padding=2)
            img_label = tk.Label(item_frame, image=img_tk, bg=self.BG_COLOR)
            img_label.pack()
            # Usar el número global pasado como argumento
            num_label = tk.Label(item_frame, text=str(global_number), font=('Segoe UI', 8, 'bold'), bg=self.BG_COLOR, fg=self.FG_COLOR)
            num_label.pack()

            # Calcular posición en la cuadrícula de la categoría (basado en cuántos hay YA en ESA categoría)
            num_images_in_cat = sum(1 for w in inner_frame.winfo_children() if isinstance(w, ttk.Frame)) # Contar frames de items
            row = num_images_in_cat // self.images_per_row
            col = num_images_in_cat % self.images_per_row
            item_frame.grid(row=row, column=col, padx=5, pady=5)

            inner_frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox('all'))

        except Exception as e:
            print(f"Error al añadir imagen a la cuadrícula de {category}: {e}")

    def add_to_history(self, number, filename, pred_class, pred_conf):
        confidence_str = f"{pred_conf*100:.2f}"
        self.history_tree.insert('', tk.END, values=(number, filename, pred_class, confidence_str))
        self.history_tree.yview_moveto(1)

    def show_recommendations(self, category):
        solution_text = SOLUTIONS.get(category, "Selecciona una categoría para ver recomendaciones.")
        self.update_solutions_text(solution_text)

    def clear_displays(self):
         for category in CLASS_NAMES:
             inner_frame = self.category_inner_frames.get(category)
             canvas = self.category_canvases.get(category)
             if inner_frame:
                 for widget in inner_frame.winfo_children():
                     widget.destroy()
             if canvas:
                 canvas.yview_moveto(0); canvas.xview_moveto(0)
                 canvas.configure(scrollregion=canvas.bbox('all'))
             self.image_references[category].clear()

         for item in self.history_tree.get_children():
             self.history_tree.delete(item)

         self.update_solutions_text("")
         self.processed_counter = 0


    def request_stop(self):
        """Activa el flag para detener el procesamiento."""
        if messagebox.askyesno("Detener Proceso", "¿Estás seguro de que quieres detener el análisis de imágenes?"):
            self.stop_processing = True
            print("Solicitud de detención recibida.")


    def run_predictions(self):
        if not self.selected_files:
            messagebox.showwarning("Sin Selección", "Por favor, selecciona imágenes primero.")
            return

        self.model = load_model_lazily()
        if self.model is None: return

        self.clear_displays()
        self.update_solutions_text("Procesando... Haz clic en un botón de categoría arriba para ver recomendaciones.")
        self.stop_processing = False # Resetear flag al iniciar

        self.progress_bar['value'] = 0
        self.progress_bar['maximum'] = len(self.selected_files)
        self.root.update_idletasks()

        # Habilitar/Deshabilitar botones
        self.select_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL) # Habilitar botón detener

        processed_count_in_run = 0 # Contador para este ciclo

        for i, filepath in enumerate(self.selected_files):
            # --- Comprobar si se solicitó detener ---
            if self.stop_processing:
                print("Proceso detenido por el usuario.")
                self.update_solutions_text("Proceso detenido por el usuario.")
                break # Salir del bucle
            # -----------------------------------------

            processed_count_in_run += 1 # Incrementar contador global para este ciclo
            filename = os.path.basename(filepath)
            print(f"Procesando #{processed_count_in_run}: {filename}")

            pred_class, pred_conf, raw_scores, img_for_thumb = predict_single_image(self.model, filepath)

            if img_for_thumb:
                # Añadir imagen a la grid con el número global
                self.add_image_to_category_grid(img_for_thumb, pred_class, processed_count_in_run)

            # Añadir al historial con el número global
            self.add_to_history(processed_count_in_run, filename, pred_class, pred_conf)

            self.progress_bar['value'] = i + 1
            if (i + 1) % 5 == 0 or (i + 1) == len(self.selected_files):
                 self.root.update_idletasks()

        # Mensaje final (ajustado si se detuvo)
        if self.stop_processing:
            messagebox.showinfo("Detenido", f"Proceso detenido. Se analizaron {processed_count_in_run-1} imágenes.")
        else:
            self.update_solutions_text("Proceso completado. Haz clic en un botón de categoría para ver recomendaciones.")
            messagebox.showinfo("Completado", f"Se procesaron {processed_count_in_run} imágenes.")

        # Resetear estado de botones al finalizar o detener
        self.select_button.config(state=tk.NORMAL)
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.stop_processing = False # Asegurarse que el flag esté bajo


    def update_solutions_text(self, text):
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
