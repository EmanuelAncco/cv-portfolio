# interfaz_inferencia.py (v1.9 - Laboratorio de Análisis Multimedia)
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageFont, ImageDraw
from ultralytics import YOLO
import os
import logging
import cv2
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("inferencia_v9.log", mode='w'),
        logging.StreamHandler()
    ]
)


class EMARCInferenceApp(tk.Tk):
    """
    Una aplicación de escritorio avanzada para el análisis interactivo de modelos
    de EMARC VISIÓN, con soporte para imágenes, video y cámara en vivo.
    """

    def __init__(self):
        super().__init__()

        self.title("Laboratorio de Inferencia - EMARC VISIÓN")
        self.geometry("1800x950")
        self.configure(bg="#2E2E2E")

        # --- VARIABLES DE ESTADO ---
        self.media_path = None
        self.media_type = None  # 'image', 'video', or 'camera'
        self.original_pil_image = None
        self.processed_pil_image = None
        self.last_inference_result_image = None
        self.model = None
        self.model_name = "Ninguno"
        self.class_colors = {}
        self.magnifier_window = None
        self.hovered_box_index = -1
        self.video_capture = None
        self.video_thread = None
        self.is_playing = False

        self.create_styles()
        self.create_widgets()

        self.after(100, self.load_model)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        accent_color = "#007ACC"
        bg_color = "#2E2E2E"
        control_bg_color = "#3C3C3C"
        text_color = "white"

        style.configure("TFrame", background=bg_color)
        style.configure("TButton", padding=8, relief="flat", background="#5E5E5E", foreground=text_color,
                        font=('Helvetica', 11, 'bold'))
        style.map("TButton", background=[('active', '#7A7A7A'), ('disabled', '#4A4A4A')])
        style.configure("Accent.TButton", background=accent_color, foreground=text_color)
        style.map("Accent.TButton", background=[('active', '#005f9e'), ('disabled', '#003f6e')])
        style.configure("TLabel", background=bg_color, foreground=text_color, font=('Helvetica', 11))
        style.configure("Header.TLabel", font=('Helvetica', 16, 'bold'))
        style.configure("H.TScale", background=control_bg_color)
        style.configure("TCheckbutton", background=control_bg_color, foreground=text_color, font=('Helvetica', 10))
        style.map("TCheckbutton", background=[('active', control_bg_color)],
                  indicatorcolor=[('selected', accent_color), ('disabled', '#555')])
        style.configure("Treeview", background="#3C3C3C", foreground=text_color, fieldbackground="#3C3C3C",
                        rowheight=25, font=('Helvetica', 10))
        style.map("Treeview", background=[('selected', accent_color)])
        style.configure("Treeview.Heading", background="#5E5E5E", foreground=text_color, font=('Helvetica', 12, 'bold'),
                        relief="flat")
        style.map("Treeview.Heading", relief=[('active', 'groove'), ('pressed', 'sunken')])

    def find_best_model(self):
        try:
            runs_dir = 'runs/detect'
            all_run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            if not all_run_dirs: return None, None
            latest_run_dir = max(all_run_dirs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
            model_path = os.path.join(runs_dir, latest_run_dir, 'weights', 'best.pt')
            return (model_path, latest_run_dir) if os.path.exists(model_path) else (None, None)
        except FileNotFoundError:
            return None, None

    def load_model(self):
        self.model_path, self.model_name = self.find_best_model()
        if self.model_path:
            try:
                self.model = YOLO(self.model_path)
                self.class_colors = self.generate_class_colors()
                logging.info(f"Modelo cargado exitosamente desde: {self.model_path}")
                self.model_status_label.config(text=f"Modelo Cargado: {self.model_name}", foreground="#7FFF00")
            except Exception as e:
                messagebox.showerror("Error de Modelo", f"No se pudo cargar el modelo.\nError: {e}")
                self.disable_controls()
        else:
            messagebox.showerror("Error de Modelo", "No se encontró ningún modelo 'best.pt'.")
            self.disable_controls()

    def generate_class_colors(self):
        if self.model and hasattr(self.model, 'names'):
            names = self.model.names
            cmap = plt.get_cmap('viridis', len(names))
            return {name: tuple(int(c * 255) for c in cmap(i)[:3]) for i, name in names.items()}
        return {}

    def create_widgets(self):
        # ... (similar a la v1.8, pero con botones de video/cámara)
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_columnconfigure(0, weight=5);
        main_frame.grid_columnconfigure(1, weight=5);
        main_frame.grid_columnconfigure(2, weight=3)
        main_frame.grid_rowconfigure(1, weight=1)

        # --- PANEL IZQUIERDO ---
        left_panel = ttk.Frame(main_frame, padding=5)
        left_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 5))
        left_panel.grid_rowconfigure(1, weight=1);
        left_panel.grid_columnconfigure(0, weight=1)

        ttk.Label(left_panel, text="Pre-visualización y Controles", style="Header.TLabel").grid(row=0, column=0,
                                                                                                pady=(0, 10))
        self.original_image_label = ttk.Label(left_panel, text="Cargue un archivo...", anchor="center", relief="sunken",
                                              background="#1C1C1C")
        self.original_image_label.grid(row=1, column=0, sticky="nsew")

        self.controls_panel = ttk.Frame(left_panel, padding=(0, 10))
        self.controls_panel.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.controls_panel.grid_columnconfigure(1, weight=1)

        # Botones de Carga
        load_frame = ttk.Frame(self.controls_panel)
        load_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        load_frame.columnconfigure((0, 1, 2), weight=1)
        self.load_img_button = ttk.Button(load_frame, text="Cargar Imagen", command=self.load_image)
        self.load_img_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.load_vid_button = ttk.Button(load_frame, text="Cargar Video", command=self.load_video)
        self.load_vid_button.grid(row=0, column=1, sticky="ew", padx=2)
        self.load_cam_button = ttk.Button(load_frame, text="Activar Cámara", command=self.start_camera)
        self.load_cam_button.grid(row=0, column=2, sticky="ew", padx=(2, 0))

        # ... (resto de controles)
        ttk.Label(self.controls_panel, text="Brillo:").grid(row=1, column=0, sticky="w")
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.brightness_slider = ttk.Scale(self.controls_panel, from_=0.1, to=3.0, variable=self.brightness_var,
                                           command=self.apply_preprocessing)
        self.brightness_slider.grid(row=1, column=1, sticky="ew")
        ttk.Label(self.controls_panel, text="Contraste:").grid(row=2, column=0, sticky="w")
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.contrast_slider = ttk.Scale(self.controls_panel, from_=0.1, to=3.0, variable=self.contrast_var,
                                         command=self.apply_preprocessing)
        self.contrast_slider.grid(row=2, column=1, sticky="ew")
        self.action_frame = ttk.Frame(self.controls_panel)
        self.action_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        self.action_frame.columnconfigure((0, 1), weight=1)
        self.rotate_button = ttk.Button(self.action_frame, text="Rotar 90°", command=self.rotate_image)
        self.rotate_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.reset_button = ttk.Button(self.action_frame, text="Resetear Cambios", command=self.reset_image)
        self.reset_button.grid(row=0, column=1, sticky="ew", padx=(2, 0))
        self.analyze_button = ttk.Button(self.controls_panel, text="Analizar Imagen", command=self.run_inference,
                                         style="Accent.TButton")
        self.analyze_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # --- PANEL CENTRAL ---
        center_panel = ttk.Frame(main_frame, padding=5)
        center_panel.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5)
        center_panel.grid_rowconfigure(1, weight=1);
        center_panel.grid_columnconfigure(0, weight=1)

        right_header = ttk.Frame(center_panel)
        right_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        right_header.columnconfigure(0, weight=1)
        ttk.Label(right_header, text="Resultado del Análisis", style="Header.TLabel").pack(side=tk.LEFT)
        self.model_status_label = ttk.Label(right_header, text="Modelo: No cargado", foreground="orange")
        self.model_status_label.pack(side=tk.RIGHT)

        self.result_image_label = ttk.Label(center_panel, text="El resultado aparecerá aquí...", anchor="center",
                                            relief="sunken", background="#1C1C1C")
        self.result_image_label.grid(row=1, column=0, sticky="nsew")
        self.result_image_label.bind("<B1-Motion>", self.show_magnifier)
        self.result_image_label.bind("<ButtonRelease-1>", self.hide_magnifier)

        # --- PANEL DERECHO ---
        right_panel = ttk.Frame(main_frame, padding=5)
        right_panel.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=(5, 0))
        right_panel.grid_rowconfigure(1, weight=1);
        right_panel.grid_columnconfigure(0, weight=1)

        ttk.Label(right_panel, text="Detalle de Detecciones", style="Header.TLabel").grid(row=0, column=0, pady=(0, 10),
                                                                                          sticky='w')

        self.tree = ttk.Treeview(right_panel, columns=('Clase', 'Confianza'), show='headings')
        self.tree.heading('Clase', text='Clase')
        self.tree.heading('Confianza', text='Confianza (%)')
        self.tree.column('Confianza', anchor='center', width=100)
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.bind('<Motion>', self.on_tree_hover)

        results_footer = ttk.Frame(right_panel)
        results_footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        results_footer.columnconfigure(1, weight=1)
        self.confidence_var = tk.DoubleVar(value=0.25)
        self.confidence_slider = ttk.Scale(results_footer, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                           variable=self.confidence_var, command=self.on_slider_change)
        self.confidence_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        self.confidence_label = ttk.Label(results_footer, text=f"Confianza: {self.confidence_var.get():.0%}")
        self.confidence_label.pack(side=tk.RIGHT)
        self.show_conf_var = tk.BooleanVar(value=True)
        self.show_conf_check = ttk.Checkbutton(results_footer, text="Mostrar Confianza", variable=self.show_conf_var,
                                               command=self.run_inference_if_media_loaded)
        self.show_conf_check.pack(side=tk.LEFT, padx=5)

        self.disable_controls()

    def disable_controls(self):
        # ... (similar a v1.8)
        pass

    def enable_controls(self):
        # ... (similar a v1.8)
        pass

    def load_image(self):
        self.stop_media()
        filepath = filedialog.askopenfilename(filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png")])
        if not filepath: return
        self.media_path = filepath
        self.media_type = 'image'
        self.original_pil_image = Image.open(self.media_path).convert("RGB")
        self.display_pil_image(self.original_pil_image, self.original_image_label)
        self.reset_image()
        self.enable_controls()

    def load_video(self):
        self.stop_media()
        filepath = filedialog.askopenfilename(filetypes=[("Archivos de Video", "*.mp4 *.avi *.mov")])
        if not filepath: return
        self.media_path = filepath
        self.media_type = 'video'
        self.enable_controls()
        # Mostrar el primer frame como previsualización
        cap = cv2.VideoCapture(filepath)
        ret, frame = cap.read()
        if ret:
            self.original_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.display_pil_image(self.original_pil_image, self.original_image_label)
        cap.release()
        self.reset_image()

    def start_camera(self):
        self.stop_media()
        self.media_type = 'camera'
        self.enable_controls()
        self.is_playing = True
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

    def stop_media(self):
        self.is_playing = False
        if self.video_thread and self.video_thread.is_alive():
            time.sleep(0.1)
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

    def video_loop(self):
        if self.media_type == 'video':
            self.video_capture = cv2.VideoCapture(self.media_path)
        elif self.media_type == 'camera':
            self.video_capture = cv2.VideoCapture(0)

        while self.is_playing and self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                if self.media_type == 'video':
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            results = self.model(frame, conf=self.confidence_var.get())
            self.last_inference_result = results[0]
            annotated_frame = self.draw_custom_boxes(results[0])
            self.display_pil_image(Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)),
                                   self.result_image_label)
            self.update_results_table()
            time.sleep(0.01)

        if self.video_capture:
            self.video_capture.release()

    def run_inference(self):
        if self.media_type == 'image':
            if not self.model or not self.processed_pil_image:
                messagebox.showwarning("Sin Imagen", "Por favor, cargue y procese una imagen antes de analizar.")
                return

            image_np_bgr = cv2.cvtColor(np.array(self.processed_pil_image), cv2.COLOR_RGB2BGR)
            results = self.model(image_np_bgr, conf=self.confidence_var.get())
            self.last_inference_result = results[0]
            self.hovered_box_index = -1
            self.redraw_results()
            self.update_results_table()
        elif self.media_type in ['video', 'camera']:
            self.is_playing = not self.is_playing
            if self.is_playing:
                self.analyze_button.config(text="Detener Análisis")
                if not self.video_thread or not self.video_thread.is_alive():
                    self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
                    self.video_thread.start()
            else:
                self.analyze_button.config(text="Analizar Video")

    def redraw_results(self):
        if not self.last_inference_result: return
        annotated_image = self.draw_custom_boxes(self.last_inference_result, self.hovered_box_index)
        self.display_pil_image(Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)),
                               self.result_image_label)

    def update_results_table(self):
        self.clear_results_table()
        if not self.last_inference_result: return

        detections = []
        for i, box in enumerate(self.last_inference_result.boxes):
            class_name = self.model.names[int(box.cls)]
            confidence = float(box.conf)
            detections.append({'id': i, 'class': class_name, 'conf': confidence})

        # Ordenar por confianza
        detections.sort(key=lambda x: x['conf'], reverse=True)

        for det in detections:
            self.tree.insert("", "end", iid=det['id'], values=(det['class'], f"{det['conf'] * 100:.1f}%"))

    def show_magnifier(self, event):
        if not self.last_inference_result: return

        # Usar la imagen ya anotada para la lupa
        annotated_image = Image.fromarray(
            cv2.cvtColor(self.draw_custom_boxes(self.last_inference_result), cv2.COLOR_BGR2RGB))

        # ... (resto de la lógica de la lupa)
        pass

    def on_closing(self):
        self.stop_media()
        self.destroy()

    # ... (resto de funciones de ayuda sin cambios significativos)
    def apply_preprocessing(self, event=None):
        if not self.original_pil_image: return
        image = self.original_pil_image.copy()
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.brightness_var.get())
        enhancer = ImageEnhance.Contrast(image)
        self.processed_pil_image = enhancer.enhance(self.contrast_var.get())
        self.display_pil_image(self.processed_pil_image, self.original_image_label)
        self.result_image_label.config(image='', text="Presiona 'Analizar' para ver el resultado")
        self.clear_results_table()

    def rotate_image(self):
        if not self.original_pil_image: return
        self.original_pil_image = self.original_pil_image.rotate(-90, expand=True)
        self.apply_preprocessing()

    def reset_image(self):
        if not self.original_pil_image: return
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.processed_pil_image = self.original_pil_image.copy()
        self.display_pil_image(self.processed_pil_image, self.original_image_label)
        self.result_image_label.config(image='', text="El resultado aparecerá aquí...")
        self.clear_results_table()

    def run_inference_if_media_loaded(self):
        if self.media_type == 'image':
            self.run_inference()

    def on_slider_change(self, value):
        self.confidence_label.config(text=f"Confianza: {float(value):.0%}")
        if self.media_type == 'image':
            self.run_inference_if_media_loaded()

    def draw_custom_boxes(self, result, highlighted_index=-1):
        img_np = result.orig_img
        pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        for i, box in enumerate(result.boxes):
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            confidence = float(box.conf)
            coords = box.xyxy[0].tolist()
            color = self.class_colors.get(class_name, (255, 255, 255))
            width = 5 if i == highlighted_index else 3

            draw.rectangle(coords, outline=color, width=width)

            if self.show_conf_var.get():
                text = f"{class_name}: {confidence:.2f}"
                text_bbox = draw.textbbox((coords[0], coords[1] - 15), text, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((coords[0], coords[1] - 15), text, fill=(0, 0, 0), font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def display_pil_image(self, pil_image, label_widget):
        w, h = label_widget.winfo_width(), label_widget.winfo_height()
        if w < 2 or h < 2: w, h = 800, 600

        display_image = pil_image.copy()
        display_image.thumbnail((w - 10, h - 10), Image.Resampling.LANCZOS)

        tk_image = ImageTk.PhotoImage(display_image)
        label_widget.image = tk_image
        label_widget.config(image=tk_image)

    def clear_results_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def on_tree_hover(self, event):
        item_id = self.tree.identify_row(event.y)
        if item_id:
            box_index = int(item_id)
            if box_index != self.hovered_box_index:
                self.hovered_box_index = box_index
                self.redraw_results()
        else:
            if self.hovered_box_index != -1:
                self.hovered_box_index = -1
                self.redraw_results()

    def hide_magnifier(self, event):
        if self.magnifier_window:
            self.magnifier_window.destroy()
            self.magnifier_window = None


if __name__ == "__main__":
    app = EMARCInferenceApp()
    app.mainloop()
