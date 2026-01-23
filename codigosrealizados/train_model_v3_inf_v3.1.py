import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageFont, ImageDraw, ImageOps
from ultralytics import YOLO
import os
import logging
import cv2
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime
import zipfile

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("inferencia_v2.6.log", mode='w'),
        logging.StreamHandler()
    ]
)


class EMARCInferenceApp(tk.Tk):
    """
    Una aplicación de escritorio avanzada para el análisis interactivo de modelos
    de EMARC VISIÓN, con carga de modelo robusta.
    """

    def __init__(self):
        """
        Inicializa la aplicación, configura la ventana principal y las variables de estado.
        """
        super().__init__()

        self.title("Laboratorio de Inferencia - EMARC VISIÓN v2.6")
        self.geometry("1800x950")
        self.configure(bg="#2E2E2E")

        # --- VARIABLES DE ESTADO ---
        self.media_path = None
        self.media_type = None
        self.original_pil_image = None
        self.processed_pil_image = None
        self.last_inference_result = None
        self.model = None
        self.model_name = "Ninguno"
        self.class_colors = {}
        self.magnifier_window = None
        self.hovered_box_index = -1

        self.video_capture = None
        self.video_thread = None
        self.is_playing = False
        self.video_paused = False
        self.video_total_frames = 0
        self.video_fps = 30
        self.current_frame_for_capture = None

        self.analysis_history = []
        self.history_labels = []

        self.create_styles()
        self.create_widgets()

        self.after(100, self.load_model)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        logging.info("Aplicación EMARCInferenceApp inicializada.")

    def create_styles(self):
        """
        Configura los estilos visuales para los widgets de Tkinter usando ttk.Style.
        """
        logging.info("Configurando estilos de la interfaz.")
        style = ttk.Style(self)
        style.theme_use('clam')

        self.accent_color = "#007ACC"
        self.bg_color = "#2E2E2E"
        self.control_bg_color = "#3C3C3C"
        self.text_color = "white"

        style.configure("TFrame", background=self.bg_color)
        style.configure("TButton", padding=8, relief="flat", background="#5E5E5E", foreground=self.text_color,
                        font=('Helvetica', 11, 'bold'))
        style.map("TButton", background=[('active', '#7A7A7A'), ('disabled', '#4A4A4A')])
        style.configure("Accent.TButton", background=self.accent_color, foreground=self.text_color)
        style.map("Accent.TButton", background=[('active', '#005f9e'), ('disabled', '#003f6e')])
        style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=('Helvetica', 11))
        style.configure("Header.TLabel", font=('Helvetica', 16, 'bold'))
        style.configure("Small.Header.TLabel", font=('Helvetica', 12, 'bold'))
        style.configure("Horizontal.TScale", background=self.control_bg_color, troughcolor="#555555",
                        sliderrelief="flat")
        style.configure("TCheckbutton", background=self.control_bg_color, foreground=self.text_color,
                        font=('Helvetica', 10))
        style.map("TCheckbutton", background=[('active', self.control_bg_color)],
                  indicatorcolor=[('selected', self.accent_color), ('disabled', '#555')])
        style.configure("Treeview", background="#3C3C3C", foreground=self.text_color, fieldbackground="#3C3C3C",
                        rowheight=25, font=('Helvetica', 10))
        style.map("Treeview", background=[('selected', self.accent_color)])
        style.configure("Treeview.Heading", background="#5E5E5E", foreground=self.text_color,
                        font=('Helvetica', 12, 'bold'),
                        relief="flat")
        style.map("Treeview.Heading", relief=[('active', 'groove'), ('pressed', 'sunken')])
        style.configure("TMenubutton", background="#5E5E5E", foreground=self.text_color, font=('Helvetica', 11, 'bold'))
        style.map("TMenubutton", background=[('active', '#7A7A7A'), ('disabled', '#4A4A4A')])
        logging.info("Estilos configurados.")

    def find_best_model(self):
        """
        CORREGIDO (v2.6): Busca de forma robusta el modelo 'best.pt' más reciente
        en los directorios de 'runs/detect', ignorando ejecuciones fallidas.
        """
        logging.info("Buscando el modelo 'best.pt' más reciente...")
        runs_dir = 'runs/detect'
        if not os.path.exists(runs_dir):
            logging.error(f"El directorio de ejecuciones '{runs_dir}' no existe.")
            return None, None

        best_pt_files = []
        try:
            for run_name in os.listdir(runs_dir):
                run_path = os.path.join(runs_dir, run_name)
                if os.path.isdir(run_path):
                    weights_path = os.path.join(run_path, 'weights', 'best.pt')
                    if os.path.exists(weights_path):
                        # Guardar la ruta completa, el nombre del run y la fecha de modificación
                        mod_time = os.path.getmtime(run_path)
                        best_pt_files.append((weights_path, run_name, mod_time))
        except Exception as e:
            logging.error(f"Error al listar directorios de ejecuciones: {e}")
            return None, None

        if not best_pt_files:
            logging.error(f"No se encontró ningún archivo 'best.pt' en las subcarpetas de '{runs_dir}'.")
            return None, None

        # Ordenar los modelos encontrados por fecha de modificación (el más reciente primero)
        best_pt_files.sort(key=lambda x: x[2], reverse=True)

        # Devolver el más reciente
        latest_model_path, latest_run_name, _ = best_pt_files[0]
        logging.info(f"Modelo 'best.pt' más reciente encontrado en: {latest_model_path}")
        return latest_model_path, latest_run_name

    def load_model(self):
        """
        Carga el modelo YOLO encontrado y actualiza la UI.
        """
        logging.info("Intentando cargar el modelo YOLO.")
        self.model_path, self.model_name = self.find_best_model()
        if self.model_path:
            try:
                self.model = YOLO(self.model_path)
                if hasattr(self.model, 'names') and self.model.names:
                    self.class_colors = self.generate_class_colors(self.model.names)
                else:
                    self.class_colors = {}
                logging.info(f"Modelo '{self.model_name}' cargado exitosamente desde: {self.model_path}")
                self.model_status_label.config(text=f"Modelo Cargado: {self.model_name}", foreground="#7FFF00")
                self.enable_controls()
            except Exception as e:
                logging.error(f"Error al cargar el modelo YOLO desde {self.model_path}: {e}")
                messagebox.showerror("Error de Modelo", f"No se pudo cargar el modelo.\nError: {e}")
                self.disable_controls()
        else:
            logging.error("No se encontró ningún modelo 'best.pt' para cargar.")
            messagebox.showerror("Error de Modelo",
                                 "No se encontró ningún modelo 'best.pt' en la carpeta 'runs/detect'.\nAsegúrese de que haya finalizado un entrenamiento correctamente.")
            self.disable_controls()

    def generate_class_colors(self, class_names):
        """
        Genera un esquema de colores consistente y estético para cada clase.
        """
        custom_palette = [
            (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
            (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
            (188, 189, 34), (23, 190, 207), (255, 152, 150), (152, 223, 138)
        ]
        colors = {}
        for i, name in enumerate(class_names.values()):
            colors[name] = custom_palette[i % len(custom_palette)]
        return colors

    def create_widgets(self):
        """
        Crea y organiza todos los widgets de la interfaz de usuario.
        """
        logging.info("Creando widgets de la interfaz.")
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_columnconfigure(0, weight=5)
        main_frame.grid_columnconfigure(1, weight=5)
        main_frame.grid_columnconfigure(2, weight=3)
        main_frame.grid_rowconfigure(1, weight=1)

        # --- PANEL IZQUIERDO: Pre-visualización y Controles ---
        left_panel = ttk.Frame(main_frame, padding=5)
        left_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 5))
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        ttk.Label(left_panel, text="Pre-visualización y Controles", style="Header.TLabel").grid(row=0, column=0,
                                                                                                pady=(0, 10))
        self.original_image_label = ttk.Label(left_panel, text="Cargue un archivo...", anchor="center", relief="sunken",
                                              background="#1C1C1C")
        self.original_image_label.grid(row=1, column=0, sticky="nsew")
        self.original_image_label.bind("<Configure>", lambda e: self.display_pil_image(
            self.processed_pil_image if self.processed_pil_image else self.original_pil_image,
            self.original_image_label) if self.processed_pil_image or self.original_pil_image else None)

        self.controls_panel = ttk.Frame(left_panel, padding=(0, 10))
        self.controls_panel.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.controls_panel.grid_columnconfigure(1, weight=1)

        load_frame = ttk.Frame(self.controls_panel)
        load_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        load_frame.columnconfigure((0, 1, 2), weight=1)
        self.load_img_button = ttk.Button(load_frame, text="Cargar Imagen", command=self.load_image)
        self.load_img_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.load_vid_button = ttk.Button(load_frame, text="Cargar Video", command=self.load_video)
        self.load_vid_button.grid(row=0, column=1, sticky="ew", padx=2)
        self.camera_menu_button = ttk.Menubutton(load_frame, text="Activar Cámara")
        self.camera_menu = tk.Menu(self.camera_menu_button, tearoff=0)
        self.camera_menu_button["menu"] = self.camera_menu
        self.camera_menu.config(postcommand=self.populate_camera_menu)
        self.camera_menu_button.grid(row=0, column=2, sticky="ew", padx=(2, 0))
        ttk.Label(self.controls_panel, text="Brillo:").grid(row=1, column=0, sticky="w")
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.brightness_slider = ttk.Scale(self.controls_panel, from_=0.1, to=3.0, variable=self.brightness_var,
                                           command=self.apply_preprocessing, style="Horizontal.TScale")
        self.brightness_slider.grid(row=1, column=1, sticky="ew")
        ttk.Label(self.controls_panel, text="Contraste:").grid(row=2, column=0, sticky="w")
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.contrast_slider = ttk.Scale(self.controls_panel, from_=0.1, to=3.0, variable=self.contrast_var,
                                         command=self.apply_preprocessing, style="Horizontal.TScale")
        self.contrast_slider.grid(row=2, column=1, sticky="ew")
        self.action_frame = ttk.Frame(self.controls_panel)
        self.action_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        self.action_frame.columnconfigure((0, 1), weight=1)
        self.rotate_button = ttk.Button(self.action_frame, text="Rotar 90°", command=self.rotate_image)
        self.rotate_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.reset_button = ttk.Button(self.action_frame, text="Resetear Cambios", command=self.reset_image)
        self.reset_button.grid(row=0, column=1, sticky="ew", padx=(2, 0))
        self.analyze_button = ttk.Button(self.controls_panel, text="Analizar Contenido", command=self.toggle_analysis,
                                         style="Accent.TButton")
        self.analyze_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.video_control_panel = ttk.Frame(self.controls_panel, padding=(0, 10))
        self.video_control_panel.grid_columnconfigure(0, weight=1)
        self.video_control_panel.grid_columnconfigure(1, weight=0)
        self.video_seek_slider = ttk.Scale(self.video_control_panel, from_=0, to=100, orient=tk.HORIZONTAL,
                                           command=self.seek_video, style="Horizontal.TScale")
        self.video_seek_slider.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.capture_button = ttk.Button(self.video_control_panel, text="Capturar Frame", command=self.capture_frame)
        self.capture_button.grid(row=0, column=1, sticky="e")
        self.video_control_panel.grid_forget()

        # --- PANEL CENTRAL: Resultado del Análisis ---
        center_panel = ttk.Frame(main_frame, padding=5)
        center_panel.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5)
        center_panel.grid_rowconfigure(1, weight=1)
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
        self.result_image_label.bind("<Configure>",
                                     lambda e: self.redraw_results() if self.last_inference_result else None)
        self.result_image_label.bind("<B1-Motion>", self.show_magnifier)
        self.result_image_label.bind("<ButtonRelease-1>", self.hide_magnifier)

        # --- PANEL DERECHO: Detalle de Detecciones y Controles ---
        right_panel = ttk.Frame(main_frame, padding=5)
        right_panel.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=(5, 0))
        right_panel.grid_rowconfigure(1, weight=2)
        right_panel.grid_rowconfigure(3, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        ttk.Label(right_panel, text="Detalle de Detecciones", style="Header.TLabel").grid(row=0, column=0, pady=(0, 10),
                                                                                          sticky='w')

        self.tree = ttk.Treeview(right_panel, columns=('Clase', 'Confianza'), show='headings')
        self.tree.heading('Clase', text='Clase')
        self.tree.heading('Confianza', text='Confianza (%)')
        self.tree.column('Confianza', anchor='center', width=100)
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.bind('<Motion>', self.on_tree_hover)
        self.tree.bind('<Leave>', self.on_tree_leave)

        results_footer = ttk.Frame(right_panel)
        results_footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        results_footer.columnconfigure(1, weight=1)

        self.show_conf_var = tk.BooleanVar(value=True)
        self.show_conf_check = ttk.Checkbutton(results_footer, text="Mostrar Confianza", variable=self.show_conf_var,
                                               command=self.run_inference_if_media_loaded)
        self.show_conf_check.pack(side=tk.LEFT, padx=5)

        self.heatmap_button = ttk.Button(results_footer, text="Ver Mapa de Confianza",
                                         command=self.show_confidence_heatmap)
        self.heatmap_button.pack(side=tk.LEFT, padx=5)

        self.confidence_var = tk.DoubleVar(value=0.25)
        self.confidence_slider = ttk.Scale(results_footer, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                           variable=self.confidence_var, command=self.on_slider_change,
                                           style="Horizontal.TScale")
        self.confidence_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        self.confidence_label = ttk.Label(results_footer, text=f"Confianza: {self.confidence_var.get():.0%}")
        self.confidence_label.pack(side=tk.RIGHT)

        history_panel = ttk.Frame(right_panel, padding=(0, 15))
        history_panel.grid(row=3, column=0, sticky="nsew")
        history_panel.grid_columnconfigure(0, weight=1)
        history_panel.grid_rowconfigure(1, weight=1)

        history_header = ttk.Frame(history_panel)
        history_header.grid(row=0, column=0, sticky="ew", pady=(10, 5))
        ttk.Label(history_header, text="Historial de Análisis", style="Small.Header.TLabel").pack(side=tk.LEFT)
        self.download_button = ttk.Button(history_header, text="Descargar Historial", command=self.download_history)
        self.download_button.pack(side=tk.RIGHT)

        history_canvas_container = ttk.Frame(history_panel, relief="sunken")
        history_canvas_container.grid(row=1, column=0, sticky="nsew")
        history_canvas_container.grid_rowconfigure(0, weight=1)
        history_canvas_container.grid_columnconfigure(0, weight=1)

        self.history_canvas = tk.Canvas(history_canvas_container, bg=self.control_bg_color, highlightthickness=0)
        self.history_scrollbar = ttk.Scrollbar(history_canvas_container, orient="horizontal",
                                               command=self.history_canvas.xview)
        self.history_canvas.configure(xscrollcommand=self.history_scrollbar.set)

        self.history_scrollbar.grid(row=1, column=0, sticky="ew")
        self.history_canvas.grid(row=0, column=0, sticky="nsew")

        self.history_frame = ttk.Frame(self.history_canvas)
        self.history_canvas.create_window((0, 0), window=self.history_frame, anchor="nw")
        self.history_frame.bind("<Configure>",
                                lambda e: self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all")))

        self.disable_controls()
        logging.info("Widgets de la interfaz creados.")

    def auto_orient_image(self, pil_image):
        """
        Corrige la orientación de la imagen basándose en los datos EXIF.
        """
        try:
            return ImageOps.exif_transpose(pil_image)
        except Exception as e:
            logging.warning(f"No se pudo aplicar la auto-orientación EXIF: {e}")
            return pil_image

    def find_available_cameras(self):
        """
        Detecta las cámaras disponibles en el sistema.
        """
        available_cameras = []
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret: available_cameras.append(i)
                    cap.release()
            except Exception:
                pass
        return available_cameras

    def populate_camera_menu(self):
        """
        Puebla el menú de selección de cámara.
        """
        self.camera_menu.delete(0, tk.END)
        cameras = self.find_available_cameras()
        if not cameras:
            self.camera_menu.add_command(label="No se encontraron cámaras", state=tk.DISABLED)
            return
        for cam_idx in cameras:
            self.camera_menu.add_command(label=f"Cámara {cam_idx}", command=lambda idx=cam_idx: self.start_camera(idx))

    def disable_controls(self):
        """
        Deshabilita los controles de usuario.
        """
        for child in self.controls_panel.winfo_children():
            try:
                child.config(state=tk.DISABLED)
            except tk.TclError:
                for grandchild in child.winfo_children():
                    grandchild.config(state=tk.DISABLED)
        self.video_control_panel.grid_forget()
        self.download_button.config(state=tk.DISABLED)
        self.heatmap_button.config(state=tk.DISABLED)

    def enable_controls(self):
        """
        Habilita los controles de usuario.
        """
        for child in self.controls_panel.winfo_children():
            try:
                child.config(state=tk.NORMAL)
            except tk.TclError:
                for grandchild in child.winfo_children():
                    grandchild.config(state=tk.NORMAL)
        self.download_button.config(state=tk.NORMAL if self.analysis_history else tk.DISABLED)
        self.heatmap_button.config(state=tk.NORMAL if self.last_inference_result else tk.DISABLED)

    def load_image(self):
        """
        Carga un archivo de imagen, lo auto-orienta y lo muestra.
        """
        self.stop_media()
        filepath = filedialog.askopenfilename(filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png *.bmp")])
        if not filepath: return
        try:
            self.media_path = filepath
            self.media_type = 'image'
            img = Image.open(self.media_path).convert("RGB")
            self.original_pil_image = self.auto_orient_image(img)
            self.display_pil_image(self.original_pil_image, self.original_image_label)
            self.reset_image()
            self.enable_controls_for_image()
            logging.info(f"Imagen cargada y orientada: {self.media_path}")
        except Exception as e:
            messagebox.showerror("Error de Carga", f"No se pudo cargar la imagen:\n{e}")

    def load_video(self):
        """
        Carga un archivo de video.
        """
        self.stop_media()
        filepath = filedialog.askopenfilename(filetypes=[("Archivos de Video", "*.mp4 *.avi *.mov *.mkv")])
        if not filepath: return
        try:
            self.media_path = filepath
            self.media_type = 'video'
            self.video_capture = cv2.VideoCapture(filepath)
            if not self.video_capture.isOpened(): raise IOError("No se pudo abrir el video.")
            self.video_total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
            ret, frame = self.video_capture.read()
            if ret:
                self.original_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.display_pil_image(self.original_pil_image, self.original_image_label)
                self.reset_image()
            self.enable_controls_for_video_camera()
            self.video_seek_slider.config(to=self.video_total_frames - 1)
            self.video_seek_slider.set(0)
        except Exception as e:
            messagebox.showerror("Error de Carga", f"No se pudo cargar el video:\n{e}")
            self.stop_media()

    def start_camera(self, camera_index=0):
        """
        Inicia la captura de video desde la cámara.
        """
        self.stop_media()
        try:
            self.media_path = f"camera_{camera_index}"
            self.media_type = 'camera'
            self.video_capture = cv2.VideoCapture(camera_index)
            if not self.video_capture.isOpened(): raise IOError(f"No se pudo abrir la cámara {camera_index}.")
            self.enable_controls_for_video_camera()
            self.is_playing = True
            self.video_paused = False
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            self.analyze_button.config(text="Detener Cámara")
            self.video_seek_slider.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error de Cámara", f"No se pudo iniciar la cámara {camera_index}:\n{e}")
            self.stop_media()

    def stop_media(self):
        """
        Detiene la reproducción de video/cámara.
        """
        self.is_playing = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=0.5)
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.analyze_button.config(text="Analizar Contenido")

    def toggle_analysis(self):
        """
        Controla el inicio/pausa/reanudación del análisis.
        """
        if self.model is None:
            messagebox.showwarning("Modelo no cargado", "Por favor, cargue un modelo.")
            return
        if self.media_type == 'image':
            if self.processed_pil_image:
                self.run_inference_on_image()
            else:
                messagebox.showwarning("Sin Imagen", "Cargue una imagen.")
        elif self.media_type in ['video', 'camera']:
            if not self.video_capture and self.media_type == 'video':
                messagebox.showwarning("Sin Video", "Cargue un video primero.")
                return
            if self.is_playing:
                self.video_paused = not self.video_paused
                self.analyze_button.config(text="Reanudar Análisis" if self.video_paused else "Pausar Análisis")
            else:
                self.is_playing = True
                self.video_paused = False
                self.analyze_button.config(text="Pausar Análisis")
                if not self.video_thread or not self.video_thread.is_alive():
                    self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
                    self.video_thread.start()

    def run_inference_on_image(self):
        """
        Ejecuta la inferencia en la imagen, garantizando el redimensionamiento.
        """
        logging.info("Ejecutando inferencia en la imagen.")
        if not self.model or not self.processed_pil_image:
            return

        try:
            logging.info(
                f"Realizando inferencia en imagen (tamaño original: {self.processed_pil_image.size}), se redimensionará a 640.")
            results = self.model.predict(
                source=self.processed_pil_image,
                imgsz=640,
                conf=self.confidence_var.get(),
                verbose=False
            )

            if results:
                self.last_inference_result = results[0]
                self.hovered_box_index = -1
                self.current_frame_for_capture = cv2.cvtColor(np.array(self.processed_pil_image), cv2.COLOR_RGB2BGR)

                self.update_results_table()
                annotated_image = self.get_annotated_image()
                self.add_to_history(annotated_image)
                self.enable_controls()

                self.redraw_results()
                logging.info("Inferencia en imagen completada.")
            else:
                self.last_inference_result = None
                self.clear_results_table()
                self.display_pil_image(self.processed_pil_image, self.result_image_label)
                logging.info("No se detectaron objetos.")

        except Exception as e:
            logging.error(f"Error durante la inferencia en la imagen: {e}", exc_info=True)
            messagebox.showerror("Error de Inferencia", f"No se pudo ejecutar la inferencia:\n{e}")
            self.last_inference_result = None
            self.clear_results_table()

    def video_loop(self):
        """
        Bucle principal para procesar fotogramas de video o cámara.
        """
        logging.info(f"Hilo de video iniciado para {self.media_type}.")
        if self.media_type == 'video' and self.video_capture is None:
            self.video_capture = cv2.VideoCapture(self.media_path)
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.video_seek_slider.get())

        while self.is_playing:
            if self.video_paused:
                time.sleep(0.1)
                continue
            if not self.video_capture or not self.video_capture.isOpened():
                logging.warning("El stream de video no está disponible. Deteniendo bucle.")
                break

            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    if self.media_type == 'video':
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                self.current_frame_for_capture = frame.copy()
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                enhancer_b = ImageEnhance.Brightness(pil_frame)
                pil_frame = enhancer_b.enhance(self.brightness_var.get())
                enhancer_c = ImageEnhance.Contrast(pil_frame)
                pil_frame_processed = enhancer_c.enhance(self.contrast_var.get())

                self.display_pil_image(pil_frame_processed, self.original_image_label)

                results = self.model.predict(
                    source=pil_frame_processed,
                    imgsz=640,
                    conf=self.confidence_var.get(),
                    verbose=False
                )

                if results and results[0].boxes:
                    self.last_inference_result = results[0]
                    self.update_results_table()
                    annotated_frame_bgr = self.draw_custom_boxes(results[0])
                    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB))
                    self.display_pil_image(annotated_pil, self.result_image_label)
                else:
                    self.last_inference_result = None
                    self.display_pil_image(pil_frame_processed, self.result_image_label)
                    self.clear_results_table()

                if self.media_type == 'video':
                    current_frame_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                    self.video_seek_slider.set(current_frame_pos)

                time.sleep(1.0 / self.video_fps)

            except Exception as e:
                logging.error(f"Error en video_loop: {e}", exc_info=True)
                break

        self.is_playing = False
        self.analyze_button.config(text="Analizar Contenido")
        logging.info("Hilo de video finalizado.")

    def seek_video(self, value):
        if self.media_type != 'video' or not self.video_capture: return
        frame_idx = int(float(value))
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_capture.read()
        if ret:
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.processed_pil_image = pil_frame
            self.apply_preprocessing()
            self.display_pil_image(self.processed_pil_image, self.original_image_label)
            if not self.is_playing or self.video_paused: self.run_inference_on_image()

    def capture_frame(self):
        if self.last_inference_result is None:
            messagebox.showwarning("Sin Frame", "No hay un resultado de análisis para capturar.")
            return
        try:
            output_dir = "capturas"
            os.makedirs(output_dir, exist_ok=True)
            annotated_image = self.get_annotated_image()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"captura_emarc_{timestamp}.png")
            annotated_image.save(filename, "PNG")
            messagebox.showinfo("Captura Guardada", f"Fotograma guardado como:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error de Captura", f"No se pudo guardar el fotograma:\n{e}")

    def add_to_history(self, full_result_image):
        """
        Añade una imagen al historial de análisis.
        """
        thumbnail = full_result_image.copy()
        thumbnail.thumbnail((100, 100), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(thumbnail)
        self.history_labels.append(photo)

        self.analysis_history.append((photo, full_result_image))
        self.update_history_display()

    def update_history_display(self):
        """
        Actualiza la galería de miniaturas en el historial.
        """
        for widget in self.history_frame.winfo_children():
            widget.destroy()

        for i, (photo, _) in enumerate(self.analysis_history):
            label = ttk.Label(self.history_frame, image=photo, relief="solid", padding=2)
            label.pack(side=tk.LEFT, padx=5, pady=5)

    def download_history(self):
        """
        Descarga las imágenes del historial a una carpeta.
        """
        if not self.analysis_history:
            messagebox.showinfo("Historial Vacío", "No hay imágenes procesadas para descargar.")
            return

        save_path = filedialog.askdirectory(title="Seleccione una carpeta para guardar el historial")
        if not save_path: return

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"EMARC_Resultados_{timestamp}"
            full_path = os.path.join(save_path, folder_name)
            os.makedirs(full_path, exist_ok=True)

            for i, (_, full_image) in enumerate(self.analysis_history):
                image_path = os.path.join(full_path, f"resultado_{i + 1}.png")
                full_image.save(image_path, "PNG")

            messagebox.showinfo("Descarga Completa",
                                f"{len(self.analysis_history)} imágenes guardadas en:\n{full_path}")
        except Exception as e:
            messagebox.showerror("Error de Descarga", f"No se pudo guardar el historial:\n{e}")

    def get_annotated_image(self):
        """
        Devuelve una imagen PIL con las anotaciones dibujadas.
        """
        if not self.last_inference_result:
            return self.processed_pil_image.copy() if self.processed_pil_image else Image.new('RGB', (100, 100))

        annotated_array = self.draw_custom_boxes(self.last_inference_result, self.hovered_box_index)
        return Image.fromarray(cv2.cvtColor(annotated_array, cv2.COLOR_BGR2RGB))

    def redraw_results(self):
        if not self.last_inference_result:
            if self.processed_pil_image: self.display_pil_image(self.processed_pil_image, self.result_image_label)
            return
        annotated_image_bgr = self.draw_custom_boxes(self.last_inference_result, self.hovered_box_index)
        annotated_pil_image = Image.fromarray(cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB))
        self.display_pil_image(annotated_pil_image, self.result_image_label)

    def update_results_table(self):
        self.clear_results_table()
        if not self.last_inference_result or not self.last_inference_result.boxes: return
        boxes = self.last_inference_result.boxes.xyxy.cpu().numpy()
        confidences = self.last_inference_result.boxes.conf.cpu().numpy()
        class_ids = self.last_inference_result.boxes.cls.cpu().numpy().astype(int)
        nms_boxes = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes]
        iou_threshold = 0.4
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, self.confidence_var.get(), iou_threshold)
        detections_to_display = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections_to_display.append({'id': i, 'class': self.model.names[class_ids[i]], 'conf': confidences[i]})
        detections_to_display.sort(key=lambda x: x['conf'], reverse=True)
        for det in detections_to_display:
            self.tree.insert("", "end", iid=det['id'], values=(det['class'], f"{det['conf'] * 100:.1f}%"))

    def show_magnifier(self, event):
        if not self.last_inference_result: return
        if not hasattr(self.result_image_label,
                       'image_original_size_for_magnifier') or self.result_image_label.image_original_size_for_magnifier is None: return

        annotated_image_pil = self.get_annotated_image()

        label_x, label_y = event.x, event.y
        label_width, label_height = self.result_image_label.winfo_width(), self.result_image_label.winfo_height()
        if label_width < 2 or label_height < 2: return
        img_width, img_height = annotated_image_pil.size
        scale = max(img_width / label_width, img_height / label_height)
        img_x, img_y = int(label_x * scale), int(label_y * scale)
        magnifier_size = 200
        zoom_factor = 3
        half_mag = magnifier_size // (2 * zoom_factor)
        crop_box = (max(0, img_x - half_mag), max(0, img_y - half_mag), min(img_width, img_x + half_mag),
                    min(img_height, img_y + half_mag))
        magnified_region = annotated_image_pil.crop(crop_box).resize((magnifier_size, magnifier_size),
                                                                     Image.Resampling.NEAREST)
        if not self.magnifier_window or not self.magnifier_window.winfo_exists():
            self.magnifier_window = tk.Toplevel(self)
            self.magnifier_window.overrideredirect(True)
            self.magnifier_window.attributes("-topmost", True)
            self.magnifier_label = ttk.Label(self.magnifier_window)
            self.magnifier_label.pack()
        mouse_x, mouse_y = self.winfo_pointerx(), self.winfo_pointery()
        self.magnifier_window.geometry(f"{magnifier_size}x{magnifier_size}+{mouse_x + 20}+{mouse_y + 20}")
        tk_mag_image = ImageTk.PhotoImage(magnified_region)
        self.magnifier_label.config(image=tk_mag_image)
        self.magnifier_label.image = tk_mag_image

    def hide_magnifier(self, event=None):
        if self.magnifier_window:
            self.magnifier_window.destroy()
            self.magnifier_window = None

    def on_closing(self):
        logging.info("Cerrando aplicación.")
        self.stop_media()
        self.destroy()

    def apply_preprocessing(self, event=None):
        if not self.original_pil_image: return
        image = self.original_pil_image.copy()
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.brightness_var.get())
        enhancer = ImageEnhance.Contrast(image)
        self.processed_pil_image = enhancer.enhance(self.contrast_var.get())
        self.display_pil_image(self.processed_pil_image, self.original_image_label)
        if self.media_type == 'image':
            self.result_image_label.config(image='', text="Presiona 'Analizar' para ver el resultado")
            self.clear_results_table()
            self.last_inference_result = None

    def rotate_image(self):
        if not self.original_pil_image: return
        self.original_pil_image = self.original_pil_image.rotate(90, expand=True)
        self.apply_preprocessing()

    def reset_image(self):
        if not self.original_pil_image: return
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.processed_pil_image = self.original_pil_image.copy()
        self.display_pil_image(self.processed_pil_image, self.original_image_label)
        self.result_image_label.config(image='', text="El resultado aparecerá aquí...")
        self.clear_results_table()
        self.last_inference_result = None

    def run_inference_if_media_loaded(self, event=None):
        if self.media_type == 'image' and self.processed_pil_image:
            self.run_inference_on_image()

    def on_slider_change(self, value):
        self.confidence_label.config(text=f"Confianza: {float(value):.0%}")
        self.run_inference_if_media_loaded()

    def display_pil_image(self, pil_image, label_widget):
        if pil_image is None:
            label_widget.config(image='', text="Cargue un archivo...")
            return
        w, h = label_widget.winfo_width(), label_widget.winfo_height()
        if w < 2 or h < 2: return
        display_image = pil_image.copy()
        display_image.thumbnail((w - 10, h - 10), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(display_image)
        label_widget.config(image=tk_image)
        label_widget.image = tk_image
        label_widget.image_original_size_for_magnifier = pil_image

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
            self.on_tree_leave()

    def on_tree_leave(self, event=None):
        if self.hovered_box_index != -1:
            self.hovered_box_index = -1
            self.redraw_results()

    def enable_controls_for_image(self):
        self.enable_controls()
        self.analyze_button.config(text="Analizar Imagen")
        self.video_control_panel.grid_forget()

    def enable_controls_for_video_camera(self):
        self.enable_controls()
        self.analyze_button.config(text="Iniciar Análisis")
        self.video_control_panel.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.capture_button.config(state=tk.NORMAL)

    def draw_custom_boxes(self, result, highlighted_index=-1):
        """
        Dibuja los cuadros delimitadores y las etiquetas en la imagen con texto legible.
        """
        pil_img = Image.fromarray(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("seguisb.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        for item_id in self.tree.get_children():
            original_idx = int(item_id)
            if original_idx < len(result.boxes):
                box = result.boxes[original_idx]
                class_id = int(box.cls)
                class_name = self.model.names[class_id]
                confidence = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                color = self.class_colors.get(class_name, (255, 255, 255))

                width = 5 if original_idx == highlighted_index else 3
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

                if self.show_conf_var.get():
                    text = f"{class_name}: {confidence:.2f}"

                    try:
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except AttributeError:  # Fallback for older PIL
                        text_width, text_height = draw.textsize(text, font=font)

                    text_y = y1 - (text_height + 4)
                    if text_y < 0:
                        text_y = y1

                    draw.rectangle([x1, text_y, x1 + text_width + 4, text_y + text_height + 4], fill=color)
                    draw.text((x1 + 2, text_y + 2), text, fill="white", font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def show_confidence_heatmap(self):
        """
        Genera y muestra un mapa de calor de confianza.
        """
        if not self.last_inference_result:
            messagebox.showinfo("Sin Análisis", "Primero debe analizar una imagen.")
            return

        try:
            image = Image.fromarray(cv2.cvtColor(self.last_inference_result.orig_img, cv2.COLOR_BGR2RGB))
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            cmap = plt.get_cmap('viridis')

            for item_id in self.tree.get_children():
                original_idx = int(item_id)
                if original_idx < len(self.last_inference_result.boxes):
                    box = self.last_inference_result.boxes[original_idx]
                    confidence = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    color = cmap(confidence)
                    fill_color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 128)

                    draw.rectangle([x1, y1, x2, y2], fill=fill_color)

            combined = Image.alpha_composite(image.convert('RGBA'), overlay)

            heatmap_window = tk.Toplevel(self)
            heatmap_window.title("Mapa de Calor de Confianza")
            heatmap_window.configure(bg=self.bg_color)

            photo = ImageTk.PhotoImage(combined.convert('RGB'))
            label = tk.Label(heatmap_window, image=photo)
            label.image = photo
            label.pack()

        except Exception as e:
            logging.error(f"Error al generar mapa de calor: {e}", exc_info=True)
            messagebox.showerror("Error", "No se pudo generar el mapa de calor.")


if __name__ == "__main__":
    logging.info("Iniciando aplicación principal.")
    app = EMARCInferenceApp()
    app.mainloop()
    logging.info("Aplicación principal finalizada")