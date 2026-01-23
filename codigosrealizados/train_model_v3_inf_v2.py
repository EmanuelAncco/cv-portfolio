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
import datetime

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("inferencia_v2.0.log", mode='w'),
        logging.StreamHandler()
    ]
)


class EMARCInferenceApp(tk.Tk):
    """
    Una aplicación de escritorio avanzada para el análisis interactivo de modelos
    de EMARC VISIÓN, con soporte para imágenes, video y cámara en vivo, con mejoras
    de robustez y control.
    """

    def __init__(self):
        """
        Inicializa la aplicación, configura la ventana principal y las variables de estado.
        """
        super().__init__()

        self.title("Laboratorio de Inferencia - EMARC VISIÓN v2.0")
        self.geometry("1800x950")
        self.configure(bg="#2E2E2E")

        # --- VARIABLES DE ESTADO ---
        self.media_path = None
        self.media_type = None  # 'image', 'video', or 'camera'
        self.original_pil_image = None
        self.processed_pil_image = None
        self.last_inference_result = None  # Almacena el objeto Results de Ultralytics
        self.model = None
        self.model_name = "Ninguno"
        self.class_colors = {}
        self.magnifier_window = None
        self.hovered_box_index = -1

        # Variables específicas para control de video/cámara
        self.video_capture = None
        self.video_thread = None
        self.is_playing = False
        self.video_paused = False  # Nuevo: Estado de pausa para video/cámara
        self.video_total_frames = 0  # Nuevo: Total de fotogramas del video
        self.video_fps = 30  # Nuevo: FPS del video o valor por defecto para cámara
        self.current_frame_for_capture = None  # Nuevo: Almacena el frame actual para capturas

        self.create_styles()
        self.create_widgets()

        # Cargar el modelo al inicio de la aplicación
        self.after(100, self.load_model)
        # Asegurarse de detener el procesamiento al cerrar la ventana
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        logging.info("Aplicación EMARCInferenceApp inicializada.")

    def create_styles(self):
        """
        Configura los estilos visuales para los widgets de Tkinter usando ttk.Style.
        """
        logging.info("Configurando estilos de la interfaz.")
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
        style.configure("Horizontal.TScale", background=control_bg_color, troughcolor="#555555", sliderrelief="flat")
        style.configure("TCheckbutton", background=control_bg_color, foreground=text_color, font=('Helvetica', 10))
        style.map("TCheckbutton", background=[('active', control_bg_color)],
                  indicatorcolor=[('selected', accent_color), ('disabled', '#555')])
        style.configure("Treeview", background="#3C3C3C", foreground=text_color, fieldbackground="#3C3C3C",
                        rowheight=25, font=('Helvetica', 10))
        style.map("Treeview", background=[('selected', accent_color)])
        style.configure("Treeview.Heading", background="#5E5E5E", foreground=text_color, font=('Helvetica', 12, 'bold'),
                        relief="flat")
        style.map("Treeview.Heading", relief=[('active', 'groove'), ('pressed', 'sunken')])
        style.configure("TMenubutton", background="#5E5E5E", foreground=text_color, font=('Helvetica', 11, 'bold'))
        style.map("TMenubutton", background=[('active', '#7A7A7A'), ('disabled', '#4A4A4A')])
        logging.info("Estilos configurados.")

    def find_best_model(self):
        """
        Busca el modelo 'best.pt' más reciente en los directorios de 'runs/detect'.
        Retorna la ruta completa del modelo y el nombre del directorio de la ejecución.
        """
        logging.info("Buscando el modelo 'best.pt' más reciente.")
        try:
            runs_dir = 'runs/detect'
            if not os.path.exists(runs_dir):
                logging.warning(f"Directorio de ejecuciones '{runs_dir}' no encontrado. No se puede cargar el modelo.")
                return None, None

            all_run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            if not all_run_dirs:
                logging.warning(f"No se encontraron subdirectorios en '{runs_dir}'.")
                return None, None

            # Ordenar por tiempo de modificación para obtener el más reciente
            latest_run_dir = max(all_run_dirs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
            model_path = os.path.join(runs_dir, latest_run_dir, 'weights', 'best.pt')

            if os.path.exists(model_path):
                logging.info(f"Modelo 'best.pt' encontrado en: {model_path}")
                return model_path, latest_run_dir
            else:
                logging.warning(f"Archivo 'best.pt' no encontrado en el directorio más reciente: {model_path}")
                return None, None
        except Exception as e:
            logging.error(f"Error al buscar el mejor modelo: {e}")
            messagebox.showerror("Error", f"No se pudo buscar el modelo:\n{e}")
            return None, None

    def load_model(self):
        """
        Carga el modelo YOLO encontrado y actualiza el estado de la UI.
        Genera colores para las clases detectables.
        """
        logging.info("Intentando cargar el modelo YOLO.")
        self.model_path, self.model_name = self.find_best_model()
        if self.model_path:
            try:
                self.model = YOLO(self.model_path)
                # Verifica si el modelo tiene la propiedad 'names' antes de generar colores
                if hasattr(self.model, 'names') and self.model.names:
                    self.class_colors = self.generate_class_colors(self.model.names)
                else:
                    logging.warning("El modelo cargado no tiene nombres de clases definidos.")
                    self.class_colors = {}
                logging.info(f"Modelo '{self.model_name}' cargado exitosamente desde: {self.model_path}")
                self.model_status_label.config(text=f"Modelo Cargado: {self.model_name}", foreground="#7FFF00")
                self.enable_controls()  # Habilitar controles si el modelo se carga con éxito
            except Exception as e:
                logging.error(f"Error al cargar el modelo YOLO desde {self.model_path}: {e}")
                messagebox.showerror("Error de Modelo", f"No se pudo cargar el modelo.\nError: {e}")
                self.disable_controls()
        else:
            logging.error("No se encontró ningún modelo 'best.pt' para cargar.")
            messagebox.showerror("Error de Modelo",
                                 "No se encontró ningún modelo 'best.pt'. Asegúrese de haber entrenado uno.")
            self.disable_controls()

    def generate_class_colors(self, class_names):
        """
        Genera un esquema de colores consistente para cada clase detectada.
        """
        logging.info("Generando colores para las clases del modelo.")
        # Utiliza un mapa de color discreto para asegurar colores distintos
        cmap = plt.get_cmap('tab10', len(class_names))  # 'tab10' es bueno para hasta 10-20 clases
        # Convierte los colores de 0-1 a 0-255 RGB
        return {name: tuple(int(c * 255) for c in cmap(i)[:3]) for i, name in class_names.items()}

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

        # Controles de carga de medios
        load_frame = ttk.Frame(self.controls_panel)
        load_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        load_frame.columnconfigure((0, 1, 2), weight=1)

        self.load_img_button = ttk.Button(load_frame, text="Cargar Imagen", command=self.load_image)
        self.load_img_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.load_vid_button = ttk.Button(load_frame, text="Cargar Video", command=self.load_video)
        self.load_vid_button.grid(row=0, column=1, sticky="ew", padx=2)

        # Selector de cámara con Menubutton
        self.camera_menu_button = ttk.Menubutton(load_frame, text="Activar Cámara")
        self.camera_menu = tk.Menu(self.camera_menu_button, tearoff=0)
        self.camera_menu_button["menu"] = self.camera_menu
        # Set the postcommand for the menu to dynamically populate it
        self.camera_menu.config(postcommand=self.populate_camera_menu)
        self.camera_menu_button.grid(row=0, column=2, sticky="ew", padx=(2, 0))

        # Sliders de pre-procesamiento
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

        # Botones de acción
        self.action_frame = ttk.Frame(self.controls_panel)
        self.action_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        self.action_frame.columnconfigure((0, 1), weight=1)

        self.rotate_button = ttk.Button(self.action_frame, text="Rotar 90°", command=self.rotate_image)
        self.rotate_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.reset_button = ttk.Button(self.action_frame, text="Resetear Cambios", command=self.reset_image)
        self.reset_button.grid(row=0, column=1, sticky="ew", padx=(2, 0))

        # Botón principal de análisis/play/pause
        self.analyze_button = ttk.Button(self.controls_panel, text="Analizar Contenido", command=self.toggle_analysis,
                                         style="Accent.TButton")
        self.analyze_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # --- Controles de Video (nuevos) ---
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
        right_panel.grid_rowconfigure(1, weight=1)
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

        # Controles de confianza y visualización
        results_footer = ttk.Frame(right_panel)
        results_footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        results_footer.columnconfigure(1, weight=1)

        self.show_conf_var = tk.BooleanVar(value=True)
        self.show_conf_check = ttk.Checkbutton(results_footer, text="Mostrar Confianza", variable=self.show_conf_var,
                                               command=self.run_inference_if_media_loaded)
        self.show_conf_check.pack(side=tk.LEFT, padx=5)

        self.confidence_var = tk.DoubleVar(value=0.25)
        self.confidence_slider = ttk.Scale(results_footer, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                           variable=self.confidence_var, command=self.on_slider_change,
                                           style="Horizontal.TScale")
        self.confidence_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        self.confidence_label = ttk.Label(results_footer, text=f"Confianza: {self.confidence_var.get():.0%}")
        self.confidence_label.pack(side=tk.RIGHT)

        # Inicialmente, deshabilitar todos los controles hasta que el modelo cargue
        self.disable_controls()
        logging.info("Widgets de la interfaz creados.")

    def find_available_cameras(self):
        """
        Detecta las cámaras disponibles en el sistema.
        Retorna una lista de índices de cámaras válidos.
        """
        logging.info("Buscando cámaras disponibles.")
        available_cameras = []
        for i in range(10):  # Try up to 10 camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()  # Try to read a frame to confirm active camera
                    if ret:
                        available_cameras.append(i)
                        logging.info(f"Cámara detectada en el índice: {i}")
                    cap.release()
            except Exception as e:
                logging.debug(f"No se pudo abrir la cámara {i}: {e}")
        if not available_cameras:
            logging.warning("No se detectaron cámaras en el sistema.")
        return available_cameras

    def populate_camera_menu(self):
        """
        Populates the camera selection menu with available camera devices.
        This function is called as a postcommand for the camera_menu.
        """
        logging.info("Populating camera selection menu.")
        self.camera_menu.delete(0, tk.END)  # Clear existing options
        cameras = self.find_available_cameras()
        if not cameras:
            self.camera_menu.add_command(label="No se encontraron cámaras", state=tk.DISABLED)
            logging.info("No cameras found to populate menu.")
            return

        for cam_idx in cameras:
            self.camera_menu.add_command(label=f"Cámara {cam_idx}", command=lambda idx=cam_idx: self.start_camera(idx))
        logging.info(f"Menu de cámaras poblado con {len(cameras)} opciones.")

    def disable_controls(self):
        """
        Deshabilita los controles de usuario cuando el modelo no está cargado.
        """
        logging.info("Deshabilitando controles.")
        controls = [
            self.load_img_button, self.load_vid_button, self.camera_menu_button,
            self.brightness_slider, self.contrast_slider,
            self.rotate_button, self.reset_button, self.analyze_button,
            self.confidence_slider, self.show_conf_check, self.capture_button
        ]
        for control in controls:
            control.config(state=tk.DISABLED)
        self.video_control_panel.grid_forget()
        logging.info("Controles deshabilitados.")

    def enable_controls(self):
        """
        Habilita los controles de usuario cuando el modelo está cargado.
        """
        logging.info("Habilitando controles.")
        controls = [
            self.load_img_button, self.load_vid_button, self.camera_menu_button,
            self.brightness_slider, self.contrast_slider,
            self.rotate_button, self.reset_button, self.analyze_button,
            self.confidence_slider, self.show_conf_check
        ]
        for control in controls:
            control.config(state=tk.NORMAL)
        logging.info("Controles habilitados.")

    def load_image(self):
        """
        Carga un archivo de imagen, lo pre-procesa y lo muestra.
        """
        logging.info("Cargando imagen...")
        self.stop_media()
        try:
            filepath = filedialog.askopenfilename(filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png *.bmp")])
            if not filepath:
                logging.info("Selección de imagen cancelada.")
                return

            if not os.path.exists(filepath):
                logging.error(f"Archivo de imagen no encontrado: {filepath}")
                messagebox.showerror("Error de Archivo", f"El archivo no existe:\n{filepath}")
                return

            self.media_path = filepath
            self.media_type = 'image'
            self.original_pil_image = Image.open(self.media_path).convert("RGB")
            self.display_pil_image(self.original_pil_image, self.original_image_label)
            self.reset_image()
            self.enable_controls_for_image()
            logging.info(f"Imagen cargada: {self.media_path}")

        except Exception as e:
            logging.error(f"Error al cargar la imagen: {e}")
            messagebox.showerror("Error de Carga", f"No se pudo cargar la imagen:\n{e}")

    def load_video(self):
        """
        Carga un archivo de video, extrae el primer fotograma para previsualización
        y configura los controles de video.
        """
        logging.info("Cargando video...")
        self.stop_media()
        try:
            filepath = filedialog.askopenfilename(filetypes=[("Archivos de Video", "*.mp4 *.avi *.mov *.mkv")])
            if not filepath:
                logging.info("Selección de video cancelada.")
                return

            if not os.path.exists(filepath):
                logging.error(f"Archivo de video no encontrado: {filepath}")
                messagebox.showerror("Error de Archivo", f"El archivo no existe:\n{filepath}")
                return

            self.media_path = filepath
            self.media_type = 'video'
            self.video_capture = cv2.VideoCapture(filepath)

            if not self.video_capture.isOpened():
                raise IOError("No se pudo abrir el archivo de video.")

            self.video_total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if self.video_fps <= 0: self.video_fps = 30

            ret, frame = self.video_capture.read()
            if ret:
                self.original_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.display_pil_image(self.original_pil_image, self.original_image_label)
                self.reset_image()
                logging.info(
                    f"Video cargado: {self.media_path} (Frames: {self.video_total_frames}, FPS: {self.video_fps})")
            else:
                logging.warning(f"No se pudo leer el primer frame del video: {self.media_path}")
                messagebox.showwarning("Advertencia", "No se pudo leer el primer fotograma del video.")

            self.enable_controls_for_video_camera()
            self.video_seek_slider.config(to=self.video_total_frames - 1)
            self.video_seek_slider.set(0)

        except Exception as e:
            logging.error(f"Error al cargar el video: {e}")
            messagebox.showerror("Error de Carga", f"No se pudo cargar el video:\n{e}")
            self.stop_media()

    def start_camera(self, camera_index=0):
        """
        Inicia la captura de video desde la cámara especificada.
        """
        logging.info(f"Iniciando cámara {camera_index}...")
        self.stop_media()
        try:
            self.media_path = f"camera_{camera_index}"
            self.media_type = 'camera'
            self.video_capture = cv2.VideoCapture(camera_index)

            if not self.video_capture.isOpened():
                raise IOError(f"No se pudo abrir la cámara {camera_index}. Puede que esté en uso o no disponible.")

            ret, frame = self.video_capture.read()
            if ret:
                self.original_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.display_pil_image(self.original_pil_image, self.original_image_label)
                self.reset_image()
                logging.info(f"Cámara {camera_index} iniciada exitosamente.")
            else:
                logging.warning(f"No se pudo leer el primer frame de la cámara {camera_index}.")
                messagebox.showwarning("Advertencia", "No se pudo leer el primer fotograma de la cámara.")

            self.enable_controls_for_video_camera()
            self.is_playing = True
            self.video_paused = False
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            self.analyze_button.config(text="Detener Cámara")
            self.video_seek_slider.config(to=1, from_=0, state=tk.DISABLED)
            self.video_seek_slider.set(0)

        except Exception as e:
            logging.error(f"Error al iniciar la cámara {camera_index}: {e}")
            messagebox.showerror("Error de Cámara", f"No se pudo iniciar la cámara {camera_index}:\n{e}")
            self.stop_media()

    def stop_media(self):
        """
        Detiene la reproducción de video/cámara y libera los recursos.
        """
        if self.is_playing:
            logging.info("Deteniendo reproducción de medios.")
            self.is_playing = False
            self.video_paused = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=0.5)
                if self.video_thread.is_alive():
                    logging.warning("El hilo de video no terminó en el tiempo esperado.")
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            self.analyze_button.config(text="Analizar Contenido")
            logging.info("Reproducción de medios detenida y recursos liberados.")

    def toggle_analysis(self):
        """
        Controla el inicio/pausa/reanudación del análisis basado en el tipo de medio.
        """
        if self.model is None:
            messagebox.showwarning("Modelo no cargado", "Por favor, cargue un modelo antes de analizar.")
            return

        if self.media_type == 'image':
            logging.info("Iniciando análisis de imagen.")
            if not self.processed_pil_image:
                messagebox.showwarning("Sin Imagen", "Por favor, cargue una imagen antes de analizar.")
                return
            self.run_inference_on_image()
        elif self.media_type in ['video', 'camera']:
            if not self.video_capture:
                messagebox.showwarning("Sin Media", "Por favor, cargue un video o inicie una cámara.")
                return

            if self.is_playing:
                self.video_paused = not self.video_paused
                if self.video_paused:
                    self.analyze_button.config(text="Reanudar Análisis")
                    logging.info(f"Análisis de {self.media_type} pausado.")
                else:
                    self.analyze_button.config(text="Pausar Análisis")
                    logging.info(f"Análisis de {self.media_type} reanudado.")
            else:
                self.is_playing = True
                self.video_paused = False
                self.analyze_button.config(text="Pausar Análisis")
                if not self.video_thread or not self.video_thread.is_alive():
                    self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
                    self.video_thread.start()
                logging.info(f"Análisis de {self.media_type} iniciado/reanudado.")

    def run_inference_on_image(self):
        """
        Ejecuta la inferencia en la imagen actualmente procesada y actualiza la UI.
        """
        logging.info("Ejecutando inferencia en la imagen.")
        if not self.model or not self.processed_pil_image:
            logging.warning("No hay modelo o imagen procesada para inferencia.")
            return

        try:
            image_np_bgr = cv2.cvtColor(np.array(self.processed_pil_image), cv2.COLOR_RGB2BGR)
            results = self.model(image_np_bgr, conf=self.confidence_var.get(), verbose=False)

            if results:
                self.last_inference_result = results[0]
                self.hovered_box_index = -1
                self.redraw_results()
                self.update_results_table()
                self.current_frame_for_capture = image_np_bgr
                logging.info("Inferencia en imagen completada y resultados actualizados.")
            else:
                self.last_inference_result = None
                self.clear_results_table()
                self.display_pil_image(self.processed_pil_image, self.result_image_label)
                logging.info("No se detectaron objetos en la imagen.")

        except Exception as e:
            logging.error(f"Error durante la inferencia en la imagen: {e}")
            messagebox.showerror("Error de Inferencia", f"No se pudo ejecutar la inferencia:\n{e}")
            self.last_inference_result = None
            self.clear_results_table()

    def video_loop(self):
        """
        Bucle principal para procesar fotogramas de video o cámara. Se ejecuta en un hilo separado.
        """
        logging.info(f"Hilo de video iniciado para {self.media_type}.")
        if self.video_capture is None and self.media_type == 'video':
            self.video_capture = cv2.VideoCapture(self.media_path)
            if not self.video_capture.isOpened():
                logging.error(f"Fallo al reabrir el video: {self.media_path}")
                self.is_playing = False
                self.video_paused = False
                self.analyze_button.config(text="Cargar Video para Analizar")
                messagebox.showerror("Error", "No se pudo reabrir el archivo de video.")
                return
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.video_seek_slider.get())

        while self.is_playing and self.video_capture and self.video_capture.isOpened():
            if self.video_paused:
                time.sleep(0.1)
                continue

            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    if self.media_type == 'video':
                        logging.info("Fin del video, reiniciando.")
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.video_seek_slider.set(0)
                        continue
                    else:
                        logging.warning("No se pudo leer el frame de la cámara. Deteniendo stream.")
                        break

                self.current_frame_for_capture = frame.copy()

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                enhancer_b = ImageEnhance.Brightness(pil_frame)
                pil_frame = enhancer_b.enhance(self.brightness_var.get())
                enhancer_c = ImageEnhance.Contrast(pil_frame)
                pil_frame = enhancer_c.enhance(self.contrast_var.get())
                frame_processed = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

                self.display_pil_image(pil_frame, self.original_image_label)

                results = self.model(frame_processed, conf=self.confidence_var.get(), verbose=False)
                self.last_inference_result = results[0]
                annotated_frame_bgr = self.draw_custom_boxes(results[0])

                self.display_pil_image(Image.fromarray(cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)),
                                       self.result_image_label)
                self.update_results_table()

                if self.media_type == 'video':
                    current_frame_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                    if current_frame_pos < self.video_total_frames:
                        self.video_seek_slider.set(current_frame_pos)

                frame_time = 1.0 / self.video_fps
                time.sleep(frame_time)

            except Exception as e:
                logging.error(f"Error durante el procesamiento de frame en video_loop: {e}")
                break

        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.is_playing = False
        self.video_paused = False
        self.analyze_button.config(text="Analizar Contenido")
        logging.info(f"Hilo de video para {self.media_type} finalizado.")

    def seek_video(self, value):
        """
        Permite al usuario saltar a un fotograma específico en el video.
        Se llama desde el slider de búsqueda de video.
        """
        if self.media_type != 'video' or not self.video_capture:
            return

        frame_idx = int(float(value))
        logging.info(f"Buscando fotograma {frame_idx} de {self.video_total_frames}.")
        try:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_capture.read()
            if ret:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                enhancer_b = ImageEnhance.Brightness(pil_frame)
                pil_frame = enhancer_b.enhance(self.brightness_var.get())
                enhancer_c = ImageEnhance.Contrast(pil_frame)
                pil_frame = enhancer_c.enhance(self.contrast_var.get())
                self.processed_pil_image = pil_frame
                self.display_pil_image(self.processed_pil_image, self.original_image_label)

                if not self.is_playing or self.video_paused:
                    self.run_inference_on_image()

            else:
                logging.warning(f"No se pudo leer el fotograma {frame_idx} al buscar en el video.")
        except Exception as e:
            logging.error(f"Error al buscar en el video: {e}")

    def capture_frame(self):
        """
        Captura el fotograma actualmente mostrado en el panel de resultados
        (con detecciones) y lo guarda en un archivo.
        """
        logging.info("Intentando capturar frame.")
        if self.last_inference_result is None or self.current_frame_for_capture is None:
            messagebox.showwarning("Sin Frame", "No hay un fotograma procesado para capturar.")
            logging.warning("No hay frame disponible para captura.")
            return

        try:
            output_dir = "capturas"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logging.info(f"Directorio de capturas creado: {output_dir}")

            annotated_img_bgr = self.draw_custom_boxes(self.last_inference_result, highlighted_index=-1)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"captura_emarc_{timestamp}.png")

            cv2.imwrite(filename, annotated_img_bgr)
            messagebox.showinfo("Captura Guardada", f"Fotograma guardado como:\n{filename}")
            logging.info(f"Fotograma capturado y guardado: {filename}")
        except Exception as e:
            logging.error(f"Error al guardar la captura del frame: {e}")
            messagebox.showerror("Error de Captura", f"No se pudo guardar el fotograma:\n{e}")

    def redraw_results(self):
        """
        Redibuja la imagen de resultados con las detecciones y el resaltado actual.
        """
        if not self.last_inference_result:
            if self.processed_pil_image:
                self.display_pil_image(self.processed_pil_image, self.result_image_label)
                self.clear_results_table()
                self.result_image_label.config(text="Presiona 'Analizar' para ver el resultado")
            return

        logging.info(f"Redibujando resultados con highlight_index: {self.hovered_box_index}.")
        annotated_image_bgr = self.draw_custom_boxes(self.last_inference_result, self.hovered_box_index)
        annotated_pil_image = Image.fromarray(cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB))
        self.display_pil_image(annotated_pil_image, self.result_image_label)

    def update_results_table(self):
        """
        Actualiza la tabla de Treeview con las detecciones, aplicando Non-Maximum Suppression (NMS).
        """
        logging.info("Actualizando tabla de resultados con NMS.")
        self.clear_results_table()
        if not self.last_inference_result or not self.last_inference_result.boxes:
            logging.info("No hay resultados de inferencia para mostrar en la tabla.")
            return

        # Extract all raw detections
        all_boxes = self.last_inference_result.boxes.xyxy.cpu().numpy().tolist()
        all_confidences = self.last_inference_result.boxes.conf.cpu().numpy().tolist()
        all_class_ids = self.last_inference_result.boxes.cls.cpu().numpy().tolist()

        # Filter detections based on confidence threshold first
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        original_indices_map = []  # To map back to original detection index for highlighting

        for i, conf in enumerate(all_confidences):
            if conf >= self.confidence_var.get():
                filtered_boxes.append(all_boxes[i])
                filtered_confidences.append(conf)
                filtered_class_ids.append(all_class_ids[i])
                original_indices_map.append(i)  # Store original index

        detections_to_display = []
        if filtered_boxes:  # Only proceed if there are boxes after confidence filtering
            try:
                # Convert filtered_boxes to [x, y, width, height] format for NMSBoxes
                nms_input_boxes = []
                for box_coords in filtered_boxes:
                    x1, y1, x2, y2 = box_coords
                    nms_input_boxes.append([x1, y1, x2 - x1, y2 - y1])

                # Apply NMS
                # The NMSBoxes function returns indices relative to the input arrays (filtered_boxes, filtered_confidences)
                indices_after_nms = cv2.dnn.NMSBoxes(nms_input_boxes, filtered_confidences, self.confidence_var.get(),
                                                     iou_threshold=0.4)
                indices_after_nms = indices_after_nms.flatten().tolist() if len(indices_after_nms) > 0 else []

                for nms_idx in indices_after_nms:
                    original_det_idx = original_indices_map[nms_idx]  # Get the original index
                    class_name = self.model.names[int(filtered_class_ids[nms_idx])]
                    confidence = float(filtered_confidences[nms_idx])
                    detections_to_display.append({'id': original_det_idx, 'class': class_name, 'conf': confidence})

            except Exception as e:
                logging.error(
                    f"Error al aplicar NMS en update_results_table: {e}. Mostrando todos los resultados filtrados por confianza.")
                # Fallback: if NMS fails, display all boxes that passed the initial confidence filter
                for i, box_coords in enumerate(filtered_boxes):
                    original_det_idx = original_indices_map[i]
                    class_name = self.model.names[int(filtered_class_ids[i])]
                    confidence = float(filtered_confidences[i])
                    detections_to_display.append({'id': original_det_idx, 'class': class_name, 'conf': confidence})
        else:
            logging.info("No detections passed the confidence threshold for NMS.")

        detections_to_display.sort(key=lambda x: x['conf'], reverse=True)

        for det in detections_to_display:
            self.tree.insert("", "end", iid=det['id'], values=(det['class'], f"{det['conf'] * 100:.1f}%"))
        logging.info(f"Tabla de resultados actualizada con {len(detections_to_display)} detecciones.")

    def show_magnifier(self, event):
        """
        Muestra una ventana de lupa que amplía la sección de la imagen bajo el cursor.
        """
        if not self.last_inference_result: return
        logging.debug("Mostrando lupa.")

        if hasattr(self.result_image_label,
                   'image_original_size_for_magnifier') and self.result_image_label.image_original_size_for_magnifier:
            annotated_image_pil = self.result_image_label.image_original_size_for_magnifier
        else:
            logging.warning("Imagen original para lupa no encontrada, regenerando.")
            annotated_image_bgr = self.draw_custom_boxes(self.last_inference_result, self.hovered_box_index)
            annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB))

        label_x, label_y = event.x, event.y

        label_width = self.result_image_label.winfo_width()
        label_height = self.result_image_label.winfo_height()

        img_width, img_height = annotated_image_pil.size
        scale_x = img_width / (label_width - 10) if (label_width - 10) > 0 else 1
        scale_y = img_height / (label_height - 10) if (label_height - 10) > 0 else 1
        actual_scale = max(scale_x, scale_y)

        img_x = int(label_x * actual_scale)
        img_y = int(label_y * actual_scale)

        magnifier_size = 200
        zoom_factor = 3

        half_mag = magnifier_size // (2 * zoom_factor)
        left = max(0, img_x - half_mag)
        top = max(0, img_y - half_mag)
        right = min(img_width, img_x + half_mag)
        bottom = min(img_height, img_y + half_mag)

        if right <= left or bottom <= top:
            logging.debug("Región de lupa inválida.")
            return

        crop_box = (left, top, right, bottom)
        magnified_region = annotated_image_pil.crop(crop_box)
        magnified_region = magnified_region.resize((magnifier_size, magnifier_size), Image.Resampling.LANCZOS)

        if not self.magnifier_window:
            self.magnifier_window = tk.Toplevel(self)
            self.magnifier_window.overrideredirect(True)
            self.magnifier_window.attributes("-topmost", True)
            self.magnifier_label = ttk.Label(self.magnifier_window, background="#000000")
            self.magnifier_label.pack()

        mouse_x, mouse_y = self.winfo_pointerx(), self.winfo_pointery()
        self.magnifier_window.geometry(f"{magnifier_size}x{magnifier_size}+{mouse_x + 30}+{mouse_y + 30}")

        tk_mag_image = ImageTk.PhotoImage(magnified_region)
        self.magnifier_label.config(image=tk_mag_image)
        self.magnifier_label.image = tk_mag_image
        logging.debug("Lupa actualizada.")

    def hide_magnifier(self, event=None):
        """
        Oculta y destruye la ventana de lupa.
        """
        if self.magnifier_window:
            logging.debug("Ocultando lupa.")
            self.magnifier_window.destroy()
            self.magnifier_window = None
            self.magnifier_label = None

    def on_closing(self):
        """
        Maneja el evento de cierre de la ventana, asegurando que los recursos se liberen.
        """
        logging.info("Cerrando aplicación. Deteniendo media y liberando recursos.")
        self.stop_media()
        self.destroy()
        logging.info("Aplicación cerrada exitosamente.")

    def apply_preprocessing(self, event=None):
        """
        Aplica ajustes de brillo y contraste a la imagen original
        y actualiza la pre-visualización.
        """
        if not self.original_pil_image:
            logging.debug("apply_preprocessing: No hay imagen original para procesar.")
            return

        logging.info("Aplicando pre-procesamiento (brillo/contraste).")
        try:
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
            logging.info("Pre-procesamiento aplicado.")
        except Exception as e:
            logging.error(f"Error al aplicar pre-procesamiento: {e}")
            messagebox.showerror("Error de Procesamiento", f"No se pudo aplicar el pre-procesamiento:\n{e}")

    def rotate_image(self):
        """
        Rota la imagen original 90 grados en sentido anti-horario y aplica el pre-procesamiento.
        """
        if not self.original_pil_image:
            messagebox.showwarning("Sin Imagen", "Cargue una imagen primero para rotar.")
            return

        logging.info("Rotando imagen 90 grados.")
        try:
            self.original_pil_image = self.original_pil_image.rotate(90, expand=True)
            self.apply_preprocessing()
            logging.info("Imagen rotada.")
        except Exception as e:
            logging.error(f"Error al rotar imagen: {e}")
            messagebox.showerror("Error de Rotación", f"No se pudo rotar la imagen:\n{e}")

    def reset_image(self):
        """
        Restablece la imagen a su estado original sin pre-procesamiento.
        """
        if not self.original_pil_image:
            return

        logging.info("Reseteando imagen y pre-procesamiento.")
        try:
            self.brightness_var.set(1.0)
            self.contrast_var.set(1.0)
            self.processed_pil_image = self.original_pil_image.copy()
            self.display_pil_image(self.processed_pil_image, self.original_image_label)
            self.result_image_label.config(image='', text="El resultado aparecerá aquí...")
            self.clear_results_table()
            self.last_inference_result = None
            logging.info("Imagen y ajustes reseteados.")
        except Exception as e:
            logging.error(f"Error al resetear imagen: {e}")
            messagebox.showerror("Error", f"No se pudo resetear la imagen:\n{e}")

    def run_inference_if_media_loaded(self):
        """
        Ejecuta la inferencia solo si se ha cargado una imagen y no es un stream de video/cámara.
        Esto es útil para los cambios de slider o checkbox.
        """
        if self.media_type == 'image':
            logging.info("Realizando inferencia debido a cambio de control (solo imagen).")
            self.run_inference_on_image()
        elif self.media_type in ['video', 'camera'] and self.is_playing and not self.video_paused:
            logging.info("Cambio de control detectado. Se aplicará en el siguiente fotograma del stream.")

    def on_slider_change(self, value):
        """
        Actualiza el texto del slider de confianza y, si es una imagen, re-ejecuta la inferencia.
        """
        self.confidence_label.config(text=f"Confianza: {float(value):.0%}")
        self.run_inference_if_media_loaded()

    def draw_custom_boxes(self, result, highlighted_index=-1):
        """
        Dibuja los cuadros delimitadores y las etiquetas en la imagen.
        Permite resaltar un cuadro específico.
        """
        img_np_bgr = result.orig_img.copy()
        pil_img = Image.fromarray(cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            logging.warning("Fuente 'arial.ttf' no encontrada, usando fuente por defecto.")
            font = ImageFont.load_default()

        # Collect all detections that pass the confidence threshold for NMS and drawing
        all_detections_info = []  # Stores (original_idx, box_coords, confidence, class_id)
        for i, box in enumerate(result.boxes):
            confidence = float(box.conf)
            if confidence >= self.confidence_var.get():
                all_detections_info.append((i, box.xyxy[0].tolist(), confidence, int(box.cls)))

        # Prepare for NMS
        nms_input_boxes = []
        nms_input_confidences = []
        original_idx_map_for_nms = []  # To map NMS output indices back to original detection indices

        for original_idx, box_coords, confidence, class_id in all_detections_info:
            x1, y1, x2, y2 = box_coords
            nms_input_boxes.append([x1, y1, x2 - x1, y2 - y1])
            nms_input_confidences.append(confidence)
            original_idx_map_for_nms.append(original_idx)

        indices_to_draw_final = []
        if nms_input_boxes:
            try:
                # Apply NMS. The output indices are relative to nms_input_boxes/confidences.
                indices_np = cv2.dnn.NMSBoxes(nms_input_boxes, nms_input_confidences, self.confidence_var.get(),
                                              iou_threshold=0.4)
                indices_to_draw_final = indices_np.flatten().tolist() if len(indices_np) > 0 else []
            except Exception as e:
                logging.error(f"Error in NMS during drawing: {e}. Drawing all eligible boxes.")
                indices_to_draw_final = list(range(len(nms_input_boxes)))

        # Now draw only the boxes that survived NMS
        for nms_result_idx in indices_to_draw_final:
            # Get the original index of the detection from our map
            original_det_idx = original_idx_map_for_nms[nms_result_idx]

            # Retrieve the full box information using the original index from the result object
            box = result.boxes[original_det_idx]
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            confidence = float(box.conf)
            coords = box.xyxy[0].tolist()
            color = self.class_colors.get(class_name, (255, 255, 255))

            # Highlight the box if its original index matches the hovered_box_index from the Treeview
            width = 5 if original_det_idx == highlighted_index else 3

            x1, y1, x2, y2 = coords
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

            if self.show_conf_var.get():
                text = f"{class_name}: {confidence:.2f}"
                text_bbox = draw.textbbox((x1, y1), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                text_x = x1
                text_y = y1 - text_height - 5
                if text_y < 0:
                    text_y = y1 + 5

                draw.rectangle([text_x, text_y, text_x + text_width + 5, text_y + text_height + 5], fill=color)
                draw.text((text_x + 2, text_y + 2), text, fill=(0, 0, 0), font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def display_pil_image(self, pil_image, label_widget):
        """
        Muestra una imagen PIL en un widget Label de Tkinter, ajustando su tamaño.
        También guarda una referencia a la imagen de tamaño original para la lupa.
        """
        if pil_image is None:
            label_widget.config(image='', text="Cargue un archivo...")
            label_widget.image = None
            label_widget.image_original_size_for_magnifier = None
            return

        try:
            w = label_widget.winfo_width()
            h = label_widget.winfo_height()
            if w == 1 or h == 1:
                w, h = label_widget.master.winfo_width(), label_widget.master.winfo_height()
                if w < 100 or h < 100:
                    w, h = 800, 600
                logging.debug(f"Widget aún no renderizado, usando tamaño estimado: {w}x{h}")
        except tk.TclError:
            w, h = 800, 600
            logging.debug(f"Error al obtener tamaño del widget, usando tamaño por defecto: {w}x{h}")

        display_image = pil_image.copy()
        display_image.thumbnail((w - 10, h - 10), Image.Resampling.LANCZOS)

        tk_image = ImageTk.PhotoImage(display_image)
        label_widget.config(image=tk_image)
        label_widget.image = tk_image

        label_widget.image_original_size_for_magnifier = pil_image
        logging.debug(f"Imagen mostrada en label '{label_widget.winfo_name()}' con tamaño {display_image.size}.")

    def clear_results_table(self):
        """
        Limpia todas las entradas de la tabla de resultados.
        """
        for item in self.tree.get_children():
            self.tree.delete(item)

    def on_tree_hover(self, event):
        """
        Maneja el evento de pasar el ratón sobre una fila en la tabla de resultados,
        resaltando la caja correspondiente en la imagen.
        """
        if self.media_type != 'image' and (self.is_playing and not self.video_paused):
            return

        item_id = self.tree.identify_row(event.y)
        if item_id:
            box_index = int(item_id)
            if box_index != self.hovered_box_index:
                self.hovered_box_index = box_index
                self.redraw_results()
                logging.debug(f"Resaltando caja con índice: {box_index}")
        else:
            self.on_tree_leave(event)

    def on_tree_leave(self, event):
        """
        Maneja el evento de que el ratón sale de la tabla,
        quitando el resaltado de la caja.
        """
        if self.hovered_box_index != -1:
            self.hovered_box_index = -1
            self.redraw_results()
            logging.debug("Quita resaltado de caja.")

    def enable_controls_for_image(self):
        """Habilita controles específicos para el modo de imagen."""
        self.enable_controls()
        self.analyze_button.config(text="Analizar Imagen", state=tk.NORMAL)
        self.video_control_panel.grid_forget()
        self.brightness_slider.config(state=tk.NORMAL)
        self.contrast_slider.config(state=tk.NORMAL)
        self.rotate_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)

    def enable_controls_for_video_camera(self):
        """Habilita controles específicos para el modo de video/cámara."""
        self.enable_controls()
        self.analyze_button.config(text="Pausar Análisis" if self.is_playing else "Iniciar Análisis", state=tk.NORMAL)
        self.video_control_panel.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.capture_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    logging.info("Iniciando aplicación principal.")
    app = EMARCInferenceApp()
    app.mainloop()
    logging.info("Aplicación principal finalizada.")
