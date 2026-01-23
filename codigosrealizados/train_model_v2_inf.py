# interfaz_inferencia.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import logging
import cv2
import threading
import time

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("inferencia.log", mode='w'),
        logging.StreamHandler()
    ]
)


class EMARCInferenceApp(tk.Tk):
    """
    Una aplicación de escritorio con interfaz gráfica para probar el modelo
    de detección de EPP de EMARC VISIÓN, con soporte para imágenes y video.
    """

    def __init__(self):
        super().__init__()

        self.title("Analizador de Inferencia - EMARC VISIÓN (Modelo más reciente)")
        self.geometry("1200x900")
        self.configure(bg="#2E2E2E")

        # --- VARIABLES DE ESTADO ---
        self.current_filepath = None
        self.is_video = False
        self.video_capture = None
        self.video_thread = None
        self.is_playing = False

        self.model_path = self.find_best_model()
        self.model = self.load_model()
        self.create_widgets()

    def find_best_model(self):
        """
        Encuentra automáticamente la ruta al archivo 'best.pt' de la ejecución
        de entrenamiento MÁS RECIENTE para no tener que actualizarla manualmente.
        """
        try:
            runs_dir = 'runs/detect'
            # Lista todas las carpetas de ejecución (ignorando archivos)
            all_run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            if not all_run_dirs:
                logging.warning("No se encontraron carpetas de ejecución en 'runs/detect'.")
                return None

            # Encuentra la carpeta más reciente basándose en la fecha de modificación
            latest_run_dir = max(all_run_dirs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))

            model_path = os.path.join(runs_dir, latest_run_dir, 'weights', 'best.pt')

            if os.path.exists(model_path):
                logging.info(f"Modelo 'best.pt' encontrado en la última ejecución: {model_path}")
                return model_path
            else:
                logging.warning(f"No se encontró 'best.pt' en la carpeta de ejecución más reciente: {latest_run_dir}")
                return None
        except FileNotFoundError:
            logging.error("'runs/detect' no encontrado. Asegúrate de estar en el directorio correcto.")
            return None

    def load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            try:
                logging.info(f"Cargando modelo desde: {self.model_path}")
                model = YOLO(self.model_path)
                logging.info("Modelo cargado exitosamente.")
                return model
            except Exception as e:
                logging.critical(f"Error crítico al cargar el modelo: {e}")
                messagebox.showerror("Error de Modelo", f"No se pudo cargar el modelo YOLO.\nError: {e}")
                self.destroy()
        else:
            logging.error("No se encontró ningún modelo 'best.pt' en las carpetas de ejecución.")
            messagebox.showerror("Error de Modelo",
                                 "No se encontró ningún modelo 'best.pt'.\nAsegúrate de que al menos un entrenamiento se haya completado.")
            self.destroy()

    def create_widgets(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure("TButton", padding=10, relief="flat", background="#5E5E5E", foreground="white",
                        font=('Helvetica', 12))
        style.map("TButton", background=[('active', '#7A7A7A')])
        style.configure("TLabel", background="#2E2E2E", foreground="white", font=('Helvetica', 10))
        style.configure("Header.TLabel", font=('Helvetica', 16, 'bold'))
        style.configure("H.TScale", background="#5E5E5E")

        control_frame = tk.Frame(self, bg="#3C3C3C", padx=10, pady=10)
        control_frame.pack(fill=tk.X, side=tk.TOP)

        ttk.Label(control_frame, text="Panel de Control", style="Header.TLabel", background="#3C3C3C").pack(
            side=tk.LEFT, padx=10)

        self.confidence_var = tk.DoubleVar(value=0.25)
        self.confidence_label = ttk.Label(control_frame, text=f"Confianza: {self.confidence_var.get():.0%}",
                                          background="#3C3C3C", foreground="white")
        self.confidence_label.pack(side=tk.RIGHT, padx=5)

        confidence_slider = ttk.Scale(control_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                      variable=self.confidence_var, command=self.on_slider_change)
        confidence_slider.pack(side=tk.RIGHT, padx=10, fill=tk.X, expand=True)

        self.load_button = ttk.Button(control_frame, text="Cargar Archivo", command=self.load_file)
        self.load_button.pack(side=tk.RIGHT, padx=20)

        self.video_controls_frame = tk.Frame(control_frame, bg="#3C3C3C")
        self.play_pause_button = ttk.Button(self.video_controls_frame, text="Play", command=self.toggle_play_pause)
        self.play_pause_button.pack(side=tk.LEFT)

        self.image_frame = tk.Frame(self, bg="#2E2E2E", bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.image_label = ttk.Label(self.image_frame, text="Carga una imagen o video para comenzar",
                                     style="Header.TLabel", anchor="center")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        results_frame = tk.Frame(self, bg="#3C3C3C", padx=10, pady=10)
        results_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.results_text = tk.StringVar()
        self.results_text.set("Resultados de la detección aparecerán aquí.")
        results_label = ttk.Label(results_frame, textvariable=self.results_text, style="TLabel", background="#3C3C3C")
        results_label.pack()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_file(self):
        self.stop_video()
        filepath = filedialog.askopenfilename(
            title="Selecciona un archivo de Imagen o Video",
            filetypes=[("Archivos Multimedia", "*.jpg *.jpeg *.png *.mp4 *.avi"), ("Todos los archivos", "*.*")]
        )
        if not filepath:
            logging.info("Selección de archivo cancelada.")
            return

        self.current_filepath = filepath
        _, ext = os.path.splitext(filepath)
        if ext.lower() in ['.mp4', '.avi', '.mov']:
            self.is_video = True
            self.video_controls_frame.pack(side=tk.RIGHT, padx=10)
            self.start_video()
        else:
            self.is_video = False
            self.video_controls_frame.pack_forget()
            self.run_image_inference()

    def run_image_inference(self):
        if not self.model or not self.current_filepath:
            return

        logging.info(f"Ejecutando inferencia en imagen: '{self.current_filepath}'")
        try:
            results = self.model(self.current_filepath, conf=self.confidence_var.get())
            annotated_image = results[0].plot()
            self.display_image(annotated_image)
            self.update_results_summary(results[0])
        except Exception as e:
            logging.error(f"Error durante la inferencia de imagen: {e}")
            messagebox.showerror("Error de Inferencia", f"Ocurrió un error al procesar la imagen.\nError: {e}")

    def on_slider_change(self, value):
        self.confidence_label.config(text=f"Confianza: {float(value):.0%}")
        if not self.is_video and self.current_filepath:
            self.run_image_inference()

    def start_video(self):
        if self.current_filepath:
            self.video_capture = cv2.VideoCapture(self.current_filepath)
            self.is_playing = True
            self.play_pause_button.config(text="Pause")
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()

    def stop_video(self):
        self.is_playing = False
        if self.video_thread and self.video_thread.is_alive():
            time.sleep(0.1)
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.video_controls_frame.pack_forget()

    def toggle_play_pause(self):
        self.is_playing = not self.is_playing
        self.play_pause_button.config(text="Pause" if self.is_playing else "Play")

    def video_loop(self):
        try:
            while self.current_filepath and self.video_capture and self.video_capture.isOpened():
                if self.is_playing:
                    ret, frame = self.video_capture.read()
                    if not ret:
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    results = self.model(frame, conf=self.confidence_var.get())
                    annotated_frame = results[0].plot()

                    self.display_image(annotated_frame)
                    self.update_results_summary(results[0])
                time.sleep(0.01)
        except Exception as e:
            logging.error(f"Error en el bucle de video: {e}")

    def display_image(self, cv2_image):
        annotated_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(annotated_image_rgb)

        pil_image.thumbnail((self.image_frame.winfo_width(), self.image_frame.winfo_height()))

        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=self.tk_image, text="")

    def update_results_summary(self, result):
        class_names = result.names
        detections = result.boxes.cls.tolist()

        conf_value = self.confidence_var.get()
        summary_prefix = f"Umbral de Confianza: {conf_value:.0%} | "

        if not detections:
            summary = summary_prefix + "No se detectó ningún objeto de interés."
        else:
            counts = {name: 0 for name in class_names.values()}
            for det in detections:
                counts[class_names[int(det)]] += 1

            summary_parts = [f"{name.upper()}: {count}" for name, count in counts.items() if count > 0]
            summary = summary_prefix + "Detecciones: " + " | ".join(summary_parts)

        self.results_text.set(summary)

    def on_closing(self):
        self.stop_video()
        self.destroy()


if __name__ == "__main__":
    app = EMARCInferenceApp()
    app.mainloop()
