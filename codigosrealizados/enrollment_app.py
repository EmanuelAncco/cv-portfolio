import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk


class EnrollmentApp:
    """
    Una aplicación simple para registrar nuevos trabajadores.
    Captura 5 imágenes de alta calidad desde la webcam y las guarda
    en una estructura de carpetas organizada.
    """

    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Crear el directorio principal de la base de datos si no existe
        self.db_dir = "database"
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)

        # Iniciar la captura de video
        self.vid = cv2.VideoCapture(0)

        # Crear un Canvas para mostrar el video
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Pedir el nombre del trabajador
        self.worker_name = simpledialog.askstring("Input",
                                                  "Ingrese el nombre del trabajador (sin espacios, ej. Emanuel_Ancco):",
                                                  parent=window)
        if not self.worker_name:
            self.window.destroy()
            return

        self.worker_dir = os.path.join(self.db_dir, self.worker_name)
        if not os.path.exists(self.worker_dir):
            os.makedirs(self.worker_dir)

        self.capture_count = 0
        self.info_label = tk.Label(window, text=f"Presiona ESPACIO para capturar. Faltan 5 fotos.",
                                   font=("Helvetica", 16))
        self.info_label.pack(pady=10)

        # Vincular la tecla Espacio al evento de captura
        self.window.bind('<space>', self.capture_snapshot)

        self.update()
        self.window.mainloop()

    def capture_snapshot(self, event=None):
        """
        Captura un fotograma, lo guarda y actualiza el contador.
        """
        if self.capture_count >= 5:
            return

        ret, frame = self.vid.read()
        if ret:
            self.capture_count += 1
            file_path = os.path.join(self.worker_dir, f"{self.capture_count}.jpg")

            # Guardar la imagen en alta calidad
            cv2.imwrite(file_path, frame)
            print(f"Imagen guardada: {file_path}")

            remaining = 5 - self.capture_count
            if remaining > 0:
                self.info_label.config(text=f"¡Captura exitosa! Faltan {remaining} fotos.")
            else:
                self.info_label.config(text="¡Registro completado!", fg="green")
                messagebox.showinfo("Éxito", f"Se han registrado 5 imágenes para {self.worker_name}.")
                self.window.after(1000, self.window.destroy)

    def update(self):
        """
        Actualiza el fotograma mostrado en el Canvas.
        """
        ret, frame = self.vid.read()
        if ret:
            # Dibujar un rectángulo guía para el rostro
            h, w, _ = frame.shape
            x1, y1 = int(w * 0.3), int(h * 0.1)
            x2, y2 = int(w * 0.7), int(h * 0.9)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Centre su rostro aqui", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__ == "__main__":
    EnrollmentApp(tk.Tk(), "EMARC VISIÓN - Aplicación de Registro")
