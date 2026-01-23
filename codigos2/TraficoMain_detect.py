import argparse
import os
import time
import cv2
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_opt():
    parser = argparse.ArgumentParser(description="Script unificado para detección con YOLOv5")
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Fuente de video o imágenes. Puede ser 0, 1 (webcam local), una carpeta de imágenes, '
             'un archivo .mp4/.avi o una URL de cámara IP (ej. http://192.168.0.10:8080/video).'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='yolov5/runs/train/prueba_trafico5/weights/best.pt',
        help='Ruta a los pesos entrenados de YOLOv5 (por defecto best.pt).'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Umbral de confianza (0-1). Detecciones con confianza menor se descartan.'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Si se activa, muestra una ventana de OpenCV con los resultados en tiempo real.'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Si se activa, guarda los frames procesados en la carpeta runs/detect.'
    )
    return parser.parse_args()


def detect_images_or_folder(model, source, conf, show, save):
    """
    Procesa ya sea una sola imagen o un directorio con imágenes.
    Dibuja las detecciones y opcionalmente las muestra/guarda.
    """
    if os.path.isdir(source):
        # Si es una carpeta, iteramos sobre todas las imágenes compatibles
        image_files = [os.path.join(source, f) for f in os.listdir(source)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        # Si es un solo archivo
        image_files = [source]

    for img_path in image_files:
        print(f"[INFO] Procesando imagen: {img_path}")
        # Inferencia
        results = model(img_path, size=416)
        # results es una lista, tomamos la primera posición
        # YOLOv5 guarda la imagen anotada en results.render()
        # results.imgs contiene las imágenes procesadas (lista)
        # results.xyxy[0] y results.names para bounding boxes y clases

        # Renderiza la imagen con las detecciones dibujadas
        rendered_img = results.render()[0]  # .render() devuelve lista de imágenes

        # Convertimos de BGR a RGB si hace falta, ya que OpenCV usa BGR
        # result.render() ya está en BGR, en general se muestra ok con OpenCV
        if show:
            cv2.imshow("Detecciones", rendered_img)
            key = cv2.waitKey(0)  # Espera hasta que presiones una tecla
            if key == 27:  # ESC para cerrar
                break

        if save:
            save_dir = "runs/detect"
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.basename(img_path)
            out_path = os.path.join(save_dir, base_name)
            cv2.imwrite(out_path, rendered_img)
            print(f"[INFO] Resultado guardado en: {out_path}")

    if show:
        cv2.destroyAllWindows()


def detect_video_stream(model, source, conf, show, save):
    """
    Procesa video o stream de cámara (webcam o IP) frame a frame.
    Aplica detección en tiempo real y opcionalmente muestra/guarda resultados.
    """
    # Si source es un dígito tipo '0' o '1', lo convertimos a int para la webcam local
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente: {source}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin de la fuente de video o error al leer.")
            break

        # Inferencia con YOLOv5
        # results = model(frame, size=416)  # También válido
        # En versiones más nuevas se usa .predict(),
        # pero con torch hub la sintaxis es model(frame).
        results = model(frame, size=416)

        # Renderiza detecciones
        # results.render() modifica internamente results.imgs
        rendered_frame = results.render()[0]

        # Mostrar en ventana
        if show:
            cv2.imshow("Detecciones", rendered_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        # Guardar frames
        if save:
            save_dir = "runs/detect"
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(out_path, rendered_frame)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    opt = parse_opt()
    # 1. Carga el modelo YOLOv5 con tus pesos
    print(f"[INFO] Cargando modelo desde: {opt.weights}")
    model = torch.hub.load('yolov5', 'custom', path=opt.weights, source='local')

    model.conf = opt.conf  # Umbral de confianza

    source = opt.source

    # 2. Distinguimos si es imagen (o carpeta) vs. video/cámara
    # Caso: Si la extensión es de video, lo tratamos como video.
    #       Si es 0, 1 o URL, también video/stream.
    #       Sino, imagen o carpeta de imágenes.

    # Posibles extensiones de video
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

    if os.path.isdir(source):
        # Si es carpeta, procesamos como imágenes
        detect_images_or_folder(model, source, opt.conf, opt.show, opt.save)
    else:
        lower_source = source.lower()
        if lower_source.endswith(video_exts) or source.isdigit() or source.startswith(
                ("rtsp://", "http://", "https://")):
            # Es un video o cámara
            detect_video_stream(model, source, opt.conf, opt.show, opt.save)
        else:
            # Asumimos que es imagen única
            detect_images_or_folder(model, source, opt.conf, opt.show, opt.save)


if __name__ == "__main__":
    main()
