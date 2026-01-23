import torch
import cv2
import sys
import csv
from pathlib import Path

# Ruta al directorio raíz del proyecto YOLOv5 (ajustar si es necesario)
ROOT = Path(__file__).resolve().parent / "yolov5"
sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (non_max_suppression, scale_boxes, check_img_size, xyxy2xywh, cv2)
from utils.torch_utils import select_device

# Configuración de parámetros
weights = ROOT / "runs" / "train" / "vehiculos_8_clases_gpu4" / "weights" / "best.pt"
source = "C:/Users/Emanuel/PyCharmMiscProject/trafico1.mp4"
imgsz = (640, 640)
conf_thres = 0.25
iou_thres = 0.45
device = ''  # auto selecciona GPU si está disponible
csv_output = Path("detecciones.csv")

# Inicializa el modelo
device = select_device(device)
model = DetectMultiBackend(weights, device=device, data=ROOT / "data" / "coco128.yaml")
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)

# Carga de datos
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

# Abrir CSV para escritura
with open(csv_output, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Frame", "Clase", "Confianza", "x1", "y1", "x2", "y2"])

    # Inferencia
    frame_count = 0
    for path, im, im0s, vid_cap, s in dataset:
        frame_count += 1
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    print(label)
                    x1, y1, x2, y2 = map(int, xyxy)
                    writer.writerow([frame_count, names[int(cls)], f"{conf:.2f}", x1, y1, x2, y2])
                    cv2.rectangle(im0s, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(im0s, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Resultado', im0s)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
