import tkinter as tk
from tkinter import filedialog, ttk
import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ====================================================
# 1) Configuración e inicialización
# ====================================================
MODEL_DIR = "segformer_cracks_v1"
RES_PPI = 96  # píxeles por pulgada (asumiendo)
CM_PER_PX = 2.54 / RES_PPI

# Cargamos el modelo y el processor
processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).eval()

# Definimos funcion de severidad
def classify_severity(avg_width_cm):
    if avg_width_cm < 0.15:
        return "Leve", "Inspeccion periódica"
    elif avg_width_cm < 0.3:
        return "Moderada", "Plan de reparacion"
    else:
        return "Severa", "Reparacion inmediata"

# ====================================================
# 2) Función de segmentación
# ====================================================
def segment_crack(pil_img):
    """
    - Recibe PIL Image
    - Retorna:
       mask_bin (np.array): máscara binaria en 0-255
       area_px (float)
       max_len_cm (float)
       avg_width_cm (float)
       damage_percent (float)
    - También retorna imágenes intermedias para la GUI
    """
    # Convertir a tensores
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=pil_img.size[::-1], mode="bilinear", align_corners=False
        )
        pred_mask = upsampled.argmax(dim=1)[0].cpu().numpy()

    # Binarizar
    mask_bin = (pred_mask == 1).astype(np.uint8) * 255

    # Limpieza morfologica
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)

    # Calcular contornos
    mask_h, mask_w = mask_bin.shape
    area_px = np.sum(mask_bin == 255)
    damage_percent = (area_px / (mask_h * mask_w)) * 100

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_len_cm = 0
    widths_cm = []

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (w_px, h_px) = rect[1]
        # Tomar dimension mayor como largo
        largo = max(w_px, h_px) * CM_PER_PX
        ancho = min(w_px, h_px) * CM_PER_PX
        if largo > max_len_cm:
            max_len_cm = largo
        widths_cm.append(ancho)

    avg_width_cm = np.mean(widths_cm) if widths_cm else 0.0

    return mask_bin, area_px, max_len_cm, avg_width_cm, damage_percent

# ====================================================
# 3) Función de pipeline
# ====================================================
def analyze_image(img_path):
    """
    Analiza la imagen y retorna:
     - original (PIL),
     - mask_bin (PIL),
     - color_masked (PIL),
     - final_annot (PIL),
     - diccionario con datos
    """
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img)

    # Segmentacion
    mask_bin, area_px, largo_cm, ancho_cm, dmg_pct = segment_crack(pil_img)

    # Convertir la mascara a RGB para mostrar
    mask_rgb = np.stack([mask_bin]*3, axis=-1)

    # Pintar color
    color_overlay = np_img.copy()
    color_overlay[mask_bin == 255] = [0,0,255]  # rojo

    # Imagen final con texto
    final_annot = color_overlay.copy()
    h, w = final_annot.shape[:2]

    area_cm2 = area_px*(CM_PER_PX**2)
    severity, action = classify_severity(ancho_cm)

    # Poner textos
    cv2.putText(final_annot, f"Area: {area_cm2:.2f} cm2", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(final_annot, f"Largo: {largo_cm:.2f} cm", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(final_annot, f"Ancho: {ancho_cm:.2f} cm", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(final_annot, f"Dano: {dmg_pct:.1f}%", (10,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(final_annot, f"Severidad: {severity}", (10,150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

    # Convertir a PIL
    orig_pil   = pil_img
    mask_pil   = Image.fromarray(mask_rgb)
    color_pil  = Image.fromarray(color_overlay)
    final_pil  = Image.fromarray(final_annot[:,:,::-1]) # BGR->RGB

    # Diccionario de datos
    info = {
        "archivo": os.path.basename(img_path),
        "area_cm2": round(area_cm2,2),
        "largo_cm": round(largo_cm,2),
        "ancho_cm": round(ancho_cm,2),
        "dano_pct": round(dmg_pct,1),
        "severity": severity,
        "action": action,
    }

    return orig_pil, mask_pil, color_pil, final_pil, info

# ====================================================
# 4) Interfaz con Tkinter
# ====================================================
import tkinter as tk
from tkinter import filedialog, ttk

class CrackApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Análisis de Grietas - Ejemplo")
        self.geometry("1280x720")

        # Boton para seleccionar archivos
        self.select_btn = tk.Button(self, text="Seleccionar Imagen", command=self.select_file)
        self.select_btn.pack(pady=5)

        # Frame para imágenes
        self.img_frame = tk.Frame(self)
        self.img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Info en Treeview
        self.tree = ttk.Treeview(self, columns=("area","largo","ancho","dano","sev","action"), show="headings")
        self.tree.heading("area",  text="Área (cm²)")
        self.tree.heading("largo", text="Largo (cm)")
        self.tree.heading("ancho", text="Ancho (cm)")
        self.tree.heading("dano",  text="Daño (%)")
        self.tree.heading("sev",   text="Severidad")
        self.tree.heading("action",text="Acción")
        self.tree.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.column("area",  width=100)
        self.tree.column("largo", width=100)
        self.tree.column("ancho", width=100)
        self.tree.column("dano",  width=80)
        self.tree.column("sev",   width=100)
        self.tree.column("action",width=200)

        # 4 labels para 4 imágenes
        self.lbl_orig   = tk.Label(self.img_frame, text="Original")
        self.lbl_mask   = tk.Label(self.img_frame, text="Máscara")
        self.lbl_color  = tk.Label(self.img_frame, text="Color")
        self.lbl_final  = tk.Label(self.img_frame, text="Final")

        self.lbl_orig.grid(row=0, column=0, padx=5, pady=5)
        self.lbl_mask.grid(row=0, column=1, padx=5, pady=5)
        self.lbl_color.grid(row=0, column=2, padx=5, pady=5)
        self.lbl_final.grid(row=0, column=3, padx=5, pady=5)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.jpeg")])
        if path:
            self.analyze_and_show(path)

    def analyze_and_show(self, img_path):
        # Analizar
        orig_pil, mask_pil, color_pil, final_pil, info = analyze_image(img_path)

        # Presencia de grieta
        presence = "Sí" if info["area_cm2"] > 0 else "No"

        # Insertar fila en tree
        self.tree.insert("", tk.END, values=(
            info["area_cm2"],
            info["largo_cm"],
            info["ancho_cm"],
            info["dano_pct"],
            info["severity"],
            info["action"]
        ))

        # Convertir PIL -> ImageTk
        # Redimensionar para no saturar la GUI
        wimg = 300
        def pil2tk(im):
            im = im.copy()
            im.thumbnail((wimg, wimg))
            return ImageTk.PhotoImage(im)

        orig_tk   = pil2tk(orig_pil)
        mask_tk   = pil2tk(mask_pil)
        color_tk  = pil2tk(color_pil)
        final_tk  = pil2tk(final_pil)

        # Actualizar labels
        self.lbl_orig.configure(image=orig_tk)
        self.lbl_orig.image = orig_tk

        self.lbl_mask.configure(image=mask_tk)
        self.lbl_mask.image = mask_tk

        self.lbl_color.configure(image=color_tk)
        self.lbl_color.image = color_tk

        self.lbl_final.configure(image=final_tk)
        self.lbl_final.image = final_tk


def analyze_image(img_path):
    return analyze_image_pipeline(img_path)

# Reemplazamos con la funcion principal de pipeline
def analyze_image_pipeline(img_path):
    # 1) Llamar a la funcion analyze_image
    orig, mask, color, final, info = analyze_image_internal(img_path)
    return orig, mask, color, final, info

def analyze_image_internal(img_path):
    orig_pil, mask_pil, color_pil, final_pil, data = None, None, None, None, {}
    # Llamamos a la pipeline anterior
    orig_pil, mask_pil, color_pil, final_pil, info = pipeline_analysis(img_path)
    return orig_pil, mask_pil, color_pil, final_pil, info



def pipeline_analysis(img_path):
    # pipeline
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img)

    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=pil_img.size[::-1], mode="bilinear", align_corners=False
        )
        pred_mask = upsampled.argmax(dim=1)[0].cpu().numpy()

    # Binarizar
    mask_bin = (pred_mask==1).astype(np.uint8)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)

    # 3 vistas
    mask_rgb = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2RGB)
    color_overlay = np_img.copy()
    color_overlay[mask_bin==255] = [0,0,255]

    # final con anotaciones
    final_annot = color_overlay.copy()

    # Calculos
    h, w = final_annot.shape[:2]
    area_px = np.sum(mask_bin==255)
    area_cm2 = area_px*(CM_PER_PX**2)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_len_cm=0
    widths=[]
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        w_px,h_px = rect[1]
        length = max(w_px,h_px)*CM_PER_PX
        width = min(w_px,h_px)*CM_PER_PX
        if length>max_len_cm: max_len_cm=length
        widths.append(width)
    avg_w = np.mean(widths) if widths else 0
    damage_pct = (area_px/(h*w))*100

    # Escribimos en la imagen
    cv2.putText(final_annot, f"Area: {area_cm2:.2f} cm2",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    cv2.putText(final_annot, f"Largo: {max_len_cm:.2f} cm",(10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    cv2.putText(final_annot, f"Ancho: {avg_w:.2f} cm",(10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    cv2.putText(final_annot, f"Dano: {damage_pct:.1f}%",(10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)

    info = {
      "archivo": os.path.basename(img_path),
      "area_cm2": area_cm2,
      "largo_cm":max_len_cm,
      "ancho_cm":avg_w,
      "dano_pct":damage_pct,
      "severity":"?"
    }

    # Convertir a PIL
    orig_pil  = Image.fromarray(np_img)
    mask_pil  = Image.fromarray(mask_rgb)
    color_pil = Image.fromarray(color_overlay)
    final_pil = Image.fromarray(final_annot[:,:,::-1]) # BGR->RGB
    return orig_pil, mask_pil, color_pil, final_pil, info


# ============================================================
# 5) main
# ============================================================
if __name__ == "__main__":
    app = CrackApp()
    app.mainloop()
