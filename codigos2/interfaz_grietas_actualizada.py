import tkinter as tk
from tkinter import filedialog, ttk
import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ================= CONFIGURACIÓN =================
MODEL_DIR = "segformer_cracks_v1"
RES_PPI = 96  # píxeles por pulgada
CM_PER_PX = 2.54 / RES_PPI

processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR).eval()

def classify_severity(avg_width_cm):
    if avg_width_cm < 0.15:
        return "Leve", "Inspección periódica"
    elif avg_width_cm < 0.3:
        return "Moderada", "Plan de reparación"
    else:
        return "Severa", "Reparación inmediata"

def segment_crack(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=pil_img.size[::-1], mode="bilinear", align_corners=False
        )
        pred_mask = upsampled.argmax(dim=1)[0].cpu().numpy()

    mask_bin = (pred_mask == 1).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)

    mask_h, mask_w = mask_bin.shape
    area_px = np.sum(mask_bin == 255)
    damage_percent = (area_px / (mask_h * mask_w)) * 100

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_len_cm = 0
    widths_cm = []

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (w_px, h_px) = rect[1]
        largo = max(w_px, h_px) * CM_PER_PX
        ancho = min(w_px, h_px) * CM_PER_PX
        if largo > max_len_cm:
            max_len_cm = largo
        widths_cm.append(ancho)

    avg_width_cm = np.mean(widths_cm) if widths_cm else 0.0
    return mask_bin, area_px, max_len_cm, avg_width_cm, damage_percent

def analyze_image(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    np_img = np.array(pil_img)

    mask_bin, area_px, largo_cm, ancho_cm, dmg_pct = segment_crack(pil_img)

    mask_rgb = np.stack([mask_bin]*3, axis=-1)
    color_overlay = np_img.copy()
    color_overlay[mask_bin == 255] = [0, 0, 255]

    final_annot = color_overlay.copy()
    h, w = final_annot.shape[:2]
    area_cm2 = area_px * (CM_PER_PX**2)
    severity, action = classify_severity(ancho_cm)

    cv2.putText(final_annot, f"Area: {area_cm2:.2f} cm2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(final_annot, f"Largo: {largo_cm:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(final_annot, f"Ancho: {ancho_cm:.2f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(final_annot, f"Daño: {dmg_pct:.1f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(final_annot, f"Severidad: {severity}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

    orig_pil  = pil_img
    mask_pil  = Image.fromarray(mask_rgb)
    color_pil = Image.fromarray(color_overlay)
    final_pil = Image.fromarray(final_annot[:, :, ::-1])

    info = {
        "archivo": os.path.basename(img_path),
        "area_cm2": round(area_cm2, 2),
        "largo_cm": round(largo_cm, 2),
        "ancho_cm": round(ancho_cm, 2),
        "dano_pct": round(dmg_pct, 1),
        "severity": severity,
        "action": action,
    }

    return orig_pil, mask_pil, color_pil, final_pil, info

# ================= GUI APP =================

class CrackApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Análisis de Grietas - Interfaz Mejorada")
        self.geometry("1280x720")

        self.select_btn = tk.Button(self, text="Seleccionar Imagen", command=self.select_file)
        self.select_btn.pack(pady=5)

        self.img_frame = tk.Frame(self)
        self.img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(self, columns=("area", "largo", "ancho", "dano", "sev", "action"), show="headings")
        self.tree.heading("area",  text="Área (cm²)")
        self.tree.heading("largo", text="Largo (cm)")
        self.tree.heading("ancho", text="Ancho (cm)")
        self.tree.heading("dano",  text="Daño (%)")
        self.tree.heading("sev",   text="Severidad")
        self.tree.heading("action",text="Acción")
        self.tree.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree.column("area", width=100)
        self.tree.column("largo", width=100)
        self.tree.column("ancho", width=100)
        self.tree.column("dano", width=80)
        self.tree.column("sev", width=100)
        self.tree.column("action", width=200)

        self.lbl_orig  = tk.Label(self.img_frame, text="Original")
        self.lbl_mask  = tk.Label(self.img_frame, text="Máscara")
        self.lbl_color = tk.Label(self.img_frame, text="Color")
        self.lbl_final = tk.Label(self.img_frame, text="Final")

        self.lbl_orig.grid(row=0, column=0, padx=5, pady=5)
        self.lbl_mask.grid(row=0, column=1, padx=5, pady=5)
        self.lbl_color.grid(row=0, column=2, padx=5, pady=5)
        self.lbl_final.grid(row=0, column=3, padx=5, pady=5)

        # Guardar referencias a las imágenes
        self.tk_orig = None
        self.tk_mask = None
        self.tk_color = None
        self.tk_final = None

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            self.analyze_and_show(path)

    def analyze_and_show(self, img_path):
        orig_pil, mask_pil, color_pil, final_pil, info = analyze_image(img_path)

        # Insertar fila
        self.tree.insert("", tk.END, values=(
            info["area_cm2"],
            info["largo_cm"],
            info["ancho_cm"],
            info["dano_pct"],
            info["severity"],
            info["action"]
        ))

        def pil2tk(im):
            im = im.copy()
            im.thumbnail((300, 300))
            return ImageTk.PhotoImage(im)

        self.tk_orig = pil2tk(orig_pil)
        self.tk_mask = pil2tk(mask_pil)
        self.tk_color = pil2tk(color_pil)
        self.tk_final = pil2tk(final_pil)

        self.lbl_orig.configure(image=self.tk_orig)
        self.lbl_mask.configure(image=self.tk_mask)
        self.lbl_color.configure(image=self.tk_color)
        self.lbl_final.configure(image=self.tk_final)

# ================= MAIN =================

if __name__ == "__main__":
    app = CrackApp()
    app.mainloop()
