import torch
from train_vit_ppe import model  # Importa tu modelo desde el script donde lo entrenaste

model_save_path = "D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/modelo_vit_ppe.pth"
torch.save(model.state_dict(), model_save_path)
print(f"✅ Modelo guardado en: {model_save_path}")
