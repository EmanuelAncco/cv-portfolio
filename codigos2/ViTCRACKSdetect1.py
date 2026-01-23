import os
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

# ================= CONFIGURACIÓN =================
model_name = "vit_base_patch16_224"
model_path = "best_vit_model.pth"
image_folder = "C:/Users/Emanuel/PyCharmMiscProject/imagenes_test"
output_csv = "predicciones_vit.csv"
class_names = ["No agrietado", "Agrietado"]

# ================= TRANSFORMACIONES =================
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ================= CARGAR MODELO =================
model = timm.create_model(model_name, pretrained=False, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ================= PREDICCIÓN POR CARPETA =================
results = []
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, filename)
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            clase = class_names[predicted.item()]
            prob = confidence.item() * 100
            print(f"{filename} → 🧠 {clase} ({prob:.2f}%)")

            results.append({"Imagen": filename, "Diagnóstico": clase, "Confianza (%)": round(prob, 2)})

        except Exception as e:
            print(f"❌ Error al procesar {filename}: {e}")

# ================= GUARDAR CSV =================
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ Resultados guardados en {output_csv}")