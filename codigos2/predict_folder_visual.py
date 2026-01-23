import os
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= CONFIGURACIÓN =================
model_name = "vit_base_patch16_224"
model_path = "best_vit_model.pth"
image_folder = "C:/Users/Emanuel/PyCharmMiscProject/imagenes_test"
salida_folder = "C:/Users/Emanuel/PyCharmMiscProject/salidas_visuales"
class_names = ["No agrietado", "Agrietado"]

# Crear carpeta de salida si no existe
os.makedirs(salida_folder, exist_ok=True)

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

# ================= PREDICCIÓN Y VISUALIZACIÓN =================
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, filename)
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            clase = class_names[predicted.item()]
            prob = confidence.item() * 100

            # Mostrar imagen con resultado
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.axis('off')
            label = f"{clase} ({prob:.2f}%)"
            ax.add_patch(patches.Rectangle((0, 0), image.size[0], 40, color='black', alpha=0.5))
            ax.text(10, 25, label, fontsize=14, color='white', weight='bold')

            # Guardar imagen anotada
            output_path = os.path.join(salida_folder, filename)
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"✅ {filename} → {label}")

        except Exception as e:
            print(f"❌ Error al procesar {filename}: {e}")

print("\n📁 Imágenes anotadas guardadas en:", salida_folder)
