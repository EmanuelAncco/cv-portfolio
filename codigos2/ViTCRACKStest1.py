import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

# ================= CONFIGURACIÓN =================
model_name = "vit_base_patch16_224"
model_path = "best_vit_model.pth"
class_names = ["No agrietado", "Agrietado"]

# ================= TRANSFORMACIONES =================
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ================= PREDICCIÓN =================
def predict_image(image_path):
    # Cargar imagen
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    # Cargar modelo
    model = timm.create_model(model_name, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Predicción
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    clase = class_names[predicted.item()]
    prob = confidence.item() * 100

    print(f"\n🧠 Resultado: {clase} ({prob:.2f}% de confianza)")
    return clase, prob

# ================= MAIN =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificador ViT de grietas")
    parser.add_argument("--img", type=str, required=True, help="Ruta de la imagen")
    args = parser.parse_args()

    if not os.path.isfile(args.img):
        print("❌ Imagen no encontrada. Revisa la ruta.")
    else:
        predict_image(args.img)
