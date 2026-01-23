import torch

# Cargar modelo SIN AutoShape
full_model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/prueba_trafico5/weights/best.pt', source='local', autoshape=False)

# Solo exportar la parte del modelo neuronal (sin la lógica de preprocesamiento de AutoShape)
model = full_model.model

# Exportar con scripting
scripted_model = torch.jit.script(model)
scripted_model.save("best_scripted.pt")

print("✅ Modelo exportado con TorchScript sin AutoShape: best_scripted.pt")
