import sys
import os
import torch
import warnings

# Añadir YOLOv5 al sys.path
yolo_path = r"C:\Users\Emanuel\PyCharmMiscProject\yolov5"
if yolo_path not in sys.path:
    sys.path.append(yolo_path)

# Importar correctamente DetectMultiBackend
from models.common import DetectMultiBackend

# Ruta del modelo entrenado
model_path = r"C:\Users\Emanuel\PyCharmMiscProject\yolov5\runs\train\prueba_trafico5\weights\best.pt"

# Sobreescribir la clase SPPF temporalmente para suprimir warnings durante el scripting
def suppress_warnings_in_sppf_forward(model):
    """Sobreescribe temporalmente la función 'forward' en la clase SPPF para suprimir los warnings."""
    original_forward = None
    # Iterar sobre los módulos del modelo para encontrar SPPF
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module) and hasattr(module, 'forward'):
            if "SPPF" in name:
                original_forward = module.forward
                def new_forward(x):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # Suprimir warnings
                        return original_forward(x)
                module.forward = new_forward
                break
    return model

# Cargar modelo
device = torch.device('cpu')
model = DetectMultiBackend(model_path, device=device)
model.eval()

# Suprimir warnings en la función SPPF
model = suppress_warnings_in_sppf_forward(model)

# Crear input dummy
dummy_input = torch.zeros(1, 3, 640, 640)

# Exportar como TorchScript usando scripting
try:
    scripted_model = torch.jit.script(model.model)  # Usar el scripting en lugar del tracing
    scripted_model.save("best_torchscript_scripted.pt")
    print("✅ Exportación a TorchScript completada con scripting.")
except Exception as e:
    print(f"Error al exportar el modelo: {e}")
import torch
import warnings

def remove_warnings_sppf_forward(model):
    """
    Busca el módulo SPPF dentro del modelo y reemplaza su forward
    para que no use 'with warnings.catch_warnings()'.
    """
    for name, module in model.named_modules():
        # Si reconoces que es SPPF, reescribe su forward
        if "SPPF" in name:
            original_forward = module.forward

            def new_forward(x):
                # Copia lo que hacía el original forward, pero sin usar warnings.catch_warnings()
                x = module.cv1(x)
                # Ojo que, si el original forward hacía más pasos,
                # tendrás que replicar la lógica completa que hace en SPPF.
                # En YOLOv5, normalmente es algo como:
                #   y1 = module.m(x)
                #   y2 = module.m(y1)
                #   ...
                # Ajusta según lo que haga tu versión de SPPF.
                y1 = module.m(x)
                return y1

            module.forward = new_forward
    return model

# Y luego:
model = remove_warnings_sppf_forward(model)
