import torch

model = torch.jit.load("C:/Users/Emanuel/PyCharmMiscProject/yolov5/runs/train/prueba_trafico5/weights/best.torchscript")
model.eval()

inp = torch.zeros((1, 3, 640, 640))
out = model(inp)
print("OUT TYPE:", type(out))
print(out)
model_ts = torch.jit.load("C:/Users/Emanuel/PyCharmMiscProject/yolov5/runs/train/prueba_trafico5/weights/best.torchscript")
print(model_ts.code)
model_jit = torch.jit.load("C:/Users/Emanuel/PyCharmMiscProject/yolov5/runs/train/prueba_trafico5/weights/best.torchscript")
print(model_jit.code)  # Muestra no solo la parte lineal, sino todo el 'forward' si hay condicionales
