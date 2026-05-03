---
titulo: "EMARC VISIÓN — Detección YOLO para seguridad en obra"
eje: "ia-campo"
seccion: "top12"
anio: 2025
stack: ["YOLOv8", "Python", "OpenCV", "PyTorch", "Ultralytics"]
metricas:
  - { valor: "90.7%", label: "mAP@0.5" }
  - { valor: "11", label: "Clases detectadas" }
  - { valor: "94%", label: "Precisión cascos" }
video: "https://www.youtube.com/embed/K500tgJHRiY"
enlaces:
  - { label: "Demo en YouTube", url: "https://www.youtube.com/watch?v=K500tgJHRiY" }
hero: "../../assets/projects/emarc_vision_detection.png"
galeria:
  - "../../assets/projects/BoxPR_curve.png"
  - "../../assets/projects/confusion_matrix_normalized.png"
  - "../../assets/projects/results.png"
  - "../../assets/projects/prueba1.png"
  - "../../assets/projects/resultado_1.png"
  - "../../assets/projects/resultado_2.png"
  - "../../assets/projects/detecciónfacial.png"
excerpt: "Sistema de visión por computadora para verificación automática del uso de EPP en obras de construcción, con mAP@0.5 del 90.7% y 11 clases de equipamiento detectadas."
orden: 4
---

EMARC VISIÓN detecta en tiempo real si los trabajadores portan los Equipos de Protección Personal (EPP) exigidos por la Norma G.050 del Reglamento Nacional de Edificaciones del Perú. El modelo YOLOv8 entrenado identifica 11 clases: casco, chaleco, guantes, gafas, botas, mascarilla y sus variantes de ausencia. La curva precisión-recall alcanza mAP@0.5 = 90.7% con precisión específica del 94% en detección de cascos.

El pipeline incluye control de acceso por reconocimiento facial como módulo adicional. La interfaz muestra las bounding boxes en video en tiempo real con código de color según estado de cumplimiento EPP. El sistema fue validado en condiciones reales de obra con distintos ángulos de cámara y condiciones de iluminación variables.
