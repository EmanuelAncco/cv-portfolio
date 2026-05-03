---
titulo: "Gen+ Vision PDK — Dashboard YOLOv8 en vivo"
eje: "ia-campo"
seccion: "top12"
anio: 2026
stack: ["Next.js", "FastAPI", "YOLOv8", "Python", "TypeScript", "Tailwind"]
metricas:
  - { valor: "87.67%", label: "mAP@0.5 YOLOv8 v3.1" }
  - { valor: "4K", label: "Resolución cámara PDK" }
  - { valor: "30s", label: "Auto-refresh feed" }
enlaces: []
excerpt: "Dashboard Next.js con login, feed de cámara 4K en vivo desde obra PDK (Teleport.io) y análisis YOLOv8 de EPP, obreros y vehículos corriendo en backend FastAPI sobre VPS."
orden: 5
---

Dashboard de visión en obra para el proyecto PDK ASCENT de Gen+ Design, desplegado en VPS AECODE (puerto 3020). El feed de la cámara Lisual en obra PDK se obtiene directamente de la infraestructura Teleport.io (feed ID `fesaj7hljlzp`), que entrega imágenes 4K (3840×2160) con retención de 30 días. El panel tiene login con cookie httpOnly de 7 días y auto-refresh cada 30 segundos con badge LIVE/INACTIVE.

El análisis de IA corre en un backend FastAPI (puerto 8088) con YOLOv8 v3.1 (mAP@0.5 = 87.67%) que dibuja bounding boxes de EPP, obreros y vehículos sobre la imagen del feed. El selector de fuente permite elegir entre PDK Live, galería histórica o upload manual. La sección Obra PDK muestra métricas de frames almacenados, frames aceptados, intervalo real y tiempo desde el último frame.
