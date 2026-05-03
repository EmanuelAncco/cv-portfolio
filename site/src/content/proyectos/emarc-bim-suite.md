---
titulo: "EMARC BIM SUITE — Automatización Revit + SAP2000"
eje: "autoria"
seccion: "top12"
anio: 2025
stack: ["C#", ".NET 4.8", "Revit API", "SAP2000", "AutoCAD API"]
metricas:
  - { valor: "20+", label: "Comandos implementados" }
  - { valor: "3", label: "Plugins (SAPtoRevit, Structural, CAD-BIM)" }
enlaces: []
excerpt: "Suite profesional de automatización BIM para Revit 2026 con tres plugins: importación SAP2000/ETABS→Revit, armado estructural automático y interoperabilidad AutoCAD↔Revit vía JSON."
orden: 12
---

EMARC BIM SUITE es un conjunto de tres plugins desarrollados en C# (.NET 4.8) para Autodesk Revit 2026 que automatizan tareas de horas a minutos en proyectos de ingeniería estructural. El plugin SAPtoRevit importa vigas, columnas y losas desde SAP2000/ETABS con mapeo inteligente de secciones, creación automática de niveles y losas con pendientes variables usando el SlabShapeEditor de la API de Revit. EMARC Structural automatiza el armado de vigas con zonas de confinamiento, columnas, zapatas y losas aligeradas y bidireccionales, con detección de interferencias (clash detection).

El tercer plugin implementa interoperabilidad bidireccional AutoCAD↔Revit: exporta geometría de polilíneas DWG a JSON desde AutoCAD y los importa como familias estructurales en Revit. Las más de 20 funcionalidades adicionales incluyen exportación de schedules a Excel, creación automática de habitaciones, acabados automáticos, generación de ejes y rejillas, y tabiquería desde líneas de AutoCAD.
