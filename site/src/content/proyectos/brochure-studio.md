---
titulo: "Brochure Studio — Editor canvas para brochures con IA"
eje: "plataformas"
seccion: "top12"
anio: 2026
stack: ["Next.js", "TypeScript", "Fabric.js", "Zustand", "Cloudinary", "Claude API"]
metricas:
  - { valor: "17", label: "Páginas editables" }
  - { valor: "200", label: "Stack máximo undo/redo" }
  - { valor: "25", label: "Tipos de bloques identificados" }
enlaces: []
excerpt: "Editor visual estilo Figma para brochures de diplomados, con sistema de capas, inspector dinámico, undo/redo de 200 pasos y análisis IA de brochures existentes vía Claude Vision."
orden: 8
---

Brochure Studio es una aplicación Next.js (puerto 3500) con editor de capas sobre HTML pixel-perfect para los brochures de diplomados AECODE. El sistema de edición funciona en una capa superpuesta sobre el diseño existente: click simple para seleccionar un bloque, doble click para editar texto inline. El panel izquierdo muestra el árbol de capas DOM con hover highlight y multi-selección, el panel derecho expone un inspector dinámico con 8 secciones acordeón (imagen, tipografía, fondo, borde, sombra, efectos, tamaño, estilo raw).

La edición de texto soporta 10 fuentes, tamaño, peso, negrita/cursiva/subrayado, color y alineación. Las imágenes se reemplazan vía Cloudinary upload con drag-and-drop o pegado desde portapapeles (Ctrl+V). El estado persiste en localStorage con migraciones automáticas de versión. La ruta `/studio` sirve las 17 páginas del brochure IA AEC Ed.03 en modo editable, con `data-field` binding para sincronización entre el panel de plantilla y el DOM.
