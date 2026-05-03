---
titulo: "Gen+ Instructor Finder"
eje: "plataformas"
seccion: "archive"
anio: 2026
stack: ["n8n", "Serper API", "Gemini", "Next.js", "VPS"]
metricas:
  - { valor: "VPS", label: "Despliegue 187.77.250.111" }
excerpt: "Panel web + workflow n8n para búsqueda de instructores AEC con Serper y Gemini, integrado a Aecodito vía comando /instructores en grupos WhatsApp Gen+."
orden: 50
---

Herramienta de búsqueda de instructores para los programas Gen+ que combina un panel web desplegado en el VPS AECODE con un sub-workflow n8n (ID Ks8nTeu4aHnghBiM). El comando `/instructores <texto>` en los grupos WhatsApp Gen+ activa el sub-workflow desde Aecodito, que ejecuta búsquedas con Serper API y consolida resultados con Gemini en un reporte de texto plano enviado de vuelta al grupo.
