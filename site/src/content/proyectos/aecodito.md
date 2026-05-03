---
titulo: "Aecodito v3.0 — Centro de operaciones AECODE"
eje: "plataformas"
seccion: "top12"
anio: 2026
stack: ["n8n", "WhatsApp", "Evolution API", "Gemini", "PostgreSQL", "pgvector"]
metricas:
  - { valor: "49", label: "Nodos n8n en workflow principal" }
  - { valor: "6", label: "Comandos ! disponibles" }
  - { valor: "13", label: "Lecciones aprendidas documentadas" }
enlaces: []
excerpt: "Agente multimodal de WhatsApp con 49 nodos n8n, comandos de operaciones (reporte, docentes, consulta), análisis de imagen y audio, RAG con pgvector y memoria conversacional en PostgreSQL."
orden: 10
---

Aecodito v3.0 es el centro de operaciones de AECODE implementado como agente de WhatsApp sobre n8n Cloud (workflow ID Ma67AAArvHx4wX15). El workflow principal tiene 49 nodos y expone dos modos: comandos `!` para operaciones estructuradas (reporte BIM, consulta inteligente, coordinación de docentes, ayuda) y conversación `@Aecodito` para diálogo libre con Gemini 2.5 Flash. Los sub-workflows !reporte (17 nodos, 3-5s) y !docentes (31 nodos, 2-3s) se invocan como workflows encadenados.

Las capacidades multimodales incluyen análisis de imágenes con Gemini Vision (planos, fotos, OCR), transcripción de audio con Whisper y RAG sobre PostgreSQL con extensión pgvector en VPS AECODE. El sistema agenda reuniones en Google Calendar con enlace Meet automático, envía correos con tres plantillas HTML (interna, externa, Demo Day) y tiene cooldown de 2 horas para reportes on-demand para evitar saturación.
