---
titulo: "AECODE FINDER v2 — Panel de empresas con IA + Notion"
eje: "plataformas"
seccion: "top12"
anio: 2026
stack: ["Next.js", "TypeScript", "Serper API", "Gemini", "Notion API", "Tailwind"]
metricas:
  - { valor: "99", label: "Empresas en BD Notion" }
  - { valor: "2", label: "Deduplicación segura" }
  - { valor: ":3002", label: "Puerto local" }
enlaces: []
excerpt: "Panel Next.js con búsqueda de empresas AEC vía Serper + Gemini, sincronización en vivo con base de datos Notion y deduplicación automática para análisis comercial."
orden: 7
---

AECODE FINDER v2 es un panel web en Next.js (puerto 3002) que permite al equipo AECODE buscar y gestionar empresas del sector AEC (Arquitectura, Ingeniería, Construcción) para prospección comercial. Las búsquedas se realizan con Serper API + Gemini directamente desde el panel, sin intermediarios. Los resultados se sincronizan en vivo con la base de datos "BD - Empresas" en Notion, que contiene 99 filas al momento del despliegue.

El sistema implementa deduplicación segura para evitar registros repetidos al sincronizar múltiples búsquedas sobre el mismo dominio o empresa. El panel expone filtros por sector, ciudad y estado de contacto, facilitando el seguimiento de leads para los diplomados AECODE.
