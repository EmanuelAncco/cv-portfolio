---
titulo: "Tickets Diplomado AECODE v1"
eje: "plataformas"
seccion: "archive"
anio: 2026
stack: ["n8n", "Next.js", "PostgreSQL", "Drizzle ORM", "NextAuth", "WhatsApp"]
metricas:
  - { valor: "70", label: "Nodos en workflow Aecodito" }
  - { valor: "tickets.187-77-250-111.nip.io", label: "Panel web" }
excerpt: "Sistema de soporte académico con comando /ticket en WhatsApp (Aecodito), categorización automática por IA, base de datos PostgreSQL y panel Next.js 16 para el equipo de diplomados."
orden: 50
---

Sistema de tickets integrado en Aecodito para gestionar consultas de estudiantes de diplomados AECODE, principalmente activación de licencias de software (Revit, Lumion, Autodesk). Los estudiantes escriben `/ticket <descripción>` en el grupo WhatsApp; Aecodito detecta el comando, clasifica el ticket por categoría (licencia/acceso/contenido/otro), lo persiste en PostgreSQL con un ID único y responde al estudiante con un comprobante. El equipo académico (Daniella y staff) gestiona los tickets desde el panel web con dashboard de gráficos, acciones en lote, asignación, historial y exportación CSV.
