---
titulo: "Coord Studio — Panel del AI Summit"
eje: "plataformas"
seccion: "archive"
anio: 2026
stack: ["Next.js", "NextAuth", "Drizzle ORM", "PostgreSQL", "Tailwind", "Recharts"]
metricas:
  - { valor: "14", label: "Áreas coordinadas" }
  - { valor: "coord.187-77-250-111.nip.io", label: "URL de producción" }
excerpt: "Panel Next.js 16 con login para coordinar 14 áreas del AI Construction Summit 2026, con briefs, registro de avances, dependencias entre áreas, riesgos y leaderboard."
orden: 50
---

Panel de coordinación del AI Construction Summit 2026 que reemplaza el tablero Miro del equipo con una aplicación Next.js 16 + Turbopack desplegada en el VPS AECODE (servicio systemd en `/opt/coord-studio`). Cubre 14 áreas (estrategia, sponsors, académico, tecnología, MKT, logística, entre otras) con un modelo de datos en PostgreSQL (schema `coord_studio`): briefs con porcentaje de completado, avances cronológicos, dependencias entre áreas con fecha requerida y registro de riesgos con severidad y plan B.
