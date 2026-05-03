---
titulo: "Email Studio — Correos masivos + certificados PDF"
eje: "plataformas"
seccion: "top12"
anio: 2026
stack: ["Next.js", "TypeScript", "Gmail API", "pdf-lib", "Google Sheets", "Tailwind"]
metricas:
  - { valor: "2 000", label: "Emails/día (coordinador@aecode.ai)" }
  - { valor: "2.5/s", label: "Rate limit API Gmail" }
  - { valor: "7", label: "Plantillas guardadas" }
enlaces: []
excerpt: "Plataforma de correos masivos con bloques visuales, personalización IA por destinatario desde Google Sheets, y módulo de certificados PDF estampados con pdf-lib para diplomados AECODE."
orden: 9
---

Email Studio es la plataforma interna de AECODE para el diseño y envío de correos masivos desde la cuenta `coordinador@aecode.ai` (Google Workspace, 2000 emails/día, rate limit 2.5/s). La interfaz de bloques permite construir correos con header, cuerpo, firma y CTA arrastrables, con preview en tiempo real. Los destinatarios se cargan desde Google Sheets y cada correo puede ser personalizado con IA (Gemini o Claude) según nombre y contexto del contacto.

El módulo `/certificados` agrega la capacidad de emitir certificados PDF masivos: pdf-lib estampa el nombre del participante sobre una plantilla Canva y el correo se envía con el PDF adjunto vía Gmail API. Las plantillas incluyen modelos para Demo Day, AI Summit, cursos internos y comunicaciones institucionales. El historial de envíos registra estado, apertura y errores por destinatario.
