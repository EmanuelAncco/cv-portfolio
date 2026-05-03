---
titulo: "Oficina Virtual AECODE — En Godot 4.4 con MCP"
eje: "autoria"
seccion: "top12"
anio: 2026
stack: ["Godot 4.4", "GDScript", "WebSocket", "Gemini API", "Pixel Art LimeZu"]
metricas:
  - { valor: "339", label: "Sprites en catálogo" }
  - { valor: "6", label: "Scripts principales de juego" }
  - { valor: "4", label: "CanvasLayers (HUD, Minimap, Bubbles, Diálogo)" }
enlaces: []
excerpt: "Oficina virtual jugable en Godot 4.4 con sprites LimeZu Modern Office, NPCs con IA (Gemini 2.0 Flash vía WebSocket), minimapa y sistema de diálogo RPG para el equipo AECODE."
orden: 11
---

Oficina Virtual AECODE es un juego pixel-art en Godot 4.4.1 que representa el espacio de trabajo del equipo AECODE como un entorno RPG 2D. El jugador controla a AECODITO y puede interactuar con NPCs agentes de IA conectados al ClawdBot Gateway en el VPS a través de WebSocket con protocolo de handshake personalizado (connect.challenge → connect → hello-ok). Cada NPC responde consultas de operaciones en tiempo real con Gemini 2.0 Flash.

La arquitectura de scripts incluye el constructor dinámico de la oficina (`office_builder.gd`, 727 líneas), el controlador del jugador (`player.gd`, 558 líneas), la máquina de estados de NPCs (`npc_agent.gd`, 325 líneas) y un visor de reportes persistente accesible con la tecla H. El catálogo de 339 sprites LimeZu Modern Office se navega desde la escena `catalogue.tscn`. El minimapa muestra la posición de todos los agentes en tiempo real.
