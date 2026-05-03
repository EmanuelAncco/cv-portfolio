---
titulo: "PachaGuard — IoT sísmico distribuido"
eje: "ia-campo"
seccion: "top12"
anio: 2025
stack: ["Verilog", "FPGA", "ESP32", "Python", "Tang Nano"]
metricas:
  - { valor: "2", label: "Subsistemas (vigilancia + sísmica)" }
  - { valor: "FPGA", label: "Hardware reconfigurable" }
enlaces:
  - { label: "Demo en YouTube", url: "https://www.youtube.com/watch?v=YuHDxx74Gqk" }
hero: "../../assets/projects/labels_correlogram.jpg"
excerpt: "Hardware reconfigurable FPGA de doble propósito: vigilancia autónoma con Edge AI y monitoreo de aceleraciones sísmicas en tiempo real para infraestructura crítica."
orden: 6
---

PachaGuard es un nodo de hardware reconfigurable basado en FPGA que combina vigilancia perimetral autónoma con capacidad Edge AI y monitoreo de aceleraciones sísmicas en tiempo real. El diseño de doble propósito permite desplegarlo como nodo de seguridad física o como sensor sísmico distribuido sin cambiar el hardware, solo reprogramando el bitstream.

El controlador principal (`pachaguard_controller.v`) implementa una FSM con UART TX/RX, buzzer de alarma y servo de actuación, validado en Tang Nano 9K y Tang Nano 1K. El subsistema ESP32 actúa como bridge WiFi para transmisión de eventos al servidor de monitoreo. La simulación sísmica demostrada en video incluye visualización de la forma de onda en display 7-segmentos en tiempo real.
