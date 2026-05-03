---
titulo: "GAIATECH FPGA — Explorer Edge-9K para FFT en hardware"
eje: "investigacion"
seccion: "top12"
anio: 2026
stack: ["Verilog", "Gowin IDE", "FPGA", "ESP32", "UART", "I2C"]
metricas:
  - { valor: "GW1NR-9", label: "FPGA Gowin" }
  - { valor: "8 640", label: "LUTs disponibles" }
  - { valor: "5", label: "Módulos Verilog para SHM" }
  - { valor: "27 MHz", label: "Clock + 2 PLLs" }
enlaces: []
excerpt: "Implementación de módulos Verilog sobre Explorer Edge-9K (Gowin GW1NR-9) para adquisición de señales sísmicas y acelerómetro en hardware, como núcleo del sistema SHM GAIATECH."
orden: 3
---

El subsistema FPGA de GAIATECH corre sobre la placa Explorer Edge-9K con Gowin GW1NR-LV9QN88PC6/I5 (8,640 LUTs, 20 DSPs, 468 Kb BSRAM, 64 Mbit SDRAM embebida). El flujo de trabajo verificado incluye el Knight Rider de prueba, control de 7-segmentos, RGB y lectura del ADC ADS1115 de 16-bit vía I2C (SCL=Pin57, SDA=Pin56). La comunicación hacia el ESP32 se realiza por UART TX en Pin 51, con éxito confirmado en sesión del 12 de abril de 2026.

Los módulos pendientes para SHM real comprenden: maestro I2C→ADS1115, acumulador FFT con DSP onboard, buffer SDRAM para ventanas de señal y un módulo UART→ESP32 de reportes. El firmware ESP32 actúa como bridge WiFi para envío de datos al dashboard Python en el VPS. El bitstream compilado vive en `impl/pnr/explorer_edge_9k.fs` y se programa por USB-C sin programador externo gracias al BL702 integrado.
