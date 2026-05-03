---
titulo: "Paper MISM-GNN — Modelo sustituto para el puente Junín"
eje: "investigacion"
seccion: "top12"
anio: 2026
stack: ["PyTorch", "PyTorch Geometric", "OpenSeesPy", "Python", "LaTeX"]
metricas:
  - { valor: "R² 0.998", label: "Precisión GNN" }
  - { valor: "65×", label: "Menos parámetros vs FFNN" }
  - { valor: "15 figs", label: "Generadas para el manuscrito" }
enlaces:
  - { label: "Repositorio (interno)", url: "https://github.com/EmanuelAncco" }
excerpt: "Red neuronal de grafos como modelo sustituto del puente arco Junín. R² 0.998 con 65× menos parámetros que la red feed-forward de referencia. Paper-ready para Elsevier Structures."
orden: 1
---

Modelo sustituto basado en GNN para análisis dinámico estructural del puente arco de Junín, validado contra simulaciones FEM en OpenSees. La GNN respeta la topología del modelo discretizado y aprende propagación de cargas con 4 capas y 65× menos parámetros que la FFNN de referencia, manteniendo R² 0.998 sobre el conjunto de validación.

El paper acompaña 4 tablas y 15 figuras generadas íntegramente desde scripts Python reproducibles.
