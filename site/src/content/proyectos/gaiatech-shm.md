---
titulo: "GAIATECH SHM — Health monitoring estructural con IA"
eje: "investigacion"
seccion: "top12"
anio: 2025
stack: ["PyTorch", "Python", "PyTorch Geometric", "SciPy", "CUDA", "GIS"]
metricas:
  - { valor: "4", label: "Modelos entrenados" }
  - { valor: "0.973", label: "ROC-AUC (LSTM-AE)" }
  - { valor: "0.0064", label: "Best val loss" }
  - { valor: "PINN-GNN", label: "Arquitectura principal" }
enlaces: []
excerpt: "Sistema de monitoreo de salud estructural basado en IA para detección de anomalías en puentes usando acelerómetros y redes neuronales de grafos físicamente informadas."
orden: 2
---

Sistema de monitoreo de salud estructural (SHM) que combina cuatro modelos de aprendizaje profundo entrenados con datos reales del dataset Z24 Bridge Benchmark (ETH Zurich). El modelo central es un LSTM-Autoencoder con arquitectura encoder 128→64→16 bottleneck que alcanza ROC-AUC 0.973 y F1-Score 0.961 para detección de anomalías vibracionales. El grafo físicamente informado codifica la geometría 3D real del puente en lugar de conectividad completa arbitraria.

Además del LSTM-AE, el sistema incluye un predictor de socavación multi-branch (GIS Scour Predictor con 4 ramas paralelas para vibración, hidráulica, datos GIS y satelitales), un clasificador de grietas ResNet18 por transfer learning (accuracy 0.856), y un PINN Euler-Bernoulli para predicción física. Las características se extraen con Discrete Wavelet Transform (DWT) aplicada a señales de cinco sensores de acelerometría. El hardware de campo usa Explorer Edge-9K (FPGA Gowin) y un bridge ESP32 para transmisión inalámbrica a la nube.
