import os

log_path = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_no_gnn\run_no_gnn_20251027-110627\training_log_no_gnn.txt"

if os.path.exists(log_path):
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

        # Buscar mejor val loss
        import re

        pattern = r'Best Val Loss.*?: ([\d.]+)'
        match = re.search(pattern, content)

        if match:
            print(f"✅ No-GNN Best Val Loss: {match.group(1)}")
        else:
            print("⚠️ No se encontró el valor")

            # Buscar último val loss
            pattern2 = r'Val Loss: ([\d.]+)'
            matches = re.findall(pattern2, content)
            if matches:
                print(f"📊 Último Val Loss: {matches[-1]}")
else:
    print("❌ Archivo no encontrado")