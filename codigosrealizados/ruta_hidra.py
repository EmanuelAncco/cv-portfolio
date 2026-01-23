import zipfile
import xml.etree.ElementTree as ET
import logging
import os
import math
import csv
import sys
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

# --- CONFIGURACIÓN DEL USUARIO ---
NOMBRE_TIF = "terreno.tif"  # Tu archivo ALOS PALSAR
NOMBRE_KMZ = "RUTA HIDRA.kmz"  # Tu archivo de ruta
MAX_TRAMOS = 10  # Máximo número de tramos para el diseño automático
SUAVIZADO_SIGMA = 10  # Intensidad del suavizado (Mayor = Más parecido a Google Earth)

# --- AJUSTE DE CONSOLA (WINDOWS) ---
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# --- VERIFICACIÓN DE LIBRERÍAS ---
MISSING_LIBS = []
try:
    import rasterio; from rasterio.warp import transform
except ImportError:
    MISSING_LIBS.append('rasterio')
try:
    import matplotlib.pyplot as plt
except ImportError:
    MISSING_LIBS.append('matplotlib')
try:
    from scipy.ndimage import map_coordinates, gaussian_filter1d; from scipy.interpolate import interp1d
except ImportError:
    MISSING_LIBS.append('scipy')

if MISSING_LIBS:
    print(f"❌ ERROR CRÍTICO: Faltan librerías necesarias: {', '.join(MISSING_LIBS)}")
    print("   Ejecuta: pip install rasterio matplotlib scipy")
    sys.exit(1)

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CIVIL_AI")


class RDPAlgorithm:
    """
    Implementación del algoritmo Ramer-Douglas-Peucker para simplificar
    polilíneas complejas (terreno) en tramos rectos (diseño de canal).
    """

    @staticmethod
    def simplify(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
        dmax = 0.0
        index = 0
        end = len(points) - 1
        for i in range(1, end):
            d = RDPAlgorithm.perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            rec_results1 = RDPAlgorithm.simplify(points[:index + 1], epsilon)
            rec_results2 = RDPAlgorithm.simplify(points[index:], epsilon)
            return rec_results1[:-1] + rec_results2
        else:
            return [points[0], points[end]]

    @staticmethod
    def perpendicular_distance(pt, line_start, line_end):
        x0, y0 = pt
        x1, y1 = line_start
        x2, y2 = line_end
        if x1 == x2: return abs(x0 - x1)
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return abs(m * x0 - y0 + b) / math.sqrt(m ** 2 + 1)


class GeoEngine:
    def __init__(self, raster_path):
        self.raster_path = raster_path
        if not os.path.exists(raster_path):
            logger.error(f"❌ NO SE ENCUENTRA EL ARCHIVO: {raster_path}")
            logger.error("   Por favor, coloca el archivo ALOS PALSAR descomprimido en esta carpeta.")
            sys.exit(1)

        with rasterio.open(self.raster_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.bounds = src.bounds
            # Leemos todo el raster en memoria (optimizacion velocidad)
            self.band1 = src.read(1)
            logger.info(f"✅ Mapa Cargado: {raster_path} | CRS: {self.crs}")

    def get_profile(self, lats, lons):
        """Extrae perfil usando interpolación bicúbica (suave)"""
        # 1. Reproyectar a coordenadas del mapa
        if self.crs != 'EPSG:4326':
            xs, ys = transform('EPSG:4326', self.crs, lons, lats)
        else:
            xs, ys = lons, lats

        # 2. Verificar límites
        if not (self.bounds.left <= xs[0] <= self.bounds.right and self.bounds.bottom <= ys[0] <= self.bounds.top):
            logger.warning("⚠️ La ruta parece estar fuera de los límites de este TIF.")

        # 3. Convertir coordenadas mundo a coordenadas pixel
        rows, cols = rasterio.transform.rowcol(self.transform, xs, ys)

        # 4. Interpolación Bicúbica (Order 3)
        # map_coordinates espera [row_indices, col_indices]
        elevs = map_coordinates(self.band1, [rows, cols], order=3, mode='nearest')

        return elevs, xs, ys


class CanalDesigner:
    def __init__(self, kmz_path, tif_path):
        self.kmz_path = kmz_path
        self.geo = GeoEngine(tif_path)

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def process(self):
        logger.info(f"📂 Analizando: {self.kmz_path}...")

        # 1. LEER KMZ (Solo la primera ruta encontrada)
        coords = []
        with zipfile.ZipFile(self.kmz_path) as z:
            kml = z.open([x for x in z.namelist() if x.endswith('.kml')][0])
            root = ET.parse(kml).getroot()
            for e in root.iter():
                if '}' in e.tag: e.tag = e.tag.split('}', 1)[1]

            # Buscar LineString o LinearRing (Polígono)
            ls = root.find(".//LineString/coordinates")
            tipo_geom = "Ruta (LineString)"

            if ls is None:
                ls = root.find(".//LinearRing/coordinates")
                tipo_geom = "Polígono (LinearRing)"

            if ls is None:
                logger.error("❌ No se encontró ninguna geometría válida (Ruta o Polígono).")
                logger.info("   Contenido detectado en el KML:")
                for elem in root.iter():
                    if elem.tag in ['Placemark', 'Folder', 'Document']:
                        name = elem.find('name').text if elem.find('name') is not None else "Sin Nombre"
                        logger.info(f"   - {elem.tag}: {name}")
                return

            logger.info(f"✅ Geometría encontrada: {tipo_geom}")

            for line in ls.text.strip().split():
                p = line.split(',')
                coords.append((float(p[1]), float(p[0])))  # Lat, Lon

        # 2. DENSIFICACIÓN (Cada 1 metro)
        logger.info("🔄 Densificando ruta (1m) para alta resolución...")
        dense_lats, dense_lons, dists = [], [], [0.0]
        total_dist = 0.0

        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            segment_dist = self._haversine(p1[0], p1[1], p2[0], p2[1])
            steps = int(max(1, segment_dist))  # 1 punto por metro

            for j in range(steps):
                f = j / steps
                dense_lats.append(p1[0] + (p2[0] - p1[0]) * f)
                dense_lons.append(p1[1] + (p2[1] - p1[1]) * f)
                if len(dists) > 1: dists.append(total_dist + (segment_dist * f))

            total_dist += segment_dist

        # Añadir punto final
        dense_lats.append(coords[-1][0])
        dense_lons.append(coords[-1][1])
        dists.append(total_dist)

        # Ajuste de longitud de listas (seguridad)
        min_len = min(len(dense_lats), len(dists))
        dense_lats = dense_lats[:min_len]
        dense_lons = dense_lons[:min_len]
        dists = dists[:min_len]

        # 3. OBTENER ELEVACIONES (Del TIF)
        raw_z, xs_proj, ys_proj = self.geo.get_profile(dense_lats, dense_lons)

        # 4. FILTRO GAUSSIANO (Efecto Google Earth)
        logger.info(f"🎨 Aplicando Suavizado 'Google Earth Style' (Sigma={SUAVIZADO_SIGMA})...")
        smooth_z = gaussian_filter1d(raw_z, sigma=SUAVIZADO_SIGMA)

        # 5. OPTIMIZACIÓN DE TRAZO (Algoritmo RDP)
        logger.info(f"📐 Calculando trazo optimizado (Max {MAX_TRAMOS} tramos)...")

        # Preparamos puntos (Distancia, Elevación) para el algoritmo
        profile_points = list(zip(dists, smooth_z))

        # Buscamos el epsilon (tolerancia) adecuado iterativamente para obtener 7-10 tramos
        epsilon = 1.0
        simplified = []
        for _ in range(20):  # Intentar ajustar epsilon
            simplified = RDPAlgorithm.simplify(profile_points, epsilon)
            if len(simplified) <= MAX_TRAMOS + 1:  # +1 porque son puntos, no tramos
                break
            epsilon += 0.5  # Aumentar tolerancia para reducir puntos

        logger.info(f"   -> Trazo optimizado conseguido: {len(simplified) - 1} tramos (Epsilon: {epsilon})")

        # 6. GENERAR ENTREGABLES
        self.export_plot(dists, smooth_z, simplified)
        self.export_csv(dists, xs_proj, ys_proj, smooth_z)
        self.export_landxml(xs_proj, ys_proj, smooth_z)
        self.export_tramos_report(simplified)

    def export_plot(self, dists, z_terr, design_pts):
        plt.figure(figsize=(15, 8))

        # Terreno
        plt.plot(dists, z_terr, color='#2ecc71', linewidth=2, label='Terreno Natural (Suavizado)', alpha=0.8)
        plt.fill_between(dists, z_terr, min(z_terr) * 0.99, color='#2ecc71', alpha=0.1)

        # Diseño
        design_dist = [p[0] for p in design_pts]
        design_z = [p[1] for p in design_pts]
        plt.plot(design_dist, design_z, color='#e74c3c', linewidth=2.5, marker='o', label='Trazo Propuesto (Canal)')

        # Etiquetas
        for x, y in design_pts:
            plt.text(x, y + 2, f"{y:.1f}", fontsize=8, color='#c0392b', ha='center')

        plt.title(f"Diseño Automatizado de Canal: {len(design_pts) - 1} Tramos")
        plt.xlabel("Estacionamiento (m)")
        plt.ylabel("Elevación (msnm)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("Plano_Diseno_Canal.png", dpi=150)
        logger.info("✅ Gráfico generado: Plano_Diseno_Canal.png")

    def export_csv(self, dists, x, y, z):
        with open("Datos_Civil3D.csv", 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Punto", "Este", "Norte", "Elevacion", "Descripcion"])
            for i in range(len(dists)):
                # Muestreo cada 5m para no saturar CSV
                if i % 5 == 0:
                    w.writerow([i, f"{x[i]:.4f}", f"{y[i]:.4f}", f"{z[i]:.4f}", f"KM{dists[i] / 1000:.3f}"])
        logger.info("✅ CSV generado: Datos_Civil3D.csv")

    def export_landxml(self, x, y, z):
        """Genera LandXML válido para Civil 3D"""
        header = f"""<?xml version="1.0"?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2" date="{datetime.now().date()}" time="{datetime.now().time()}">
<Units><Metric areaUnit="squareMeter" linearUnit="meter" volumeUnit="cubicMeter" temperatureUnit="celsius" pressureUnit="mmHG"/></Units>
<CgPoints>"""

        footer = "</CgPoints>\n</LandXML>"

        with open("Superficie_Canal.xml", 'w') as f:
            f.write(header + "\n")
            # Exportar todos los puntos (1m) para máxima precisión de superficie
            for i in range(len(z)):
                # Civil 3D: Norte(Y) Este(X)
                f.write(f'<CgPoint name="{i + 1}">{y[i]:.6f} {x[i]:.6f} {z[i]:.4f}</CgPoint>\n')
            f.write(footer)
        logger.info("✅ LandXML generado: Superficie_Canal.xml (Arrastrar a Civil 3D)")

    def export_tramos_report(self, points):
        with open("Reporte_Tramos.txt", 'w', encoding='utf-8') as f:
            f.write("REPORTE DE DISEÑO DE CANAL (OPTIMIZADO)\n")
            f.write("=======================================\n\n")
            f.write(f"{'TRAMO':<6} | {'INICIO (m)':<10} | {'FIN (m)':<10} | {'LONG (m)':<10} | {'PENDIENTE (%)':<15}\n")
            f.write("-" * 65 + "\n")

            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                dist = p2[0] - p1[0]
                drop = p2[1] - p1[1]  # Diferencia altura
                slope = (drop / dist) * 100

                f.write(f"{i + 1:<6} | {p1[0]:<10.2f} | {p2[0]:<10.2f} | {dist:<10.2f} | {slope:<15.3f}\n")
        logger.info("✅ Reporte generado: Reporte_Tramos.txt")


if __name__ == "__main__":
    print("--- EMAIRC: Motor de Diseño 'Civil AI' v22.1 ---")
    if os.path.exists(NOMBRE_KMZ) and os.path.exists(NOMBRE_TIF):
        designer = CanalDesigner(NOMBRE_KMZ, NOMBRE_TIF)
        designer.process()
    else:
        print(f"❌ Faltan archivos. Asegúrate de tener '{NOMBRE_KMZ}' y '{NOMBRE_TIF}'")