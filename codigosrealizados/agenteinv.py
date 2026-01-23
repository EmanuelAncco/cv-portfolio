import re
import pandas as pd
import os
import logging
import time
import requests
from deep_translator import GoogleTranslator

# --- CONFIGURACIÓN DE INGENIERÍA PESIMISTA ---
# Configuración de Logging para auditoría
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auditoria_extraccion.log"),
        logging.StreamHandler()
    ]
)


class AcademicFetcher:
    """
    Clase encargada de conectar con APIs científicas y procesar texto real.
    Usa 'Ingeniería Pesimista': asume que la red fallará, que el DOI no existe, etc.
    """

    def __init__(self):
        self.api_url = "https://api.semanticscholar.org/graph/v1/paper/"
        # Traductor instanciado una sola vez para reutilizar sesión si es posible
        self.translator = GoogleTranslator(source='auto', target='es')

    def fetch_abstract(self, doi):
        """
        Consulta la API de Semantic Scholar usando el DOI.
        Retorna: Título y Abstract en inglés.
        """
        if not doi:
            return None, None

        clean_doi = doi.replace("http://dx.doi.org/", "").replace("doi:", "").strip()

        try:
            # Petición a la API (Campos: título y abstract)
            url = f"{self.api_url}DOI:{clean_doi}?fields=title,abstract"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get('title'), data.get('abstract')
            elif response.status_code == 429:
                logging.warning("Límite de velocidad API alcanzado. Esperando 5s...")
                time.sleep(5)  # Backoff simple
                return None, None
            else:
                logging.warning(f"No encontrado en API: {clean_doi} (Status {response.status_code})")
                return None, None

        except Exception as e:
            logging.error(f"Error de conexión para DOI {clean_doi}: {str(e)}")
            return None, None

    def translate_text(self, text):
        """Traduce el texto real al español."""
        if not text: return ""
        try:
            # Limitamos a 4500 caracteres para no romper la API de traducción gratuita
            return self.translator.translate(text[:4500])
        except Exception as e:
            logging.error(f"Error de traducción: {str(e)}")
            return text  # Retorna original si falla traducción

    def summarize_structure(self, abstract_es, title_es):
        """
        Intenta extraer/resumir Estructura: Obj, Met, Res, Conc.
        Usa heurística de palabras clave sobre el texto REAL traducido.
        """
        if not abstract_es:
            return f"No se encontró abstract para '{title_es}'. Se requiere revisión manual."

        # Heurística simple para simular la estructura pedida
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', abstract_es)

        # Contenedores
        obj, met, res, conc = "", "", "", ""

        for s in sentences:
            s_lower = s.lower()
            if not obj and any(x in s_lower for x in ['objetivo', 'propósito', 'analizar', 'investigar', 'examinar']):
                obj = s
            elif not met and any(x in s_lower for x in
                                 ['metodología', 'método', 'encuesta', 'datos', 'muestra', 'análisis', 'estudio']):
                met = s
            elif not res and any(x in s_lower for x in ['resultado', 'hallazgo', 'muestra que', 'reveló', 'indica']):
                res = s
            elif not conc and any(x in s_lower for x in ['conclusión', 'concluye', 'finalmente', 'implica']):
                conc = s

        # Si la heurística falla, tomamos las primeras oraciones (fallback lógico)
        if not obj and len(sentences) > 0: obj = sentences[0]
        if not res and len(sentences) > 1: res = sentences[len(sentences) // 2]  # Mitad
        if not conc and len(sentences) > 2: conc = sentences[-1]  # Final

        # Ensamblaje final (Recortando excesos para intentar cumplir las 6 líneas)
        full_text = f"El objetivo fue {obj[:150]}... La metodología incluyó {met[:150]}... Los resultados muestran que {res[:150]}... Se concluye que {conc[:150]}."
        return full_text


class RisProcessor:
    def __init__(self, output_folder="Resultados_Referencias_Reales"):
        self.output_folder = output_folder
        self.fetcher = AcademicFetcher()
        self.ensure_directory()

    def ensure_directory(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def parse_ris(self, file_path):
        references = []
        current_ref = {}
        try:
            with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue
                if line.startswith("ER  -"):
                    if current_ref: references.append(current_ref)
                    current_ref = {}
                    continue
                match = re.match(r"^([A-Z0-9]{2})  - (.*)", line)
                if match:
                    tag, content = match.groups()
                    if tag == 'TI':
                        current_ref['title'] = content
                    elif tag == 'AU':
                        if 'authors' not in current_ref: current_ref['authors'] = []
                        current_ref['authors'].append(content)
                    elif tag == 'PY':
                        current_ref['year'] = content[:4]
                    elif tag == 'DO':
                        current_ref['doi'] = content  # Capturamos DOI
                    elif tag == 'DA' and 'year' not in current_ref:
                        y = re.search(r'\d{4}', content)
                        if y: current_ref['year'] = y.group(0)
            return references
        except Exception:
            return []

    def format_authors(self, authors_list):
        if not authors_list: return "Autor Desconocido"
        clean = [a.split(",")[0].strip() if "," in a else a.strip() for a in authors_list]
        if len(clean) == 1:
            return clean[0]
        elif len(clean) == 2:
            return f"{clean[0]} y {clean[1]}"
        else:
            return f"{clean[0]} et al."

    def process_files(self, file_paths):
        all_data = []
        total_refs = 0

        for file_path in file_paths:
            variable_ctx = "Marketing Digital" if "MARK" in file_path else "Confianza de Marca"
            refs = self.parse_ris(file_path)
            total_refs += len(refs)

            print(f"📂 Procesando archivo: {file_path} ({len(refs)} referencias detectadas)")

            for i, ref in enumerate(refs):
                doi = ref.get('doi')
                title_orig = ref.get('title', 'Sin Título')

                # Feedback visual de progreso
                print(f"   [{i + 1}/{len(refs)}] Buscando info real para: {title_orig[:30]}...")

                # 1. Fetching Real
                real_title, real_abstract_en = self.fetcher.fetch_abstract(doi)

                # 2. Traducción y Procesamiento
                if real_abstract_en:
                    abstract_es = self.fetcher.translate_text(real_abstract_en)
                    final_body = self.fetcher.summarize_structure(abstract_es, real_title or title_orig)
                    source_status = "✅ Datos Reales (API)"
                else:
                    # Fallback si no hay DOI o falla API (pero avisando)
                    final_body = "⚠️ No se encontró abstract en bases de datos abiertas. Se requiere revisión manual del DOI: " + (
                                doi or "No DOI")
                    source_status = "❌ Falló Extracción"

                # Formateo de Cita
                authors = ref.get('authors', [])
                year = ref.get('year', '2024')
                formatted_auth = self.format_authors(authors)
                citation_header = f'En el estudio de {formatted_auth} ({year}) “{title_orig}”, indica:'

                # Comentario Contextualizado (Esto sí es inferencia tuya como investigador)
                comment = (f"Este estudio sobre {variable_ctx} aporta evidencia empírica crucial para contrastar "
                           f"con los hallazgos en la cafetería Donovan, específicamente en la dimensión de {variable_ctx.lower()}.")

                all_data.append({
                    "Variable": variable_ctx,
                    "Estatus Datos": source_status,
                    "Referencia Redactada": f"{citation_header} {final_body}",
                    "Comentario (Aporte)": comment,
                    "DOI": doi,
                    "Abstract Original (Backup)": real_abstract_en if real_abstract_en else ""
                })

                # Pausa para ser amable con la API
                time.sleep(0.5)

        # Guardar Excel
        if all_data:
            df = pd.DataFrame(all_data)
            output_path = os.path.join(self.output_folder, "Matriz_Referencias_REALES.xlsx")
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            print(f"\n✅ PROCESO COMPLETADO. Archivo guardado: {output_path}")
        else:
            print("❌ No se generaron datos.")


if __name__ == "__main__":
    files = ["MARKDIGITAL.ris", "confianzaMARCA.ris"]
    valid_files = [f for f in files if os.path.exists(f)]

    if valid_files:
        print("🚀 Iniciando Agente Investigador Real...")
        print("Nota: Esto tomará unos minutos porque estamos descargando y traduciendo en tiempo real.")
        proc = RisProcessor()
        proc.process_files(valid_files)
    else:
        print("❌ Error: Coloca los archivos .ris en la misma carpeta.")