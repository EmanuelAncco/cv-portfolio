import os
import re
import logging
import pandas as pd
from docx import Document
from difflib import SequenceMatcher
from datetime import datetime
import unicodedata

# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================
CONFIG = {
    "base_path": r"C:\Users\Emanuel\Downloads",
    "word_filename": "7. Especificaciones Técnicas - copia (2).docx",
    "budget_filename": "presupuesto total.xlsx",
    "output_folder": "Resultados_Auditoria",
    "word_style_name": "Heading 3",
    "min_similarity_ratio": 0.85,
}

# ==========================================
# CONFIGURACIÓN DE LOGGING
# ==========================================
results_dir = os.path.join(CONFIG["base_path"], CONFIG["output_folder"])
os.makedirs(results_dir, exist_ok=True)

log_filename = os.path.join(results_dir, f"auditoria_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BudgetAuditor:
    def __init__(self, config):
        self.config = config
        self.df_budget = pd.DataFrame()
        self.word_items = []
        self.report = []
        # Diccionario maestro: {Codigo_Normalizado: {Desc: "Texto", Codigo_Original: "01.01"}}
        self.budget_map = {}
        # Diccionario flexible inverso: {Sufijo_Normalizado: [Codigos_Excel_Completos]}
        self.budget_flexible_map = {}

    def normalize_code_structure(self, code_str):
        """
        Convierte códigos equivalentes a un formato estándar único.
        Ejemplos:
        "01.01" -> "1.1"
        "1.1" -> "1.1"
        "02.05.00" -> "2.5.0"
        """
        try:
            # 1. Limpieza básica
            clean = str(code_str).strip()
            # Quitar punto final si existe (ej "1.1.")
            clean = clean.rstrip('.')

            # 2. Separar por puntos
            parts = clean.split('.')

            # 3. Convertir cada parte a entero para quitar ceros a la izquierda (01 -> 1)
            # Solo si la parte es numérica. Si es "A", se queda "A".
            normalized_parts = []
            for p in parts:
                if p.isdigit():
                    normalized_parts.append(str(int(p)))
                else:
                    normalized_parts.append(p)

            # 4. Reconstruir
            return ".".join(normalized_parts)
        except Exception:
            # Si falla algo raro, devolvemos el original limpio
            return str(code_str).strip()

    def normalize_text(self, text):
        """Limpieza AGRESIVA de texto."""
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        text = text.upper()
        text = text.replace('\xa0', ' ').replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def similarity(self, a, b):
        return SequenceMatcher(None, self.normalize_text(a), self.normalize_text(b)).ratio()

    def find_header_row(self, file_path):
        logger.info("Buscando fila de cabecera...")
        try:
            df_preview = pd.read_excel(file_path, header=None, nrows=20, engine='openpyxl')
            best_idx = 0
            max_matches = 0
            for idx, row in df_preview.iterrows():
                row_str = row.astype(str).str.lower().tolist()
                matches = sum(
                    1 for x in row_str if x in ["item", "descripción", "descripcion", "unidad", "metrado", "precio"])
                if matches > max_matches:
                    max_matches = matches
                    best_idx = idx
            logger.info(f"Mejor candidata para cabecera: Fila {best_idx + 1}")
            return best_idx
        except Exception as e:
            logger.error(f"Error buscando cabecera: {e}")
            return 0

    def load_budget_data(self):
        file_path = os.path.join(self.config["base_path"], self.config["budget_filename"])
        try:
            header_row = self.find_header_row(file_path)
            df = pd.read_excel(file_path, header=header_row, engine='openpyxl')
            df.columns = [str(c).strip() for c in df.columns]

            # --- LÓGICA DE SELECCIÓN DE COLUMNAS ---
            col_item = None
            col_desc = None

            for idx, col in enumerate(df.columns):
                c_lower = col.lower()
                if "item" in c_lower and not col_item: col_item = col
                if ("descripci" in c_lower or "partida" in c_lower) and not col_desc: col_desc = col

            if not col_item: col_item = df.columns[0]
            if not col_desc:
                # Heurística de longitud si falla el nombre
                max_len = 0
                candidate_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                for col in df.columns[:6]:
                    if col == col_item: continue
                    try:
                        avg_len = df[col].astype(str).str.len().mean()
                        if avg_len > max_len:
                            max_len = avg_len
                            candidate_col = col
                    except:
                        pass
                col_desc = candidate_col

            # Verificación de columna vacía
            if df[col_desc].isna().mean() > 0.95:
                current_idx = df.columns.get_loc(col_desc)
                if current_idx + 1 < len(df.columns):
                    col_desc = df.columns[current_idx + 1]

            logger.info(f"USANDO COLUMNAS -> Item: '{col_item}' | Desc: '{col_desc}'")

            df_clean = df[[col_item, col_desc]].copy()
            df_clean.columns = ["Item_Code", "Item_Desc"]
            df_clean = df_clean.dropna(subset=["Item_Code"])
            df_clean["Item_Code"] = df_clean["Item_Code"].astype(str).str.strip()
            df_clean = df_clean[df_clean["Item_Code"].str.contains(r'\d', na=False)]

            # === CORRECCIÓN DE JERARQUÍA: "EXCEL EMPIEZA EN 1.1" ===
            # Requisito: El excel tiene un nivel extra "1.1" que descuadra todo.
            # Accion: Normalizar códigos restando ese nivel extra.
            # 1.1 -> 1
            # 1.1.X -> 1.X

            def correct_hierarchy_offset(code):
                s_code = str(code).strip()
                # Caso raíz exacta
                if s_code == "1.1":
                    return "1"
                # Caso subpartidas (1.1.01...)
                if s_code.startswith("1.1."):
                    # Reemplazamos el prefijo '1.1.' por '1.'
                    # Ej: 1.1.01.02 -> 1.01.02
                    return "1." + s_code[4:]
                return s_code

            logger.info("Aplicando corrección de jerarquía: Asumiendo raíz Excel en 1.1")
            df_clean["Item_Code_Original"] = df_clean["Item_Code"]  # Backup
            df_clean["Item_Code"] = df_clean["Item_Code"].apply(correct_hierarchy_offset)

            self.df_budget = df_clean

            # === CREAR INDICES DE BÚSQUEDA ===
            self.budget_map = {}
            self.budget_flexible_map = {}

            for _, row in self.df_budget.iterrows():
                raw_code = row["Item_Code"]  # Ya corregido
                orig_code = row["Item_Code_Original"]
                raw_desc = row["Item_Desc"]

                # 1. Índice Normalizado Exacto
                norm_key = self.normalize_code_structure(raw_code)
                self.budget_map[norm_key] = {
                    "original_code": orig_code,
                    "desc": str(raw_desc)
                }

                # 2. Índice Flexible (Sufijos)
                parts = norm_key.split('.')
                if len(parts) > 1:
                    suffix_1 = ".".join(parts[1:])
                    if suffix_1 not in self.budget_flexible_map:
                        self.budget_flexible_map[suffix_1] = []
                    self.budget_flexible_map[suffix_1].append(norm_key)

            logger.info(f"Presupuesto indexado. {len(self.budget_map)} claves únicas.")

        except Exception as e:
            logger.critical(f"Error cargando Excel: {e}")
            raise

    def extract_from_word(self):
        file_path = os.path.join(self.config["base_path"], self.config["word_filename"])
        try:
            doc = Document(file_path)
            items_found = []
            pattern = re.compile(r"^(\d+[\.\d]*)\s+(.*)")

            for para in doc.paragraphs:
                text = para.text.strip()
                if not text: continue

                regex_match = pattern.match(text)
                is_style = self.config["word_style_name"].lower() in para.style.name.lower()

                if regex_match or (is_style and text[0].isdigit()):
                    if regex_match:
                        code, desc = regex_match.group(1), regex_match.group(2)
                    else:
                        parts = text.split(" ", 1)
                        code = parts[0]
                        desc = parts[1] if len(parts) > 1 else "SIN DESCRIPCION"

                    code = code.rstrip('.')

                    # Normalizamos el código del Word también
                    norm_code = self.normalize_code_structure(code)

                    items_found.append({
                        "Word_Code_Raw": code.strip(),
                        "Word_Code_Norm": norm_code,  # Clave normalizada
                        "Word_Desc": desc.strip(),
                        "Original_Text": text
                    })

            self.word_items = items_found
            logger.info(f"Word procesado: {len(items_found)} partidas encontradas.")
            pd.DataFrame(items_found).to_excel(os.path.join(results_dir, "Debug_Word_Raw.xlsx"), index=False)

        except Exception as e:
            logger.error(f"Error Word: {e}")
            raise

    def run_audit(self):
        logger.info("Comparando (ESTRATEGIA: JERARQUÍA CORREGIDA + EXACTITUD)...")

        for item in self.word_items:
            w_code_norm = item["Word_Code_Norm"]
            w_code_raw = item["Word_Code_Raw"]
            w_desc = item["Word_Desc"]

            status = "NO ENCONTRADO"
            excel_desc = "---"
            excel_code_orig = "---"
            score = 0.0

            match_data = None
            match_method = "NONE"

            # 1. INTENTO EXACTO (Con jerarquía Excel ya corregida)
            if w_code_norm in self.budget_map:
                match_data = self.budget_map[w_code_norm]
                match_method = "EXACTO (JERARQ. AJUSTADA)"

            # 2. INTENTO FLEXIBLE (Por si acaso queda algún sufijo suelto)
            if not match_data and w_code_norm in self.budget_flexible_map:
                candidates = self.budget_flexible_map[w_code_norm]
                best_candidate_key = candidates[0]
                match_data = self.budget_map[best_candidate_key]
                match_method = "SUFIJO COINCIDENTE"

            if match_data:
                excel_desc = match_data["desc"]
                excel_code_orig = match_data["original_code"]
                score = self.similarity(w_desc, excel_desc)

                if score >= self.config["min_similarity_ratio"]:
                    status = f"OK"
                else:
                    status = f"ALERTA: TEXTO DIFERENTE"
            else:
                status = "CODIGO NO EXISTE EN PRESUPUESTO"

            self.report.append({
                "Codigo_Word": w_code_norm,
                "Codigo_Excel_Original": excel_code_orig,  # El que salía en el Excel (ej 1.1.1.2)
                "Estado": status,
                "Similitud": f"{score:.2f}",
                "Texto_Word": w_desc,
                "Texto_Excel": excel_desc,
            })

        output_file = os.path.join(results_dir, "Reporte_Auditoria_FINAL.xlsx")
        pd.DataFrame(self.report).to_excel(output_file, index=False)
        logger.info(f"Reporte generado: {output_file}")

        # Resumen de alertas
        df_res = pd.DataFrame(self.report)
        print("\n--- RESUMEN DE COINCIDENCIAS ---")
        print(df_res["Estado"].value_counts())


if __name__ == "__main__":
    try:
        auditor = BudgetAuditor(CONFIG)
        auditor.extract_from_word()
        auditor.load_budget_data()
        if not auditor.df_budget.empty:
            auditor.run_audit()
    except Exception as e:
        logger.critical(f"Fallo: {e}")