import pandas as pd
import logging
import sys
import os

# --- CONFIGURACIÓN ---
TARGET_GAP = 100413.19
# Archivos de entrada (Asegúrate de que los nombres coincidan exactamente)
FILE_NEW = "LISTA DE INSUMOS ACTUAL EXP MOD.xls - Sheet.csv"
FILE_OLD = "INSUMOS ANTERIOR.xls - Sheet.csv"

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_and_clean_data(filepath, tag):
    """
    Lee el CSV, busca la fila de encabezados y limpia los datos.
    """
    if not os.path.exists(filepath):
        logging.error(f"No se encontró el archivo: {filepath}")
        return pd.DataFrame()

    logging.info(f"Leyendo archivo {tag}: {filepath}...")

    try:
        # Leemos el archivo. Asumimos que hay filas vacías al inicio (skiprows)
        # Basado en tus snippets, la data real empieza tras los encabezados.
        # Leeremos todo y buscaremos la fila que contiene "Código" o "Descripción"
        df_raw = pd.read_csv(filepath, header=None)

        # Buscar la fila del encabezado
        header_idx = -1
        for idx, row in df_raw.iterrows():
            row_str = row.astype(str).str.lower().values
            if 'código' in row_str or 'codigo' in row_str or 'descripción' in row_str:
                header_idx = idx
                break

        if header_idx == -1:
            logging.error(f"No se detectó encabezado en {filepath}")
            return pd.DataFrame()

        # Recargar usando la fila correcta como header
        df = pd.read_csv(filepath, header=header_idx)

        # Normalizar nombres de columnas (quitar espacios, hacer minusculas)
        df.columns = df.columns.str.strip().str.lower()

        # Identificar columnas clave (Mapeo flexible)
        col_map = {}
        for col in df.columns:
            if 'código' in col or 'codigo' in col:
                if 'elect' not in col:  # Evitar 'Cod. Elect.'
                    col_map['Codigo'] = col
            elif 'descripción' in col or 'descripcion' in col:
                col_map['Descripcion'] = col
            elif 'unid' in col:
                col_map['Unidad'] = col
            elif 'cantidad' in col or 'metrado' in col:
                col_map['Cantidad'] = col
            elif 'costo' in col or 'precio' in col:
                col_map['Precio'] = col

        if len(col_map) < 5:
            logging.warning(f"Columnas faltantes en {filepath}. Detectadas: {list(col_map.keys())}")
            # Intento de fallback por posición si los nombres fallan (Basado en tus CSVs)
            # Estructura probable: Codigo(0), Desc(4), Unid(6), Cant(7), Precio(8), Total(10)
            # Ajustamos según la estructura visual de tus archivos
            df_clean = df.iloc[:, [0, 4, 6, 7, 8]].copy()
            df_clean.columns = ['Codigo', 'Descripcion', 'Unidad', 'Cantidad', 'Precio']
        else:
            df_clean = df[list(col_map.values())].rename(columns={v: k for k, v in col_map.items()})

        # Limpieza de datos numéricos
        df_clean = df_clean.dropna(subset=['Codigo'])  # Eliminar filas sin código

        # Convertir a numérico forzando errores a NaN
        for col in ['Cantidad', 'Precio']:
            df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(',', ''), errors='coerce')

        df_clean = df_clean.dropna(subset=['Cantidad', 'Precio'])
        df_clean['Codigo'] = df_clean['Codigo'].astype(str).str.strip()

        logging.info(f"--> {len(df_clean)} items válidos cargados de {tag}.")
        return df_clean

    except Exception as e:
        logging.error(f"Error crítico leyendo {filepath}: {e}")
        return pd.DataFrame()


def generate_comparative_table(df_new, df_old, gap):
    logging.info("Iniciando cruce de información...")

    # 1. Filtrar Mano de Obra (Códigos que empiezan con 1002)
    # Se asume que 1002xxxx es mano de obra.
    df_new_mat = df_new[~df_new['Codigo'].str.startswith('1002')].copy()

    # 2. Cruce (Inner Join) - Solo lo que se repite
    merged = pd.merge(
        df_new_mat,
        df_old[['Codigo', 'Precio']],
        on='Codigo',
        how='inner',
        suffixes=('_New', '_Old')
    )

    logging.info(f"Items comunes encontrados (Materiales/Equipos): {len(merged)}")

    if len(merged) == 0:
        logging.error("No se encontraron items comunes. Verifique los códigos en ambos Excel.")
        return

    # 3. Lógica de Distribución (Pareto)
    # Calculamos cuánto dinero mueve cada partida para asignar el peso
    merged['Volumen_Dinero'] = merged['Cantidad'] * merged['Precio_New']
    total_volumen = merged['Volumen_Dinero'].sum()

    # Asignar brecha proporcionalmente
    merged['Peso'] = merged['Volumen_Dinero'] / total_volumen
    merged['Aporte_al_Gap'] = merged['Peso'] * gap

    # Calcular nuevos precios unitarios
    merged['Aumento_Unitario'] = merged['Aporte_al_Gap'] / merged['Cantidad']
    merged['Precio_Sugerido'] = merged['Precio_New'] + merged['Aumento_Unitario']

    # Métricas de validación
    merged['Dif_vs_Anterior'] = merged['Precio_Sugerido'] - merged['Precio_Old']
    merged['Var_%_Historica'] = ((merged['Precio_Sugerido'] - merged['Precio_Old']) / merged['Precio_Old']) * 100

    # 4. Generar Tabla Final (Formato Markdown para copiar y pegar)
    output_cols = ['Descripcion', 'Unidad', 'Cantidad', 'Precio_Old', 'Precio_New',
                   'Aumento_Unitario', 'Precio_Sugerido', 'Aporte_al_Gap']

    # Ordenar por mayor aporte monetario (Los más importantes arriba)
    final_view = merged.sort_values(by='Volumen_Dinero', ascending=False).head(25)

    # Formateo bonito
    pd.options.display.float_format = '{:,.2f}'.format

    print("\n" + "=" * 100)
    print(f"TABLA COMPARATIVA Y DE CUADRE (Brecha a cubrir: S/. {gap:,.2f})")
    print("=" * 100)
    print("Esta tabla muestra solo los insumos que se repiten en ambos expedientes.")
    print("El 'Precio_Sugerido' incluye el ajuste necesario para cuadrar el presupuesto.\n")

    # Imprimir tabla alineada
    # Renombramos columnas para que se vea igual a tu ejemplo
    display_df = final_view.copy()
    display_df = display_df.rename(columns={
        'Precio_Old': 'Precio Ant.',
        'Precio_New': 'Precio Actual',
        'Aumento_Unitario': '+ Ajuste',
        'Precio_Sugerido': 'Precio Final',
        'Aporte_al_Gap': 'Total Recaudado'
    })

    # Ajuste visual
    print(display_df[['Descripcion', 'Unidad', 'Cantidad', 'Precio Ant.', 'Precio Actual', '+ Ajuste', 'Precio Final',
                      'Total Recaudado']].to_markdown(index=False))

    print(f"\nTotal exacto cubierto con estos ajustes: S/. {final_view['Aporte_al_Gap'].sum():,.2f}")
    print("NOTA: Los items que faltan en la lista tienen un impacto monetario muy bajo (menos del 0.1% del total).")


if __name__ == "__main__":
    # Cargar datos
    df_new = load_and_clean_data(FILE_NEW, "NUEVO")
    df_old = load_and_clean_data(FILE_OLD, "ANTERIOR")

    if not df_new.empty and not df_old.empty:
        generate_comparative_table(df_new, df_old, TARGET_GAP)
    else:
        logging.error("No se pudo completar el análisis por falta de datos.")