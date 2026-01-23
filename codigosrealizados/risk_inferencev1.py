# -*- coding: utf-8 -*-
"""
emairc_dashboard_app.py (v4.3)

Plataforma de Inteligencia de Riesgos para EMAIRC VISIÓN 🧠

Objetivo:
Evolucionar la herramienta de inferencia a una plataforma de análisis integral,
incorporando análisis masivo de reportes, un generador de charlas de seguridad
avanzado y una innovadora simulación de mapa de riesgos proactivo.

Mejoras sobre v4.2:
- Se soluciona el error 'showTab is not defined' moviendo los manejadores de eventos
  desde los atributos 'onclick' en el HTML a un bloque de script centralizado.
- Se utiliza el evento 'DOMContentLoaded' para asegurar que todo el HTML esté cargado
  antes de que el JavaScript intente adjuntar los listeners, eliminando condiciones
  de carrera y mejorando la robustez de la aplicación.
"""
import logging
from pathlib import Path
import sys
import pandas as pd
import torch
import uvicorn
import pickle
import io
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertModel
from typing import List, Dict

# --- 1. CONFIGURACIÓN INICIAL ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

# NOTA: Se recomienda usar rutas relativas o variables de entorno en lugar de rutas absolutas.
# Por ahora, se mantienen las rutas que proporcionaste.
OUTPUT_DIR = Path(r"D:\Python_proyectos_2025\SEGURIDAD2.0\ModeloV3.2\output")
DATA_FILE_PATH = Path(r"D:\Python_proyectos_2025\SEGURIDAD2.0\ModeloV3.2\data\fatalities_augmented_FINAL.csv")

if not OUTPUT_DIR.exists():
    logging.critical(f"¡ERROR CRÍTICO! Directorio de modelos no encontrado: {OUTPUT_DIR}")
    sys.exit(1)


# --- Modelos de Datos para la API ---
class PredictionRequest(BaseModel): narrative: str


class BatchPredictionRequest(BaseModel): narratives: List[str]


class ToolboxTalkRequest(BaseModel): categories: List[str]


class ProactiveRiskRequest(BaseModel): tasks: List[Dict]


class PredictionDetails(BaseModel): label: str; confidence: float


class PredictionResponse(BaseModel):
    nature_title: PredictionDetails
    part_of_body_title: PredictionDetails
    event_title: PredictionDetails


# --- 2. LÓGICA DE CARGA DE MODELOS Y DATOS ---
class RiskClassifier(torch.nn.Module):
    def __init__(self, n_classes, model_name='distilbert-base-uncased'):
        super(RiskClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs[0][:, 0])


def find_latest_model_path(target_column: str) -> Path:
    target_dir = OUTPUT_DIR / target_column
    if not target_dir.exists(): raise FileNotFoundError(f"Directorio no encontrado: {target_dir}")
    all_runs = sorted([d for d in target_dir.iterdir() if d.is_dir()], reverse=True)
    if not all_runs: raise FileNotFoundError(f"No se encontraron entrenamientos en: {target_dir}")
    return all_runs[0]


def load_model_artifacts(model_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_path = model_path / f"label_encoder_{model_path.parent.name}.pkl"
    with open(encoder_path, "rb") as f: encoder = pickle.load(f)
    model = RiskClassifier(n_classes=len(encoder.classes_))
    # weights_only=True es una buena práctica de seguridad.
    model.load_state_dict(torch.load(model_path / "risk_predictor_model.bin", map_location=device, weights_only=True))
    model.to(device);
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained(model_path / "tokenizer")
    return model, tokenizer, encoder, device


logging.info("--- EMAIRC VISIÓN: Iniciando Plataforma v4.3 ---")
MODELS = {}
for model_type in ["NatureTitle", "Part_of_Body_Title", "EventTitle"]:
    try:
        latest_path = find_latest_model_path(model_type)
        model, tokenizer, encoder, device = load_model_artifacts(latest_path)
        MODELS[model_type] = {"model": model, "tokenizer": tokenizer, "encoder": encoder, "device": device}
        logging.info(f"✅ Modelo '{model_type}' cargado.")
    except Exception as e:
        logging.error(f"❌ Falló la carga del modelo '{model_type}'. Error: {e}")
        MODELS[model_type] = None

try:
    DF_MAIN = pd.read_csv(DATA_FILE_PATH, sep=';', encoding='utf-8-sig')
    logging.info(f"✅ Dataset de auditoría cargado.")
except Exception:
    logging.warning("⚠️ No se pudo cargar el dataset de auditoría. La pestaña 'Auditoría Global' no funcionará.")
    DF_MAIN = None


# --- 4. LÓGICA DE NEGOCIO ---
def predict(narrative: str, model_type: str) -> (str, float):
    if not MODELS.get(model_type): return "Modelo no disponible", 0.0
    artifacts = MODELS[model_type]
    with torch.no_grad():
        encoding = artifacts["tokenizer"].encode_plus(narrative, max_length=256, padding='max_length', truncation=True,
                                                      return_tensors='pt', add_special_tokens=True)
        outputs = artifacts["model"](input_ids=encoding['input_ids'].to(artifacts["device"]),
                                     attention_mask=encoding['attention_mask'].to(artifacts["device"]))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction_idx = torch.max(probabilities, dim=1)
        predicted_label = artifacts["encoder"].inverse_transform(prediction_idx.cpu().numpy())[0]
        return predicted_label, confidence.cpu().item()


TOOLBOX_TALKS_CONTENT = {
    "altura": {
        "title": "Trabajos en Altura (Andamios, Escaleras, Techos)",
        "points": [
            "**Inspección de Equipos:** Antes de usar, revisar arneses, líneas de vida y puntos de anclaje. ¿Alguien ve algún defecto en su equipo hoy?",
            "**Regla de los 3 Puntos:** Mantener siempre tres puntos de contacto al subir o bajar escaleras.",
            "**Zona de Exclusión:** Asegurar que el área debajo de la zona de trabajo esté delimitada para proteger al personal de objetos que caen. ¿Está nuestra zona señalizada?",
        ],
        "call_to_action": "La complacencia es el mayor riesgo en altura. Doble chequeo, siempre. Cuidémonos unos a otros."
    },
    "electrico": {
        "title": "Trabajos Eléctricos",
        "points": [
            "**Energía Cero (LOTO):** Antes de cualquier intervención, confirmar que el equipo está desenergizado, bloqueado y etiquetado. ¿Quién es el responsable de verificar el bloqueo hoy?",
            "**EPP Dieléctrico:** El uso de guantes y calzado dieléctrico no es negociable. Revisen su estado ahora mismo.",
            "**Herramientas Aisladas:** Utilizar únicamente herramientas con certificación de aislamiento. Las herramientas incorrectas pueden ser mortales.",
        ],
        "call_to_action": "Con la electricidad no hay segundas oportunidades. Verifiquen, luego trabajen."
    },
    "excavacion": {
        "title": "Trabajos en Zanjas y Excavaciones",
        "points": [
            "**Inspección de Taludes:** Antes de ingresar, un supervisor debe inspeccionar las paredes de la zanja en busca de fisuras o signos de inestabilidad.",
            "**Vías de Escape:** Debe haber una escalera o rampa de acceso y escape cada 7.5 metros de distancia horizontal. ¿Están nuestras vías de escape despejadas?",
            "**Acopio de Material:** No acumular tierra o materiales a menos de 1 metro del borde de la excavación.",
        ],
        "call_to_action": "Una zanja puede convertirse en una tumba en segundos. Respeten los bordes y las protecciones."
    },
    "soldadura": {
        "title": "Trabajos de Soldadura (Corte y Esmerilado)",
        "points": [
            "**Permiso de Trabajo en Caliente:** ¿Tenemos el permiso firmado y visible en el área de trabajo?",
            "**Control de Incendios:** Mantener un extintor de incendios adecuado y operativo a menos de 3 metros. Retirar todos los materiales combustibles del área.",
            "**Protección Completa:** El uso de careta de soldar, guantes de cuero, ropa ignífuga y protección respiratoria es obligatorio.",
        ],
        "call_to_action": "El fuego no perdona. Un área limpia y un equipo de protección completo son nuestra mejor defensa."
    }
}


def generate_toolbox_talk(categories: List[str]) -> str:
    if not categories:
        return "Por favor, seleccione al menos una categoría de trabajo para generar la charla de seguridad."

    full_talk = "📢 **Charla de Seguridad (Toolbox Talk) para Hoy:**\n\n"
    for category in categories:
        if category in TOOLBOX_TALKS_CONTENT:
            content = TOOLBOX_TALKS_CONTENT[category]
            full_talk += f"\n--- \n### {content['title']}\n\n"
            for point in content['points']:
                full_talk += f"- {point}\n"
            full_talk += f"\n**Llamado a la Acción:** *{content['call_to_action']}*\n"

    return full_talk


# --- 5. APLICACIÓN WEB CON FASTAPI ---
app = FastAPI(title="EMAIRC VISIÓN - Plataforma de Inteligencia")


@app.post("/api/predict", response_model=PredictionResponse)
async def handle_prediction(request: PredictionRequest):
    label_nature, conf_nature = predict(request.narrative, "NatureTitle")
    label_body, conf_body = predict(request.narrative, "Part_of_Body_Title")
    label_event, conf_event = predict(request.narrative, "EventTitle")
    return PredictionResponse(nature_title={"label": label_nature, "confidence": conf_nature},
                              part_of_body_title={"label": label_body, "confidence": conf_body},
                              event_title={"label": label_event, "confidence": conf_event})


@app.post("/api/batch-predict")
async def handle_batch_predict(request: BatchPredictionRequest):
    results = []
    for narrative in request.narratives:
        if not narrative or not isinstance(narrative, str): continue
        ln, cn = predict(narrative, "NatureTitle")
        lb, cb = predict(narrative, "Part_of_Body_Title")
        le, ce = predict(narrative, "EventTitle")
        results.append({"Narrativa": narrative, "Naturaleza": ln, "Confianza_Naturaleza": cn, "Parte_Cuerpo": lb,
                        "Confianza_Cuerpo": cb, "Causa_Evento": le, "Confianza_Causa": ce})
    return JSONResponse(content=results)


@app.get("/api/dataset-audit")
async def get_dataset_audit():
    if DF_MAIN is None: return JSONResponse(status_code=500, content={"error": "Dataset no disponible."})
    top_n = 10
    event_counts = DF_MAIN['EventTitle'].value_counts().nlargest(top_n)
    nature_counts = DF_MAIN['NatureTitle'].value_counts().nlargest(top_n)
    body_counts = DF_MAIN['Part_of_Body_Title'].value_counts().nlargest(top_n)
    return {"event": {"labels": event_counts.index.tolist(), "data": event_counts.values.tolist()},
            "nature": {"labels": nature_counts.index.tolist(), "data": nature_counts.values.tolist()},
            "body": {"labels": body_counts.index.tolist(), "data": body_counts.values.tolist()}}


@app.post("/api/generate-toolbox-talk", response_model=Dict[str, str])
async def handle_toolbox_talk(request: ToolboxTalkRequest):
    talk = generate_toolbox_talk(request.categories)
    return {"talk": talk}


@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    html_content = """
    <!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EMAIRC VISIÓN - Plataforma de Inteligencia</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/papaparse@5.3.0"></script>
    <script src="https://unpkg.com/xlsx/dist/xlsx.full.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .tab-btn { transition: all 0.3s; cursor: pointer; }
        .tab-btn.active { border-color: #3B82F6; color: #3B82F6; background-color: #EFF6FF; }
        .toolbox-btn { transition: all 0.2s; }
        .toolbox-btn.selected { background-color: #16A34A; color: white; transform: scale(1.05); }
        #risk-map-canvas { background-size: contain; background-position: center; background-repeat: no-repeat; cursor: crosshair; }
        /* Estilo para el contenedor de notificaciones */
        #notification-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            display: none;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #EF4444; /* Rojo por defecto */
            color: white;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <!-- Contenedor para notificaciones -->
    <div id="notification-container"></div>

    <div class="container mx-auto p-4 md:p-8 max-w-7xl">
        <header class="text-center mb-8"><h1 class="text-4xl font-bold text-gray-900">EMAIRC VISIÓN</h1><p class="text-xl text-gray-600 mt-2">🧠 Plataforma de Inteligencia de Riesgos v4.3</p></header>
        <div class="mb-8 flex flex-wrap justify-center border-b border-gray-300">
            <!-- CORRECCIÓN: Se eliminan los 'onclick' y se añade 'data-tab' para identificarlos en JS -->
            <button id="tab-interactive-btn" data-tab="interactive" class="tab-btn active text-lg font-semibold py-3 px-6 border-b-2 border-transparent">Análisis Interactivo</button>
            <button id="tab-batch-btn" data-tab="batch" class="tab-btn text-lg font-semibold py-3 px-6 border-b-2 border-transparent">Análisis Masivo</button>
            <button id="tab-toolbox-btn" data-tab="toolbox" class="tab-btn text-lg font-semibold py-3 px-6 border-b-2 border-transparent">Charlas de Seguridad</button>
            <button id="tab-riskmap-btn" data-tab="riskmap" class="tab-btn text-lg font-semibold py-3 px-6 border-b-2 border-transparent">Mapa de Riesgo</button>
            <button id="tab-audit-btn" data-tab="audit" class="tab-btn text-lg font-semibold py-3 px-6 border-b-2 border-transparent">Auditoría Global</button>
        </div>

        <!-- Contenido de las Pestañas -->
        <div id="content-interactive" class="tab-content">
            <!-- Módulo de Análisis Interactivo -->
            <main class="bg-white p-6 rounded-xl shadow-lg"><div class="mb-6"><label for="narrative" class="block text-lg font-medium text-gray-700 mb-2">Ingrese la Narrativa del Accidente</label><textarea id="narrative" rows="8" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 transition"></textarea></div><div class="text-center"><button id="analyze-btn" class="bg-blue-600 text-white font-bold py-3 px-8 rounded-lg hover:bg-blue-700 transition-transform transform hover:scale-105 disabled:bg-gray-400"><span id="btn-text">Analizar Riesgo</span><span id="btn-spinner" class="hidden"><svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>Analizando...</span></button></div></main>
            <section id="results-section" class="mt-8 hidden"><h2 class="text-2xl font-bold text-center mb-6">Resultados del Análisis</h2><div class="grid grid-cols-1 md:grid-cols-3 gap-6"><div id="nature-card" class="bg-white p-5 rounded-xl shadow-md border-l-4"><h3 class="flex items-center font-semibold text-gray-600 mb-2"><svg class="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>Naturaleza</h3><p id="nature-result" class="text-lg font-medium p-3 rounded-md"></p><p class="text-sm text-gray-500 mt-2">Confianza: <span id="nature-confidence" class="font-semibold"></span></p></div><div id="body-card" class="bg-white p-5 rounded-xl shadow-md border-l-4"><h3 class="flex items-center font-semibold text-gray-600 mb-2"><svg class="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>Parte del Cuerpo</h3><p id="body-result" class="text-lg font-medium p-3 rounded-md"></p><p class="text-sm text-gray-500 mt-2">Confianza: <span id="body-confidence" class="font-semibold"></span></p></div><div id="event-card" class="bg-white p-5 rounded-xl shadow-md border-l-4"><h3 class="flex items-center font-semibold text-gray-600 mb-2"><svg class="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>Causa del Evento</h3><p id="event-result" class="text-lg font-medium p-3 rounded-md"></p><p class="text-sm text-gray-500 mt-2">Confianza: <span id="event-confidence" class="font-semibold"></span></p></div></div></section>
            <section id="final-alert-section" class="mt-8 hidden"><div class="bg-white p-5 rounded-xl shadow-md border-l-4 border-blue-500"><h3 class="flex items-center font-semibold text-gray-600 mb-3"><svg class="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"></path></svg>Alerta Consolidada</h3><p id="final-alert-message" class="text-lg font-medium text-blue-800"></p></div></section>
        </div>

        <div id="content-batch" class="tab-content hidden">
            <!-- Módulo de Análisis Masivo -->
            <div class="bg-white p-6 rounded-xl shadow-lg">
                <h2 class="text-2xl font-bold text-center mb-4">Análisis Masivo de Reportes</h2>
                <p class="text-center text-gray-600 mb-6">Cargue un archivo CSV con una columna llamada "FinalNarrative" para analizarlas en lote, generar un dashboard y exportar los resultados a Excel.</p>
                <div class="text-center">
                    <input type="file" id="csv-upload" accept=".csv" class="hidden"/>
                    <button id="upload-btn" class="bg-indigo-600 text-white font-bold py-3 px-8 rounded-lg hover:bg-indigo-700 transition-transform transform hover:scale-105">Cargar Archivo CSV</button>
                </div>
                <div id="batch-results" class="mt-8 hidden">
                    <div class="flex justify-between items-center mb-6">
                        <h3 id="batch-title" class="text-2xl font-bold">Resultados del Lote</h3>
                        <button id="download-excel-btn" class="bg-green-600 text-white font-bold py-2 px-6 rounded-lg hover:bg-green-700 transition">Descargar Excel</button>
                    </div>
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        <div><h3 class="text-xl font-semibold text-center mb-4">Top Causas de Eventos (Lote)</h3><canvas id="batchEventChart"></canvas></div>
                        <div><h3 class="text-xl font-semibold text-center mb-4">Top Naturalezas de Lesión (Lote)</h3><canvas id="batchNatureChart"></canvas></div>
                    </div>
                </div>
                 <div id="batch-spinner" class="text-center mt-8 hidden"><svg class="animate-spin mx-auto h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg><p class="mt-2 text-lg">Analizando, por favor espere...</p></div>
            </div>
        </div>

        <div id="content-toolbox" class="tab-content hidden">
            <!-- Módulo de Charlas de Seguridad -->
            <div class="bg-white p-6 rounded-xl shadow-lg">
                <h2 class="text-2xl font-bold text-center mb-4">Generador de Charlas de Seguridad</h2>
                <p class="text-center text-gray-600 mb-6">Seleccione las categorías de trabajo del día para generar una charla de seguridad ("Toolbox Talk") enfocada en los riesgos más probables.</p>
                <div id="toolbox-categories" class="flex flex-wrap justify-center gap-4 mb-6">
                    <button class="toolbox-btn bg-gray-200 text-gray-700 font-semibold py-2 px-4 rounded-full" data-category="altura">Trabajos en Altura</button>
                    <button class="toolbox-btn bg-gray-200 text-gray-700 font-semibold py-2 px-4 rounded-full" data-category="electrico">Trabajos Eléctricos</button>
                    <button class="toolbox-btn bg-gray-200 text-gray-700 font-semibold py-2 px-4 rounded-full" data-category="excavacion">Excavaciones</button>
                    <button class="toolbox-btn bg-gray-200 text-gray-700 font-semibold py-2 px-4 rounded-full" data-category="soldadura">Trabajos en Caliente</button>
                </div>
                <div id="toolbox-result-card" class="bg-gray-50 p-4 rounded-lg border border-gray-200 hidden">
                    <div id="toolbox-result" class="text-gray-700 whitespace-pre-wrap prose"></div>
                </div>
            </div>
        </div>

        <div id="content-riskmap" class="tab-content hidden">
            <!-- Módulo de Mapa de Riesgo -->
            <div class="bg-white p-6 rounded-xl shadow-lg">
                <h2 class="text-2xl font-bold text-center mb-4">Mapa de Riesgo Proactivo</h2>
                <p class="text-center text-gray-600 mb-6">Suba un plano de la obra, luego seleccione una tarea y haga clic en el plano para colocarla y visualizar las zonas de riesgo.</p>
                <div class="flex flex-col md:flex-row gap-6 items-center">
                    <div class="flex-shrink-0">
                        <input type="file" id="plan-upload" accept="image/*" class="hidden"/>
                        <button id="upload-plan-btn" class="bg-gray-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-gray-700">Subir Plano</button>
                        <div id="task-palette" class="mt-4 space-y-2">
                            <p class="font-semibold">Seleccione Tarea:</p>
                            <button class="task-btn border-2 border-transparent p-2 rounded-lg w-full text-left" data-task="altura" data-risk="high">🏗️ Trabajos en Altura</button>
                            <button class="task-btn border-2 border-transparent p-2 rounded-lg w-full text-left" data-task="excavacion" data-risk="high">🚜 Excavación</button>
                            <button class="task-btn border-2 border-transparent p-2 rounded-lg w-full text-left" data-task="soldadura" data-risk="medium">🔥 Soldadura</button>
                            <button class="task-btn border-2 border-transparent p-2 rounded-lg w-full text-left" data-task="electrico" data-risk="high">⚡️ Trabajos Eléctricos</button>
                        </div>
                    </div>
                    <div class="flex-grow w-full h-96 bg-gray-200 rounded-lg relative overflow-hidden">
                        <canvas id="risk-map-canvas" class="w-full h-full"></canvas>
                        <div id="plan-placeholder" class="absolute inset-0 flex items-center justify-center text-gray-500">Suba un plano para comenzar</div>
                    </div>
                </div>
            </div>
        </div>

        <div id="content-audit" class="tab-content hidden">
            <!-- Módulo de Auditoría Global -->
            <div class="bg-white p-6 rounded-xl shadow-lg"><h2 class="text-2xl font-bold text-center mb-6">Auditoría de Riesgos del Dataset Global</h2><p class="text-center text-gray-600 mb-8">Estos gráficos muestran las 10 categorías más frecuentes en el dataset completo, revelando los patrones de riesgo históricos.</p><div class="grid grid-cols-1 lg:grid-cols-2 gap-8"><div><h3 class="text-xl font-semibold text-center mb-4">Top 10 Causas de Eventos</h3><canvas id="eventChart"></canvas></div><div><h3 class="text-xl font-semibold text-center mb-4">Top 10 Naturalezas de Lesión</h3><canvas id="natureChart"></canvas></div><div class="lg:col-span-2"><h3 class="text-xl font-semibold text-center mb-4">Top 10 Partes del Cuerpo Afectadas</h3><canvas id="bodyChart"></canvas></div></div></div>
        </div>
    </div>

    <script>
    // CORRECCIÓN: Toda la lógica de JS se envuelve en DOMContentLoaded
    // para asegurar que el HTML esté listo antes de ejecutar el script.
    document.addEventListener('DOMContentLoaded', () => {

        // --- Lógica Global de la App ---
        const tabs = ['interactive', 'batch', 'toolbox', 'riskmap', 'audit'];
        const tabButtons = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        function showTab(tabId) {
            tabContents.forEach(content => {
                // Oculta todos los contenidos
                if (!content.id.endsWith(tabId)) {
                    content.classList.add('hidden');
                } else {
                    content.classList.remove('hidden');
                }
            });

            tabButtons.forEach(button => {
                // Desactiva todos los botones
                if (button.dataset.tab !== tabId) {
                    button.classList.remove('active');
                } else {
                    button.classList.add('active');
                }
            });

            if (tabId === 'audit') {
                loadGlobalAuditCharts();
            }
        }

        // Adjuntar listeners a los botones de las pestañas
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                showTab(button.dataset.tab);
            });
        });

        // --- Sistema de Notificaciones ---
        const notificationContainer = document.getElementById('notification-container');
        function showNotification(message, type = 'error', duration = 3000) {
            notificationContainer.textContent = message;
            notificationContainer.style.backgroundColor = type === 'error' ? '#EF4444' : '#10B981';
            notificationContainer.style.display = 'block';
            setTimeout(() => {
                notificationContainer.style.display = 'none';
            }, duration);
        }

        // --- Lógica Módulo Análisis Interactivo ---
        const analyzeBtn = document.getElementById('analyze-btn');
        analyzeBtn.addEventListener('click', async () => {
            const narrative = document.getElementById('narrative').value.trim();
            if (!narrative) {
                showNotification('Por favor, ingrese una narrativa.');
                return;
            }
            const btnText = document.getElementById('btn-text');
            const btnSpinner = document.getElementById('btn-spinner');
            analyzeBtn.disabled = true;
            btnText.classList.add('hidden');
            btnSpinner.classList.remove('hidden');
            document.getElementById('results-section').classList.add('hidden');
            document.getElementById('final-alert-section').classList.add('hidden');
            try {
                const response = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ narrative }) });
                if (!response.ok) throw new Error(`Error del servidor: ${response.statusText}`);
                const data = await response.json();
                updateCard('nature', data.nature_title, 'green');
                updateCard('body', data.part_of_body_title, 'red');
                updateCard('event', data.event_title, 'yellow');
                document.getElementById('final-alert-message').textContent = buildAlertMessage(data);
                document.getElementById('results-section').classList.remove('hidden');
                document.getElementById('final-alert-section').classList.remove('hidden');
            } catch (error) {
                console.error('Error en análisis interactivo:', error);
                showNotification('Ocurrió un error al analizar. Revise la consola para más detalles.');
            } finally {
                analyzeBtn.disabled = false;
                btnText.classList.remove('hidden');
                btnSpinner.classList.add('hidden');
            }
        });

        const CONFIDENCE_THRESHOLD = 0.40;
        const colorStyles = {
            green: { border: 'border-green-500', text: 'text-green-800', bg: 'bg-green-100' },
            red: { border: 'border-red-500', text: 'text-red-800', bg: 'bg-red-100' },
            yellow: { border: 'border-yellow-500', text: 'text-yellow-800', bg: 'bg-yellow-100' },
            gray: { border: 'border-gray-400', text: 'text-gray-700', bg: 'bg-gray-100' }
        };

        function updateCard(type, data, colorName) {
            const card = document.getElementById(`${type}-card`);
            const resultP = document.getElementById(`${type}-result`);
            const confidenceSpan = document.getElementById(`${type}-confidence`);

            confidenceSpan.textContent = `${(data.confidence * 100).toFixed(1)}%`;

            card.className = 'bg-white p-5 rounded-xl shadow-md border-l-4';
            resultP.className = 'text-lg font-medium p-3 rounded-md';

            let label = data.label.replace(/, unspecified/gi, '').replace(/-nonspecified injury/gi, '');
            let styles;

            if (data.confidence < CONFIDENCE_THRESHOLD) {
                resultP.textContent = "Análisis no concluyente";
                styles = colorStyles.gray;
            } else {
                resultP.textContent = label;
                styles = colorStyles[colorName] || colorStyles.gray;
            }

            card.classList.add(styles.border);
            resultP.classList.add(styles.text, styles.bg);
        }

        function buildAlertMessage(data) {
            const parts = [];
            const cleanLabel = (d) => d.label.replace(/, unspecified/gi, '').replace(/-nonspecified injury/gi, '').toLowerCase();
            if (data.event_title.confidence >= CONFIDENCE_THRESHOLD) parts.push(`Riesgo potencial de "${cleanLabel(data.event_title)}"`);
            if (data.nature_title.confidence >= CONFIDENCE_THRESHOLD) parts.push(`lo que podría resultar en "${cleanLabel(data.nature_title)}"`);
            if (data.part_of_body_title.confidence >= CONFIDENCE_THRESHOLD) parts.push(`afectando principalmente la zona de "${cleanLabel(data.part_of_body_title)}"`);
            if (parts.length === 0) return "El análisis no es concluyente. Se recomienda una revisión manual.";
            let message = `¡Atención! ${parts.join(', ')}. Se requiere verificar el área y tomar precauciones.`;
            return message.charAt(0).toUpperCase() + message.slice(1);
        }

        // --- Lógica Módulo Análisis Masivo ---
        const uploadBtn = document.getElementById('upload-btn');
        const csvUpload = document.getElementById('csv-upload');
        const batchResultsSection = document.getElementById('batch-results');
        const batchSpinner = document.getElementById('batch-spinner');
        let batchData = [];
        let batchCharts = {};

        uploadBtn.addEventListener('click', () => csvUpload.click());
        csvUpload.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                batchResultsSection.classList.add('hidden');
                batchSpinner.classList.remove('hidden');
                Papa.parse(file, {
                    header: true,
                    skipEmptyLines: true,
                    complete: async (results) => {
                        const narratives = results.data.map(row => row.FinalNarrative).filter(Boolean);
                        if (narratives.length === 0) {
                            showNotification('El archivo CSV no contiene la columna "FinalNarrative" o está vacía.');
                            batchSpinner.classList.add('hidden');
                            return;
                        }
                        document.getElementById('batch-title').textContent = `Resultados de ${file.name}`;
                        try {
                            const response = await fetch('/api/batch-predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ narratives }) });
                            if (!response.ok) throw new Error(`Error del servidor: ${response.statusText}`);
                            batchData = await response.json();
                            displayBatchResults(batchData);
                        } catch (e) {
                            console.error('Error en análisis masivo:', e);
                            showNotification('Error en análisis masivo. Revise la consola.');
                        }
                        finally { batchSpinner.classList.add('hidden'); }
                    }
                });
            }
        });

        function displayBatchResults(data) {
            batchResultsSection.classList.remove('hidden');
            const eventCounts = data.reduce((acc, row) => { acc[row.Causa_Evento] = (acc[row.Causa_Evento] || 0) + 1; return acc; }, {});
            const natureCounts = data.reduce((acc, row) => { acc[row.Naturaleza] = (acc[row.Naturaleza] || 0) + 1; return acc; }, {});
            const top10Events = Object.entries(eventCounts).sort((a, b) => b[1] - a[1]).slice(0, 10);
            const top10Natures = Object.entries(natureCounts).sort((a, b) => b[1] - a[1]).slice(0, 10);
            if (batchCharts.event) batchCharts.event.destroy();
            if (batchCharts.nature) batchCharts.nature.destroy();
            const chartOptions = { indexAxis: 'y', responsive: true, plugins: { legend: { display: false } } };
            batchCharts.event = new Chart(document.getElementById('batchEventChart'), { type: 'bar', data: { labels: top10Events.map(e => e[0]), datasets: [{ data: top10Events.map(e => e[1]), backgroundColor: '#4F46E5' }] }, options: chartOptions });
            batchCharts.nature = new Chart(document.getElementById('batchNatureChart'), { type: 'bar', data: { labels: top10Natures.map(n => n[0]), datasets: [{ data: top10Natures.map(n => n[1]), backgroundColor: '#10B981' }] }, options: chartOptions });
        }

        document.getElementById('download-excel-btn').addEventListener('click', () => {
            if (batchData.length === 0) {
                showNotification('No hay datos para descargar.', 'info');
                return;
            }
            const worksheet = XLSX.utils.json_to_sheet(batchData);
            const workbook = XLSX.utils.book_new();
            XLSX.utils.book_append_sheet(workbook, worksheet, "Resultados");
            XLSX.writeFile(workbook, "Resultados_Analisis_EMARC.xlsx");
        });

        // --- Lógica Módulo Auditoría Global ---
        let globalCharts = {};
        async function loadGlobalAuditCharts() {
            if (Object.keys(globalCharts).length > 0) return;
            try {
                const response = await fetch('/api/dataset-audit');
                if (!response.ok) throw new Error(`Error del servidor: ${response.statusText}`);
                const data = await response.json();
                const chartOptions = { indexAxis: 'y', responsive: true, plugins: { legend: { display: false } } };
                globalCharts.event = new Chart(document.getElementById('eventChart'), { type: 'bar', data: { labels: data.event.labels, datasets: [{ data: data.event.data, backgroundColor: '#3B82F6' }] }, options: chartOptions });
                globalCharts.nature = new Chart(document.getElementById('natureChart'), { type: 'bar', data: { labels: data.nature.labels, datasets: [{ data: data.nature.data, backgroundColor: '#10B981' }] }, options: chartOptions });
                globalCharts.body = new Chart(document.getElementById('bodyChart'), { type: 'bar', data: { labels: data.body.labels, datasets: [{ data: data.body.data, backgroundColor: '#EF4444' }] }, options: chartOptions });
            } catch (error) { console.error("Error al cargar datos de auditoría:", error); showNotification('No se pudieron cargar los datos de auditoría.');}
        }

        // --- Lógica Módulo Charlas de Seguridad ---
        const toolboxCategories = document.getElementById('toolbox-categories');
        const toolboxResultCard = document.getElementById('toolbox-result-card');
        const toolboxResult = document.getElementById('toolbox-result');
        toolboxCategories.addEventListener('click', async (e) => {
            if (e.target.tagName === 'BUTTON') {
                e.target.classList.toggle('selected');
                const selectedCategories = Array.from(toolboxCategories.querySelectorAll('.selected')).map(btn => btn.dataset.category);
                if (selectedCategories.length > 0) {
                    try {
                        const response = await fetch('/api/generate-toolbox-talk', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ categories: selectedCategories }) });
                        if (!response.ok) throw new Error(`Error del servidor: ${response.statusText}`);
                        const data = await response.json();
                        toolboxResult.innerHTML = data.talk.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\\n/g, '<br>');
                        toolboxResultCard.classList.remove('hidden');
                    } catch (error) { console.error("Error al generar charla:", error); showNotification('No se pudo generar la charla de seguridad.'); }
                } else {
                    toolboxResultCard.classList.add('hidden');
                }
            }
        });

        // --- Lógica Módulo Mapa de Riesgo ---
        const planUploadBtn = document.getElementById('upload-plan-btn');
        const planUpload = document.getElementById('plan-upload');
        const canvas = document.getElementById('risk-map-canvas');
        const ctx = canvas.getContext('2d');
        const taskPalette = document.getElementById('task-palette');
        let selectedTask = null;

        planUploadBtn.addEventListener('click', () => planUpload.click());
        planUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    const img = new Image();
                    img.onload = () => {
                        canvas.width = img.width;
                        canvas.height = img.height;
                        canvas.style.backgroundImage = `url('${event.target.result}')`;
                        document.getElementById('plan-placeholder').classList.add('hidden');
                    }
                    img.src = event.target.result;
                };
                reader.readAsDataURL(file);
            }
        });

        taskPalette.addEventListener('click', (e) => {
            const button = e.target.closest('.task-btn');
            if (button) {
                document.querySelectorAll('.task-btn').forEach(btn => btn.classList.remove('bg-blue-200', 'border-blue-500'));
                button.classList.add('bg-blue-200', 'border-blue-500');
                selectedTask = { type: button.dataset.task, risk: button.dataset.risk, icon: button.textContent.split(' ')[0] };
            }
        });

        canvas.addEventListener('click', (e) => {
            if (!selectedTask) {
                showNotification("Por favor, seleccione una tarea de la paleta.");
                return;
            }
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;

            ctx.font = '24px sans-serif';
            ctx.fillText(selectedTask.icon, x - 12, y + 8);

            let radius = selectedTask.risk === 'high' ? 50 : 30;
            let color = selectedTask.risk === 'high' ? 'rgba(255, 0, 0, 0.3)' : 'rgba(255, 255, 0, 0.4)';
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI, false);
            ctx.fillStyle = color;
            ctx.fill();
        });
    });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    logging.info("Iniciando la aplicación de inferencia de EMAIRC VISIÓN...")
    logging.info("Accede a la interfaz en http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
