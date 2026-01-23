import requests
import json
from requests.exceptions import JSONDecodeError

# 1. PEGA AQUÍ TU URL DE *PRUEBA* (TEST URL) DEL NODO WEBHOOK
# La obtienes de la interfaz de n8n
url = "https://emairc.app.n8n.cloud/webhook/new-lead-webhook"

# --- DATOS REALES (DE TUS CSV DE COSTAMAR) ---
# Usamos 'nombre' y 'telefono' porque esto es lo que
# tu script 'normalize_lead_data.js' (corregido) ahora espera.
data = {
    "nombre": "DARIAN HINOJOSA",
    "telefono": "955600170",
    "Source": "Test_Costamar_CSV"
}

# --- O TAMBIÉN PUEDES USAR DATOS DE EMAIRC ---
# (El script JS ahora maneja ambos)
# data = {
#    "Site_Name": "Constructora SACYR Perú",
#    "URL": "https://www.sacyr.com/peru/proyectos"
# }

print(f"Enviando datos a: {url}")
print(f"Datos: {json.dumps(data, indent=2)}")

try:
    response = requests.post(url, json=data)
    print(f"\n--- RESPUESTA ---")
    print(f"HTTP Status Code: {response.status_code}")

    if response.text:
        try:
            print(f"Respuesta JSON del Webhook: {response.json()}")
        except JSONDecodeError:
            print(f"Respuesta (Texto) del Webhook: {response.text}")
    else:
        print("Webhook de prueba activado (Respuesta sin cuerpo JSON).")

except Exception as e:
    print(f"\n--- ERROR INESPERADO ---")
    print(f"Ocurrió un error: {e}")