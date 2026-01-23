import speech_recognition as sr
import pyttsx3
from openai import OpenAI

# --- CONFIGURA TU CLAVE API DE OPENAI ---
client = OpenAI(api_key="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")

# Inicializa motor de voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Cambiar voz si deseas (opcional)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Cambia a otra si quieres voz más grave

def speak(text):
    print(f"Archon: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Escuchando...")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio, language="es-PE")
        print(f"Tú: {query}")
        return query
    except sr.UnknownValueError:
        speak("No entendí, ¿puedes repetirlo?")
        return ""
    except sr.RequestError:
        speak("Error de conexión con el servicio de voz.")
        return ""

def ask_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres Archon, el asistente personal de Emanuel. Habla con sabiduría, respeto y motivación."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Bucle principal
speak("Hola Emanuel, soy Archon. Estoy listo para ayudarte.")
while True:
    pregunta = listen()
    if pregunta.lower() in ["salir", "terminar", "detente"]:
        speak("Hasta luego, Emanuel. Siempre estaré contigo.")
        break
    if pregunta:
        respuesta = ask_openai(pregunta)
        speak(respuesta)
