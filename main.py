import os
import io
import uuid
import asyncio
import html
import pdfplumber
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# --- Configuración Base y Seguridad ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Mike AI Assistant API", description="Backend para asistente virtual con RAG ligero")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Restringimos los orígenes en producción para evitar abusos de la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatbotportafolio-production.up.railway.app", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DOCUMENTO_DEFAULT = """
PERFIL PROFESIONAL DE JOAQUÍN IGNACIO GODOY

INFORMACIÓN GENERAL:
Nombre: Joaquín Ignacio Godoy.
Ubicación: Córdoba, Argentina | Modalidad: Remoto.
Contacto: joacogodoy454@gmail.com
Perfil: Desarrollador Backend & AI Automation Developer (con conocimientos en frontend).
Idiomas: Español (Nativo), Inglés (B2).
Edad: Nacio en octubre de 2001, 24 años.
Github: https://github.com/JoaquinGodoy11
linkedin: https://www.linkedin.com/in/joaquin-godoy-39015b319/
RESUMEN:
Desarrollador enfocado en resolver problemas y construir arquitecturas sólidas. Estudiante de Ingeniería en Sistemas de Información, capacitado para diseñar APIs REST listas para producción (Python/FastAPI) e integrar LLMs (arquitecturas RAG) en aplicaciones escalables.
 Experiencia práctica con microservicios (Docker), pipelines CI/CD y testing automatizado.

HABILIDADES TÉCNICAS:
- Backend & Arquitectura: Python, FastAPI, Pydantic, SQLAlchemy, APIs REST, Arquitectura en Capas, Node.js.
- IA & Datos: Integración de LLMs (OpenAI API, RAG), Bases de Datos Vectoriales, LangChain, Pandas, Apache Spark, Prompt Engineering.
- Infraestructura: PostgreSQL, SQL, Alembic, Docker, AWS (conceptos básicos).
- DevOps & Automatización: GitHub Actions (CI/CD), Linux CLI, n8n, Webhooks, Airtable, ServiceTitan, Grafana.
- Testing: pytest, Unit & Integration Testing, enfoque Test-Driven.

EXPERIENCIA LABORAL:
- AI Voice Agent Developer (Freelance, Sept 2025 - Presente): Arquitectura y despliegue de agentes de voz para agencias, automatizando flujos de recepcionistas y reduciendo el manejo manual de llamadas en más del 60%.[cite: 6] Implementación de lógica conversacional multi-paso con integración CRM en tiempo real.
- IT Systems & Infrastructure Technician (Freelance, 2016 - 2024): Soporte técnico, configuración de entornos OS (Linux/Windows), redes y resolución de problemas de hardware.

PROYECTOS TÉCNICOS:
- Task Manager API: API REST lista para producción con aislamiento estricto de usuarios, auth JWT y arquitectura en capas. CI/CD vía GitHub Actions.
- Mike AI Chatbot: Chatbot RAG que procesa PDF/TXT usando LangChain y bases de datos vectoriales.

EDUCACIÓN:
- Ingeniería en Sistemas de Información (En curso) - UTN FRC.
- Certificación AI Engineer - PY Consulting (2025).
"""

MODELOS_PERMITIDOS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

# Estado en memoria (Trade-off: en producción cloud usaríamos Redis para escalar workers)
sesiones: dict = {}

# --- Lógica de RAG (Retrieval-Augmented Generation) ---

def chunkear_texto(texto: str, tamano_chunk: int = 800, solapamiento: int = 150) -> list:
    """Divide el documento en fragmentos superpuestos para mantener el contexto durante la búsqueda."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamano_chunk
        chunks.append(texto[inicio:fin])
        inicio += tamano_chunk - solapamiento
    return chunks if chunks else [texto]

def inicializar_motor_rag(texto: str):
    """Pre-calcula la matriz TF-IDF para búsquedas rápidas en memoria sin saturar RAM con embeddings densos."""
    chunks = chunkear_texto(texto)
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(chunks)
    except ValueError: # Manejo por si el texto es demasiado corto o carece de vocabulario
        tfidf_matrix = None
    return chunks, vectorizer, tfidf_matrix

def obtener_sesion(session_id: str) -> dict:
    if session_id not in sesiones:
        chunks, vectorizer, matrix = inicializar_motor_rag(DOCUMENTO_DEFAULT)
        sesiones[session_id] = {
            "documento_raw": DOCUMENTO_DEFAULT,
            "chunks": chunks,
            "vectorizer": vectorizer,
            "tfidf_matrix": matrix,
            "historial": [],
            "nombre_archivo": "perfil_joaquin.txt"
        }
    return sesiones[session_id]

def construir_system_prompt(contexto_recuperado: str) -> str:
    return f"""Sos Mike, el asistente virtual de Joaquín Godoy, un desarrollador Backend e IA.

CONTEXTO RECUPERADO DEL DOCUMENTO:
{contexto_recuperado}

REGLAS ABSOLUTAS:
1. Si el usuario saluda, respondé amigablemente.
2. Para preguntas técnicas o personales, basate ÚNICAMENTE en el contexto recuperado.
3. Si el contexto recuperado no responde la pregunta, indicá claramente: "No tengo esa información en los documentos provistos, por favor contactá a Joaquín."
4. Bajo ninguna circunstancia inventes información ni asumas datos fuera del contexto.
5. Respondé siempre en el mismo idioma en el que el usuario te hace la pregunta
"""

# El escape HTML lo delegamos al frontend para no corromper los prompts de código al LLM
def sanitizar(texto: str) -> str:
    return texto.strip()

# --- Funciones de I/O Pesadas ---

def extraer_texto_pdf(contenido: bytes) -> tuple[str, int]:
    """Extracción síncrona de PDF. Se aísla para ejecutarla en un Threadpool."""
    texto = ""
    with pdfplumber.open(io.BytesIO(contenido)) as pdf:
        paginas = len(pdf.pages)
        for pagina in pdf.pages:
            texto += pagina.extract_text() or ""
    return texto, paginas

# --- Modelos Pydantic ---
class Pregunta(BaseModel):
    mensaje: str
    modelo: str = "llama-3.1-8b-instant" # Por defecto un modelo más rápido
    session_id: str = ""

# --- Endpoints ---

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/upload")
@limiter.limit("10/hour")
async def upload(request: Request, file: UploadFile = File(...), session_id: str = ""):
    MAX_SIZE = 5 * 1024 * 1024  # 5MB límite duro
    contenido = await file.read()

    if len(contenido) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="El archivo supera el límite de 5MB")

    texto = ""
    paginas = None

    if file.filename.endswith(".pdf"):
        # Descargamos el bloqueo del Event Loop enviando la tarea pesada de CPU a un hilo secundario
        texto, paginas = await asyncio.to_thread(extraer_texto_pdf, contenido)
    elif file.filename.endswith(".txt"):
        texto = contenido.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Solo PDF o TXT")

    if not texto.strip():
        raise HTTPException(status_code=400, detail="El documento está vacío o es ilegible")

    session_id = session_id or str(uuid.uuid4())
    sesion = obtener_sesion(session_id)
    
    # Procesamos el nuevo documento para RAG
    chunks, vectorizer, matrix = inicializar_motor_rag(texto)
    
    sesion.update({
        "documento_raw": texto,
        "chunks": chunks,
        "vectorizer": vectorizer,
        "tfidf_matrix": matrix,
        "historial": [],
        "nombre_archivo": file.filename
    })

    return {
        "status": "ok",
        "archivo": file.filename,
        "caracteres": len(texto),
        "paginas": paginas,
        "session_id": session_id
    }

@app.post("/chat")
@limiter.limit("30/minute")
async def chat(request: Request, pregunta: Pregunta):
    mensaje = sanitizar(pregunta.mensaje)

    if not mensaje or len(mensaje) > 2000:
        raise HTTPException(status_code=400, detail="Mensaje inválido o excede longitud permitida")
    if pregunta.modelo not in MODELOS_PERMITIDOS:
        raise HTTPException(status_code=400, detail="Modelo de inferencia no autorizado")

    session_id = pregunta.session_id or str(uuid.uuid4())
    sesion = obtener_sesion(session_id)

    # --- Motor de Recuperación (Retrieval) ---
    contexto_relevante = ""
    if sesion["tfidf_matrix"] is not None and sesion["chunks"]:
        query_vec = sesion["vectorizer"].transform([mensaje])
        similitudes = cosine_similarity(query_vec, sesion["tfidf_matrix"]).flatten()
        
        # Recuperamos los top 3 chunks más relevantes
        indices_top = similitudes.argsort()[-3:][::-1]
        
        # Umbral dinámico de similitud (evita inyectar basura si la pregunta es off-topic)
        chunks_filtrados = [sesion["chunks"][i] for i in indices_top if similitudes[i] > 0.05]
        if chunks_filtrados:
            contexto_relevante = "\n...\n".join(chunks_filtrados)
        else:
            contexto_relevante = "No se encontraron coincidencias exactas en el documento."

    # Gestión de memoria a corto plazo
    if len(sesion["historial"]) > 10:
        sesion["historial"] = sesion["historial"][-10:]

    sesion["historial"].append({"role": "user", "content": mensaje})

    mensajes_llm = [
        {"role": "system", "content": construir_system_prompt(contexto_relevante)}
    ] + sesion["historial"]

    try:
        respuesta = client.chat.completions.create(
            model=pregunta.modelo,
            messages=mensajes_llm,
            temperature=0.1 # Temperatura baja para grounding en RAG
        )
        contenido = respuesta.choices[0].message.content or "Sin respuesta"
    except Exception as e:
        sesion["historial"].pop() # Rollback si falla Groq
        raise HTTPException(status_code=502, detail="Error en upstream AI provider")

    sesion["historial"].append({"role": "assistant", "content": contenido})

    return {"respuesta": contenido, "session_id": session_id}

@app.post("/reset")
async def reset(pregunta: Pregunta):
    session_id = pregunta.session_id or str(uuid.uuid4())
    if session_id in sesiones:
        del sesiones[session_id]
    obtener_sesion(session_id) # Fuerza la recreación con defaults
    return {"status": "ok", "session_id": session_id}

@app.get("/estado")
async def get_estado(session_id: str = ""):
    sesion = obtener_sesion(session_id)
    return {"archivo": sesion["nombre_archivo"], "mensajes": len(sesion["historial"])}

app.mount("/static", StaticFiles(directory="static"), name="static")