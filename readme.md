# 🤖 AI Chatbot con RAG

Chatbot inteligente que responde preguntas sobre documentos propios usando IA. Construido con FastAPI y Groq.

## 🛠️ Stack
- **Backend:** Python, FastAPI
- **IA:** Groq API (LLaMA 3.3 70B)
- **Frontend:** HTML, CSS, JavaScript vanilla
- **RAG:** Procesamiento de PDF y TXT con pdfplumber

## ⚙️ Instalación

1. Cloná el repositorio
2. Creá el entorno virtual:
```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
```
3. Instalá dependencias:
```bash
   pip install -r requirements.txt
```
4. Creá un archivo `.env`: 
GROQ_API_KEY=tu_api_key_aqui

5. Corré el servidor:
```bash
   uvicorn main:app --reload
```
6. Abrí `http://localhost:8000`

## ✨ Features
- Subida de archivos PDF y TXT
- Respuestas basadas exclusivamente en el documento cargado
- Memoria de conversación dentro de la sesión
- Protección contra prompt injection
- Reset de conversación
- Animación de tipeo y timestamps en mensajes

## 🔒 Seguridad
- API key protegida con variables de entorno
- Archivos sensibles excluidos con `.gitignore`
- System prompt reforzado contra manipulación