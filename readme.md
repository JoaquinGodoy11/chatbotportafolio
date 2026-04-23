# Mike — AI Chatbot con RAG

 **Demo en vivo:** [chatbotportafolio-production.up.railway.app](https://chatbotportafolio-production.up.railway.app)

---

Construí este proyecto para demostrar cómo integrar un LLM con documentos propios de forma segura y deployarlo en producción. La idea es simple: subís cualquier PDF o TXT y el chatbot responde preguntas sobre ese documento. Sin inventar, sin salirse del contexto.

## ¿Qué tiene adentro?

El backend corre en FastAPI y se comunica con la API de Groq para inferencia. Implementé RAG básico manteniendo el documento cargado en el contexto del modelo junto con el historial de conversación. Cada usuario tiene su propia sesión.

En el lado de seguridad, hicimos unas configuraciones iniciales como: rate limiting por IP, CORS configurado para el dominio de producción, sanitización de inputs en backend y frontend para prevenir XSS, y validación de modelos contra una whitelist.

## Stack

- **Backend:** Python, FastAPI, Groq API (LLaMA 3.3 70B)
- **Frontend:** HTML, CSS, JavaScript vanilla
- **Deploy:** Railway
- **Librerías:** pdfplumber, slowapi, python-dotenv

## Instalación local

```bash
git clone https://github.com/JoaquinGodoy11/chatbotportafolio
cd chatbotportafolio
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Creá un archivo `.env`:
GROQ_API_KEY=tu_api_key_aqui
Corré el servidor:
```bash
uvicorn main:app --reload
```

Abrí `http://localhost:8000`

## Features

- Subida de archivos PDF y TXT (máx 5MB)
- Selección de modelo (LLaMA 70B o 8B)
- Memoria de conversación por sesión
- Protección contra prompt injection
- Export del chat a TXT
- Reset de sesión

---

Hecho por [Joaquín Godoy](https://github.com/JoaquinGodoy11)