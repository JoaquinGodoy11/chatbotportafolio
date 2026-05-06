The API and the frontend UI will be available at `http://127.0.0.1:8000`[cite: 8].

***

### Tips finales para tu repo de GitHub:
1.  **Nombre del archivo:** Asegurate de que el archivo se llame exactamente `README.md` (con mayúsculas) para que GitHub lo renderice automáticamente en la página principal[cite: 8].
2.  **Imágenes:** Si tenés una captura de pantalla de cómo quedó la terminal (como la que me pasaste), podés subirla a una carpeta `/assets` en tu repo e insertarla en el README con `![Preview](./assets/screenshot.png)` para que entre por los ojos apenas abran el link[cite: 8].
3.  **About:** En la parte derecha de tu repo en GitHub, completá la sección "About" con el link de Railway y las etiquetas `python`, `fastapi`, `rag` e `ai`[cite: 8].

¡Quedó de diez, Joaquín! Cualquier reclutador técnico que lea esa sección de **Architecture Decisions** se va a dar cuenta de que sabés lo que hacésAquí tenés el contenido completo del **README.md** listo para copiar y pegar. He unificado los mejores puntos técnicos de tu versión original con las correcciones de infraestructura y los badges profesionales que le dan ese nivel de "proyeto listo para producción"[cite: 8].

***

# 🤖 Mike — AI Agent with RAG Architecture (FastAPI + Groq)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Railway-blue?style=for-the-badge)](https://chatbotportafoliojoa-production.up.railway.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=flat-square&logo=python)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)]()

This microservice demonstrates the implementation of a Virtual Assistant powered by Large Language Models (LLMs)[cite: 8]. The core of the project is a lightweight RAG (Retrieval-Augmented Generation) architecture, designed to query documents (PDF/TXT) dynamically, mitigating AI hallucinations and optimizing token consumption[cite: 8].

## 🛠 Tech Stack and Key Concepts

*   **Backend:** Python, FastAPI (Asynchronous)[cite: 8].
*   **AI & Inference:** Groq API (Support for Llama 3.1 8B & Llama 3.3 70B via UI dropdown)[cite: 8].
*   **RAG Engine:** Dynamic chunking and lexical vectorization (TF-IDF via `scikit-learn`) for high-speed retrieval in memory-constrained environments[cite: 8].
*   **Security & Resilience:** IP-based Rate Limiting (`slowapi`), strict CORS origin control, and safe HTTP exception handling[cite: 8].
*   **Frontend:** Vanilla HTML/CSS/JS with a responsive, terminal-inspired aesthetic (Retro-CLI)[cite: 8].

## 📐 Architecture Decisions & Trade-offs (Demo vs. Production)

To keep this project agile and suitable for a portfolio demonstration on a free-tier environment (Railway), the following design decisions were made[cite: 8]:

1.  **TF-IDF vs. Dense Embeddings:** I opted for linear algebra (TF-IDF and cosine similarity) for context retrieval instead of heavy transformer-based embeddings[cite: 8]. It provides incredibly fast lexical search while keeping RAM consumption extremely low[cite: 8].
2.  **Non-blocking Event Loop:** The processing and text extraction of PDFs (CPU-bound operations) was delegated to a secondary *Threadpool* using `asyncio.to_thread()` to prevent blocking the FastAPI asynchronous event loop[cite: 8].
3.  **State Management:** Conversational history and vectors are currently kept in the container's RAM[cite: 8]. In a real-world production microservices architecture, this state would be externalized to **Redis** (for session management) and a vector database like **pgvector** or **ChromaDB** to allow for horizontal scaling[cite: 8].

## 🚀 Local Installation and Execution

**1. Clone the repository and set up the virtual environment:**
```bash
git clone [https://github.com/JoaquinGodoy11/chatbotportafolio](https://github.com/JoaquinGodoy11/chatbotportafolio)
cd chatbotportafolio
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt


2. Environment Variables:
Create a .env file in the root directory and add your Groq API Key:
GROQ_API_KEY=your_api_key_here


3. Run the application:
uvicorn main:app --reload