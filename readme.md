The API and the frontend UI will be available at `http://127.0.0.1:8000`[cite: 8].

# 🤖 Mike — AI Agent with RAG Architecture (FastAPI + Groq)

This microservice demonstrates the implementation of a virtual assistant powered by Large Language Models (LLMs). The core of the project is a lightweight RAG (Retrieval-Augmented Generation) architecture, designed to dynamically query documents (PDF/TXT), mitigating AI hallucinations and optimizing token consumption.

## Tech Stack and Key Concepts
- Backend: Python, FastAPI (Asynchronous)
- AI & Inference: Groq API (Llama 3.1 8B & Llama 3.3 70B via UI dropdown)
- RAG Engine: Dynamic chunking and lexical vectorization (TF-IDF via scikit-learn) for high-speed retrieval in memory-constrained environments
- Security & Resilience: IP-based rate limiting (slowapi), strict CORS origin control, safe HTTP exception handling
- Frontend: Vanilla HTML/CSS/JS with a responsive, terminal-inspired aesthetic (Retro-CLI)

## Architecture Decisions & Trade-offs (Demo vs. Production)
To keep this project agile and suitable for a portfolio demo on a free-tier environment (Railway), these design decisions were made:
- **TF-IDF vs. Dense Embeddings:** chose linear algebra (TF-IDF + cosine similarity) over transformer-based embeddings for fast lexical search with low RAM usage.
- **Non-blocking Event Loop:** PDF text extraction (CPU-bound) is delegated to a secondary threadpool using `asyncio.to_thread()` to avoid blocking FastAPI's async event loop.
- **State Management:** conversational history and vectors are kept in the container's RAM. In a production system, this would be externalized to Redis (session management) and a vector database like pgvector or ChromaDB for horizontal scaling.

## Local Installation and Execution

1. Clone the repository and set up a virtual environment:
git clone https://github.com/JoaquinGodoy11/chatbotportafolio
cd chatbotportafolio
python -m venv venv
