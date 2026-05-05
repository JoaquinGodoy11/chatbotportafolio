# Mike — AI Agent with RAG Architecture (FastAPI + Groq)

**Live Demo:** [https://chatbotportafolio-production.up.railway.app](https://chatbotportafolio-production.up.railway.app)

This microservice demonstrates the implementation of a Virtual Assistant powered by Large Language Models (LLMs). The core of the project is a lightweight RAG architecture, designed to query documents (PDF/TXT), mitigating AI hallucinations and optimizing token consumption.

## 🛠 Tech Stack and Key Concepts

- **Backend:** Python, FastAPI (asynchronous).
- **AI & Inference:** Groq API. We can choose between two models using a dropdown menu (Llama 3.1 8B / 3.3 70B).
- **RAG Engine:** Dynamic chunking and lexical vectorization (TF-IDF via `scikit-learn`) for high-speed retrieval in memory-constrained environments.
- **Security & Resilience:** IP-based Rate Limiting (`slowapi`), origin control (strict CORS), and safe HTTP exception handling.

## 📐 Architecture Decisions and Trade-offs (Demo vs. Production)

To keep this project agile and suitable for a portfolio demonstration on a free tier (Railway), the following design decisions were made:

1. **TF-IDF Based RAG vs. Dense Embeddings:** I opted to implement linear algebra (TF-IDF and cosine similarity) for context retrieval, since it consumes less RAM.
2. **Non-blocking Event Loop:** The processing and text extraction of PDFs (CPU-bound) was delegated to a secondary *Threadpool* using `asyncio.to_thread()`.
3. **State Management:** The conversational history and vectors are kept in the container's RAM (`dict`). In a real-world production microservices architecture, this state would be externalized to **Redis** or a vector database (like pgvector or ChromaDB) to allow for horizontal scaling.

## 🚀 Local Installation and Execution

```bash
git clone [https://github.com/JoaquinGodoy11/chatbotportafolio](https://github.com/JoaquinGodoy11/chatbotportafolio)
cd chatbotportafolio
python -m venv venv

# Windows
venv\Scripts\activate


# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt