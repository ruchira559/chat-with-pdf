# 📄 AI Document Intelligence: Research DQA Analyzer

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://chat-with-pdf-eh9ct9bhaafgdedk9prpnp.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Made with LangChain](https://img.shields.io/badge/Made%20with-LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langchain)

An industry-grade **Retrieval-Augmented Generation (RAG)** application designed to automate Data Quality Assessment (DQA) in scientific research. This tool allows users to upload complex PDF documents and conduct a context-aware dialogue with an LLM, backed by mathematical evidence from the source text.

---

## 🚀 Key Features

- **Semantic Search & Retrieval**: Implements a high-performance RAG pipeline using **ChromaDB** and **HuggingFace Embeddings**.
- **State-of-the-Art LLM**: Powered by **Llama-3.1-8b** via **Groq Cloud** for near-instant inference speeds.
- **Source Transparency**: Every response includes an "Evidence" expander showing the exact page and text chunk used to generate the answer.
- **Data Portability**: Integrated feature to **Download Chat History** as a `.txt` report for offline analysis.
- **Session Persistence**: Utilizes `st.session_state` to maintain the indexed document and conversation history across interactions.

---

## 📊 Performance Benchmarks
This system was validated using the research paper *MacMaster & Sinistore (2024)*. The AI successfully replicated the study's results with the following accuracy:

| Indicator | Target (PDF) | AI Verification Result |
| :--- | :--- | :--- |
| **Temporal Coverage** | **100%** | ✅ Verified |
| **Geographic Coverage** | **91%** | ✅ Verified |
| **Technology Coverage** | **73%** | ✅ Verified |

---

## 🛠️ Technical Architecture



1. **Ingestion**: `PyPDFLoader` extracts text from uploaded files.
2. **Chunking**: `RecursiveCharacterTextSplitter` creates 1000-character overlaps to preserve context.
3. **Vectorization**: `all-MiniLM-L6-v2` transformer model generates 384-dimensional embeddings.
4. **Retrieval**: Semantic similarity search retrieves the top 5 most relevant document segments.
5. **Generation**: Groq's Llama-3.1 synthesizes the final response using a strict "System Prompt."

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_USERNAME]/chat-with-pdf.git
cd chat-with-pdf
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Setup Secrets
In your Streamlit Cloud dashboard (Settings > Secrets) or a local `.streamlit/secrets.toml`:
```Ini, TOML
GROQ_API_KEY = "your_actual_groq_key_here"
```
### 4. Run the App
```bash
streamlit run app.py
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📂 Project Structure

```text
chat-with-pdf/
├── app.py                         # Main Streamlit application (UI & RAG Logic)
├── research_and_prototyping.ipynb # Original development & testing notebook
├── requirements.txt               # Python dependencies for Cloud deployment
├── data.pdf                       # Sample research paper (MacMaster & Sinistore, 2024)
├── LICENSE                        # MIT Open Source License
└── README.md                      # Project documentation and setup guide
```

---

## 🤝 Contact

**Ruchir Agrawal** [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ruchir-a-308240128/)  
**Project Link:** [https://github.com/ruchira559/chat-with-pdf](https://github.com/ruchira559/chat-with-pdf)
