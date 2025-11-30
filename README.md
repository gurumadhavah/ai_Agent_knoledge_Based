# 📚 Company KnowledgeBase RAG Agent

This is a local Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, and **Google Gemini**. It allows users to upload PDF documents to build a custom knowledge base and ask questions, receiving accurate answers based *only* on the provided documents.

## 🏗 Architecture

The application consists of two main pipelines:

1. **Ingestion:** PDFs are uploaded, split into chunks, embedded using Google's Embedding model, and stored in a local FAISS vector index.
2. **Retrieval:** User questions are embedded, relevant document chunks are retrieved from FAISS, and passed to Gemini 1.5 Flash to generate an answer.

### Architecture Diagram

![RAG Architecture](/img/Rooman_ai_agent_arch.png)

The diagram illustrates the complete data flow:
- **Left Side (Ingestion Pipeline):** User uploads PDFs → PyPDFLoader extracts text → Recursive Character Text Splitter creates chunks → Google Gemini Embeddings converts to vectors → Stored in FAISS Vector Store
- **Right Side (RAG Inference Pipeline):** User asks question → Query gets embedded → Vector Search retrieves Top K relevant documents → Context + Question sent to Google Gemini 2.5 Flash (LLM) → Final answer returned to user

## 🚀 Features

* **Document Ingestion:** Upload multiple PDF files simultaneously.
* **Vector Search:** Uses FAISS (Facebook AI Similarity Search) for fast local retrieval.
* **AI-Powered:** Uses Google Gemini 1.5 Flash for high-quality, fast inference.
* **Source Citing:** The agent cites the specific filename and page number for every answer.
* **Chat History:** Maintains context within the current session.
* **Persistence:** The Vector Store is saved to disk (`faiss_index`), so you don't have to rebuild the KB every time you restart the app.

## 🛠 Prerequisites

* Python 3.10+
* A Google Cloud API Key (for Gemini)

## 📦 Installation

1. **Clone the repository (or create your folder):**
   ```bash
   mkdir my-rag-agent
   cd my-rag-agent
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv myenv
   # Windows
   myenv\Scripts\activate
   # Mac/Linux
   source myenv/bin/activate
   ```

3. **Install Dependencies:**
   Create a `requirements.txt` file (see below) and run:
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. Create a `.env` file in the root directory.
2. Add your Google API key:

   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

   *Alternatively, if deploying to Streamlit Cloud, add this to your App Secrets.*

## 🏃‍♂️ How to Run

Execute the Streamlit application:

```bash
streamlit run app.py
```

1. The app will open in your browser (usually `http://localhost:8501`).
2. **Sidebar:** Upload PDF files and click "Build/Rebuild Knowledge Base".
3. **Chat:** Once the success message appears, type your questions in the main chat input.

## 📂 Project Structure

```
├── app.py                 # Main Streamlit frontend application
├── rag_pipeline.py        # Backend logic (Loading, Splitting, FAISS, Chain)
├── .env                   # API Keys (Do not commit to GitHub)
├── faiss_index/           # Folder auto-generated to store vectors
└── requirements.txt       # Python dependencies
```

## 🔧 Tech Stack

* **Frontend:** Streamlit
* **Framework:** LangChain
* **LLM:** Google Gemini 1.5 Flash / Gemini 2.5 Flash
* **Embeddings:** Google Generative AI Embeddings
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Document Processing:** PyPDF, RecursiveCharacterTextSplitter
* **API:** Google Cloud AI Platform

## ⚠️ Limitations

* **PDF Only:** Currently supports only PDF documents (no Word, Excel, or other formats)
* **Local Storage:** Vector store is saved locally; not suitable for distributed deployments without modifications
* **Context Window:** Limited by the LLM's context window size
* **No OCR:** Cannot extract text from scanned/image-based PDFs
* **Single Language:** Best performance with English documents
* **Session-Based Chat:** Chat history is cleared when the app restarts

## ❗ Troubleshooting

* **KeyError/AttributeError regarding session_state:** This usually happens if the session state isn't initialized correctly. Ensure you are using the latest code where initialization is outside `@st.cache_resource`.
* **"GoogleGenerativeAIEmbeddings not found":** Ensure you installed `langchain-google-genai`.
* **Empty Response:** If the bot says "I don't know," the answer likely isn't in your PDFs. Try increasing the `chunk_size` in `rag_pipeline.py`.

## 🚀 Potential Improvements

* **Multi-Format Support:** Add support for Word documents, Excel files, and text files
* **OCR Integration:** Implement OCR capabilities for scanned PDFs using tools like Tesseract
* **Advanced Chunking:** Experiment with semantic chunking strategies for better context preservation
* **Persistent Chat History:** Store conversation history in a database for continuity across sessions
* **Multi-Language Support:** Add language detection and support for non-English documents
* **Cloud Deployment:** Implement cloud-based vector storage (e.g., Pinecone, Weaviate) for scalability
* **User Authentication:** Add user management and authentication for multi-user scenarios
* **Enhanced UI:** Implement document preview, highlighting of relevant passages, and confidence scores
* **Hybrid Search:** Combine vector search with keyword-based search for improved retrieval
* **Fine-tuning:** Fine-tune embeddings on domain-specific data for specialized knowledge bases

## 👤 Author

**Gurumadhava H**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue in the repository.