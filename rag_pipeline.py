import os
import tempfile
import streamlit as _st # Import Streamlit privately for secrets access
from dotenv import load_dotenv
from typing import Optional, List
from langchain_core.documents import Document

# --- LangChain Component Imports ---
# Community Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Text Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain Classic (used for simplicity in this chain structure)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Google GenAI Imports
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except ImportError:
    # Set to None if the package is not installed (e.g., if only OpenAI is used)
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None

# OpenAI Imports (Fallback/Alternative - can be removed if strictly Gemini)
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# --- Configuration ---
# Load environment variables from .env file (for local use)
load_dotenv()
DB_PATH = "faiss_index"


# --- API Key Utility Functions ---

def _load_key(key_name: str) -> Optional[str]:
    """Tries to load an API key from env, then Streamlit secrets."""
    key = os.getenv(key_name)
    if key:
        return key

    # Try Streamlit secrets if running in the cloud
    try:
        key = _st.secrets.get(key_name) if hasattr(_st, "secrets") else None
        if key:
            # Set into os.environ so LangChain libraries can find it
            os.environ[key_name] = key
            return key
    except Exception:
        # Not running in Streamlit or secrets not available
        pass
    
    return os.getenv(key_name) # Final check

#def ensure_openai_api_key_loaded():
#    return _load_key("OPENAI_API_KEY")

def ensure_google_api_key_loaded():
    return _load_key("GOOGLE_API_KEY")

def get_preferred_provider() -> Optional[str]:
    """Return the preferred provider based on available keys: 'google' or 'openai' or None."""
    if ensure_google_api_key_loaded():
        return "google"
   # if ensure_openai_api_key_loaded():
   #     return "openai"
    return None

# --- 1. Data Ingestion Pipeline (The "Indexing" Phase) ---

def process_uploaded_files(uploaded_files: list) -> List[Document]:
    """
    Handles Streamlit file upload, saves them temporarily, loads them with PyPDFLoader, 
    updates metadata, and deletes the temporary files.
    """
    all_documents = []
    
    for uploaded_file in uploaded_files:
        # Use tempfile to get a safe path for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Update metadata to show the original uploaded file name
            for doc in documents:
                doc.metadata['source'] = uploaded_file.name
                doc.metadata['page'] = doc.metadata.get('page', 'N/A')
            
            all_documents.extend(documents)
        except Exception as e:
            print(f"Error processing {uploaded_file.name}: {e}")
        finally:
            # Crucially, delete the temporary file after processing
            os.remove(tmp_file_path)

    return all_documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into smaller, overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    return texts

def create_and_save_vector_store(texts: List[Document], db_path: str = DB_PATH):
    """Creates embeddings and builds/saves the FAISS vector store using the preferred provider."""
    provider = get_preferred_provider()
    if provider is None:
        raise ValueError(
            "No API key found. Please set either GOOGLE_API_KEY or OPENAI_API_KEY in .env or Streamlit secrets."
        )

    embeddings = None
    used_provider = None

    # --- GOOGLE EMBEDDINGS SETUP ---
    if provider == "google":
        key = ensure_google_api_key_loaded()
        if GoogleGenerativeAIEmbeddings is not None and key:
            try:
                # Use the corrected, current Gemini Embedding Model
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="gemini-embedding-001", 
                    google_api_key=key
                )
                used_provider = "google"
            except Exception as e:
                # Fallback warning
                print(f"Warning: Google embeddings failed (model config or runtime). Error: {e}")
                
    # --- OPENAI EMBEDDINGS FALLBACK ---
    #if embeddings is None and provider == "openai":
        #key = ensure_openai_api_key_loaded()
       # if key:
         #   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        #    used_provider = "openai"
        
    if embeddings is None:
        raise ValueError("Failed to initialize any usable embeddings provider.")

    print(f"Creating FAISS index using {used_provider} embeddings...")
    vector_store = FAISS.from_documents(texts, embeddings)
    
    os.makedirs(db_path, exist_ok=True)
    vector_store.save_local(db_path)
    print(f"Vector store saved to {db_path}")
    return vector_store

def load_vector_store(db_path: str = DB_PATH):
    """Loads an existing FAISS vector store using the preferred provider's embeddings."""
    provider = get_preferred_provider()
    if provider is None:
        raise ValueError(
            "No API key found. Cannot load vector store without an embeddings provider key."
        )

    embeddings = None

    # --- GOOGLE EMBEDDINGS LOAD ---
    if provider == "google" and GoogleGenerativeAIEmbeddings is not None:
        key = ensure_google_api_key_loaded()
        if key:
            # Use the corrected, current Gemini Embedding Model
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001", 
                google_api_key=key
            )
        
    # --- OPENAI EMBEDDINGS LOAD (FALLBACK) ---
    #if embeddings is None and provider == "openai":
     #   key = ensure_openai_api_key_loaded()
      #  if key:
       #     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if embeddings is None:
        raise ValueError("Failed to load embeddings provider. Check API keys and configuration.")
        
    print(f"Loading FAISS index...")
    # Setting allow_dangerous_deserialization=True is required for FAISS.load_local()
    vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# --- 2. RAG Chain Setup ---

def setup_rag_chain(vector_store):
    """Sets up the Retrieval-Augmented Generation chain using the preferred provider's LLM."""
    provider = get_preferred_provider()
    
    # 1. Initialize LLM
    if provider == "google":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed; cannot use Google LLM")
        # Use the corrected, current Gemini Chat Model
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    #elif provider == "openai":
    #    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    else:
        raise ValueError("No LLM provider initialized. Check API keys.")

    # 2. Define Retrieval Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful knowledge base agent. Use ONLY the following retrieved 
    context to answer the user's question. If you cannot find the answer in 
    the context, state clearly that the answer is not available in the documents.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """)

    # 3. Create Document Chain (Combines retrieved docs with prompt and sends to LLM)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 4. Create Retriever (Searches the Vector Store)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 5. Create Retrieval Chain (Connects Retriever to Document Chain)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain