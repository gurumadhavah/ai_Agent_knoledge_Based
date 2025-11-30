import streamlit as st
import os
from rag_pipeline import (
    process_uploaded_files,
    split_documents,
    create_and_save_vector_store,
    load_vector_store,
    setup_rag_chain,
    DB_PATH
)

# --- Streamlit Configuration ---
st.set_page_config(page_title="KnowledgeBase Agent", layout="wide")
st.title("📚 Company KnowledgeBase RAG Agent")

# --- Session State and Initialization ---

# 1. Initialize Session State (Run this on EVERY rerun, do not cache)
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'rag_chain' not in st.session_state:
    st.session_state['rag_chain'] = None

# 2. Define the Cached Resource Loader
# Only put heavy lifting (loading files/models) here.
@st.cache_resource
def load_cached_chain():
    """Attempts to load the RAG chain from disk."""
    if os.path.exists(DB_PATH):
        try:
            vector_store = load_vector_store()
            rag_chain = setup_rag_chain(vector_store)
            print("Existing knowledge base loaded successfully!")
            return rag_chain
        except Exception as e:
            print(f"Error loading existing vector store: {e}. Starting with an empty agent.")
            return None
    return None

# 3. Load the chain and update session state if it exists
# We call the cached function. If it returns a chain, we assign it to session state
# (unless we just built a new one manually).
cached_chain = load_cached_chain()
if cached_chain and st.session_state.rag_chain is None:
    st.session_state.rag_chain = cached_chain

# --- Sidebar for File Upload and KB Creation ---
with st.sidebar:
    st.header("1. Document Ingestion")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Files to Build KB",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    # Build Button
    if st.button("Build/Rebuild Knowledge Base (KB)"):
        if not uploaded_files:
            st.warning("Please upload one or more PDF files first.")
        else:
            with st.spinner(f"Processing {len(uploaded_files)} documents..."):
                try:
                    # 1. Process files
                    documents = process_uploaded_files(uploaded_files)
                    
                    # 2. Split and Vectorize
                    texts = split_documents(documents)
                    vector_store = create_and_save_vector_store(texts)
                    
                    # 3. Setup Agent
                    # Clear the cache so the next load picks up the new DB
                    load_cached_chain.clear()
                    
                    # Assign the new chain directly to session state
                    st.session_state.rag_chain = setup_rag_chain(vector_store)
                    
                    st.success(f"Knowledge Base built from {len(uploaded_files)} file(s) and ready to use!")
                    st.session_state.messages = [] # Clear chat history on rebuild
                except Exception as e:
                    st.error(f"An error occurred during KB creation: {e}")

    st.markdown("---")
    st.header("2. Reset")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if the RAG chain is ready
    if st.session_state.rag_chain is None:
        st.warning("The RAG Agent is not initialized. Please upload files and click 'Build KB' first.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating response..."):
            
            try:
                # The invocation that runs the entire RAG pipeline
                response = st.session_state.rag_chain.invoke({"input": prompt})
                
                answer = response.get('answer', 'No answer generated.')
                context = response.get('context', [])
                
                sources = "\n".join([
                    f"- *Source:* {doc.metadata.get('source', 'Unknown Source')} (Page: {doc.metadata.get('page', 'N/A')})"
                    for doc in context
                ])
                
                full_response = f"{answer}\n\n---\n\n*Sources Used:*\n{sources}"
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            except Exception as e:
                st.error(f"An error occurred during response generation. Check your API key or quota. Error: {e}")
                st.session_state.messages.pop()