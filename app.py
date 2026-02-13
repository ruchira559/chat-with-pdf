import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# --- UI Setup ---
st.set_page_config(page_title="AI Document Intelligence", layout="wide")
st.title("📄 AI Document Analyzer (RAG System)")

# --- State Management Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # To store the chat history

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None  # To store the PDF brain

# Sidebar for Setup & Clear Chat
with st.sidebar:
    st.header("Settings")
    # Add a Clear Chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []   # Correctly resets the existing lists
        st.rerun()
    
    # Adding Download History Button
    st.divider()
    chat_history_str = ""
    for msg in st.session_state.messages:
        chat_history_str += f"{msg['role'].capitalize()}: {msg['content']}\n\n"

    st.download_button(
        label="📥 Download Chat History",
        data=chat_history_str,
        file_name="chat_analysis.txt",
        mime="text/plain"
    )
    st.write("Built with Groq & LangChain (2026)")

# --- 1: File Uploader ---
uploaded_file = st.file_uploader("Upload a PDF for analysis", type="pdf")

if uploaded_file and st.session_state.vector_db is None:
    with st.status("Reading and indexing your PDF..."):
        # Save temp file to disk (PyPDFLoader needs a path)
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Ingestion Logic
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load() # This is where most errors occur
            
            # Continue with chunking if loading succeeds
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(pages)
            
            # Embedding & Storage
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vector_db = Chroma.from_documents(chunks, embeddings)
            st.success("PDF Indexed successfully!")

        except Exception as e:
            # Handle the error gracefully
            st.error(f"❌ Error reading PDF: {e}")
            st.stop() # Prevents the app from running rest of the code

# --- 2: Chat Interface ---
# Display existing messages from history
for message in st.session_state.messages:    # Find the initialized list
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask something about the document..."):
    # 1. Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate response using RAG
    if st.session_state.vector_db:
        llm = ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"], # We'll set this in Phase 5
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_db.as_retriever(),
            return_source_documents=True
        )
        
        response = qa_chain.invoke({"query": prompt})
        answer = response["result"]
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
            # Simple Source Citation
            if response["source_documents"]:
                with st.expander("View Sources"):
                    for doc in response["source_documents"]:
                        st.write(f"- {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:

        st.error("Please upload a PDF to begin.")
