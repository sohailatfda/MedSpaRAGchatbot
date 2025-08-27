import streamlit as st
import tempfile
import os
from document_processor import DocumentProcessor
from rag_engine import RAGEngine

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()

def display_chat_message(message):
    """Display a chat message in the appropriate format."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available (for assistant messages)
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 Source Documents", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:** {source['filename']}")
                    st.markdown(f"*Relevance Score: {source['score']:.3f}*")
                    st.text(source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"])
                    if i < len(message["sources"]):
                        st.divider()

def handle_file_upload(uploaded_file) -> bool:
    """Handle file upload and processing."""
    try:
        # Process documents from uploaded zip file
        processed_documents = st.session_state.document_processor.process_documents(
            uploaded_file.getvalue()
        )
        
        if not processed_documents:
            st.error("No valid documents found in the zip file.")
            return False
        
        # Initialize RAG engine if not already done
        if st.session_state.rag_engine is None:
            st.session_state.rag_engine = RAGEngine()
        
        # Load documents into RAG engine
        st.session_state.rag_engine.load_documents(processed_documents)
        
        # Clear previous chat messages
        st.session_state.messages = []
        
        # Display processing summary
        total_chunks = sum(doc['num_chunks'] for doc in processed_documents)
        st.success(
            f"Successfully processed {len(processed_documents)} documents "
            f"with {total_chunks} text chunks!"
        )
        
        return True
    
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes = int(size_bytes / 1024)
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def validate_openai_key() -> bool:
    """Validate if OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key is not None and api_key != "default_key"
