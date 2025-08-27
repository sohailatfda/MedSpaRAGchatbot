import streamlit as st
import os
import tempfile
import zipfile
from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from utils import initialize_session_state, display_chat_message, handle_file_upload

def main():
    st.set_page_config(
        page_title="RAG Document Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Document Chatbot")
    st.markdown("Upload a zip file containing documents and ask questions about their content!")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a zip file",
            type=['zip'],
            help="Upload a zip file containing PDF, TXT, or DOCX documents"
        )
        
        if uploaded_file is not None:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    success = handle_file_upload(uploaded_file)
                    if success:
                        st.success("Documents processed successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to process documents. Please check the file format.")
        
        # Display document stats if available
        if st.session_state.rag_engine is not None:
            st.subheader("📊 Document Statistics")
            stats = st.session_state.rag_engine.get_stats()
            st.write(f"**Documents processed:** {stats['num_documents']}")
            st.write(f"**Text chunks:** {stats['num_chunks']}")
            st.write(f"**Vector embeddings:** {stats['num_embeddings']}")
            
            if st.button("Clear Documents"):
                st.session_state.rag_engine = None
                st.session_state.messages = []
                st.success("Documents cleared!")
                st.rerun()
    
    # Main chat interface
    if st.session_state.rag_engine is None:
        st.info("👈 Please upload and process documents to start chatting!")
    else:
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(message)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message({"role": "user", "content": prompt})
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating response..."):
                    try:
                        response = st.session_state.rag_engine.query(prompt)
                        
                        # Display the response
                        st.markdown(response["answer"])
                        
                        # Display sources if available
                        if response["sources"]:
                            with st.expander("📚 Source Documents", expanded=False):
                                for i, source in enumerate(response["sources"], 1):
                                    st.markdown(f"**Source {i}:** {source['filename']}")
                                    st.markdown(f"*Relevance Score: {source['score']:.3f}*")
                                    st.text(source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"])
                                    st.divider()
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response["answer"],
                            "sources": response["sources"]
                        })
                    
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })

if __name__ == "__main__":
    main()
