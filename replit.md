# RAG Document Chatbot

## Overview

A Retrieval-Augmented Generation (RAG) chatbot application built with Streamlit that allows users to upload zip files containing documents and ask questions about their content. The system processes PDF, TXT, and DOCX files, creates vector embeddings for semantic search, and uses OpenAI's GPT models to generate context-aware responses based on the uploaded documents.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with a simple two-panel layout
- **UI Components**: Main chat interface with sidebar for document upload and statistics
- **State Management**: Session-based storage using Streamlit's session state for maintaining conversation history and processed documents

### Backend Architecture
- **Modular Design**: Separated into distinct components for document processing, RAG engine, and utility functions
- **Document Processing Pipeline**: Multi-step process for extracting, parsing, and chunking documents from zip files
- **RAG Implementation**: Custom RAG engine combining semantic search with generative AI responses

### Data Processing Flow
1. **Document Upload**: Zip file upload and extraction of supported document types
2. **Text Extraction**: Format-specific parsing for PDF, TXT, and DOCX files
3. **Text Chunking**: Breaking documents into manageable pieces for embedding generation
4. **Vector Embedding**: Converting text chunks to numerical vectors using sentence transformers
5. **Similarity Search**: Using FAISS for efficient nearest neighbor search
6. **Response Generation**: Combining retrieved context with OpenAI GPT models for answer synthesis

### Search and Retrieval
- **Embedding Model**: SentenceTransformer 'all-MiniLM-L6-v2' for generating document embeddings
- **Vector Database**: FAISS (Facebook AI Similarity Search) for fast similarity search
- **Retrieval Strategy**: Semantic similarity matching between user queries and document chunks

### AI Integration
- **Language Model**: OpenAI GPT-5 (latest model as of August 2025) for response generation
- **Prompt Engineering**: Context-aware prompting using retrieved document segments
- **Response Enhancement**: Source citation and relevance scoring for transparency

## External Dependencies

### AI and Machine Learning Services
- **OpenAI API**: GPT-5 language model for text generation and question answering
- **SentenceTransformers**: Pre-trained model for text embeddings generation
- **FAISS**: Facebook's library for efficient similarity search and clustering

### Document Processing Libraries
- **PyPDF2**: PDF document parsing and text extraction
- **python-docx**: Microsoft Word document processing
- **zipfile**: Built-in Python library for zip file handling

### Web Framework
- **Streamlit**: Web application framework for the user interface and session management

### Core Python Libraries
- **NumPy**: Numerical computing for vector operations
- **Pathlib**: Modern path handling and file system operations
- **tempfile**: Temporary file and directory management for document processing

### Configuration Requirements
- OpenAI API key required for GPT model access
- No database persistence - all data stored in session state during runtime
- File upload limitations based on Streamlit's default settings