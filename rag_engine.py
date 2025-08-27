import os
import numpy as np
from typing import List, Dict, Any
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

class RAGEngine:
    """Retrieval-Augmented Generation engine for document querying."""
    
    def __init__(self):
        # Initialize OpenAI client
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        api_key = os.getenv("OPENAI_API_KEY", "default_key")
        self.openai_client = OpenAI(api_key=api_key)
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Storage for documents and embeddings
        self.documents = []
        self.chunks = []
        self.chunk_metadata = []
        self.embeddings = None
        self.faiss_index = None
    
    def load_documents(self, processed_documents: List[Dict[str, Any]]):
        """Load processed documents and create embeddings."""
        self.documents = processed_documents
        self.chunks = []
        self.chunk_metadata = []
        
        # Collect all chunks with metadata
        for doc_idx, document in enumerate(processed_documents):
            for chunk_idx, chunk in enumerate(document['chunks']):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'doc_idx': doc_idx,
                    'chunk_idx': chunk_idx,
                    'filename': document['filename'],
                    'file_type': document['file_type']
                })
        
        if not self.chunks:
            raise ValueError("No chunks found in processed documents")
        
        # Generate embeddings
        with st.spinner("Generating embeddings..."):
            self.embeddings = self.embedding_model.encode(
                self.chunks,
                show_progress_bar=True,
                convert_to_tensor=False
            )
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.faiss_index.add(normalized_embeddings.astype(np.float32))
    
    def retrieve_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most similar chunks to the query."""
        if self.faiss_index is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search similar chunks
        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Ensure valid index
                metadata = self.chunk_metadata[idx]
                results.append({
                    'content': self.chunks[idx],
                    'score': float(score),
                    'filename': metadata['filename'],
                    'file_type': metadata['file_type'],
                    'doc_idx': metadata['doc_idx'],
                    'chunk_idx': metadata['chunk_idx']
                })
        
        return results
    
    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generate response using OpenAI with retrieved context."""
        if not retrieved_chunks:
            return "I couldn't find relevant information in the documents to answer your question."
        
        # Prepare context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Source {i} - {chunk['filename']}]:\n{chunk['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following document excerpts, please provide a comprehensive answer to the user's question. 
If the information is not sufficient or not present in the provided context, please say so clearly.

Context from documents:
{context}

User Question: {query}

Please provide a detailed answer based on the context above. If you reference specific information, 
mention which source document it came from."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # Using the newest OpenAI model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided document context. "
                        "Always be accurate and cite your sources when possible. If information is not in the context, say so clearly."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=1000
            )
            
            return response.choices[0].message.content or "No response generated"
        
        except Exception as e:
            return f"Error generating response: {str(e)}. Please check your OpenAI API key configuration."
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Main query function that retrieves relevant chunks and generates response."""
        if self.faiss_index is None:
            return {
                "answer": "No documents loaded. Please upload and process documents first.",
                "sources": []
            }
        
        # Retrieve similar chunks
        retrieved_chunks = self.retrieve_similar_chunks(question, top_k)
        
        # Generate response
        answer = self.generate_response(question, retrieved_chunks)
        
        return {
            "answer": answer,
            "sources": retrieved_chunks
        }
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about loaded documents."""
        return {
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunks),
            "num_embeddings": len(self.embeddings) if self.embeddings is not None else 0
        }
