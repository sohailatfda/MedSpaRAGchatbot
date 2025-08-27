import os
import zipfile
import tempfile
from typing import List, Dict, Any
import streamlit as st
from pathlib import Path
import PyPDF2
import docx
from io import BytesIO

class DocumentProcessor:
    """Handles extraction and processing of documents from zip files."""
    
    def __init__(self):
        self.supported_extensions = {'.pdf', '.txt', '.docx'}
    
    def extract_zip(self, zip_file_path: str, extract_to: str) -> List[str]:
        """Extract zip file and return list of extracted file paths."""
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Get list of files in zip
                file_list = zip_ref.namelist()
                
                # Filter for supported file types
                supported_files = [
                    f for f in file_list 
                    if Path(f).suffix.lower() in self.supported_extensions
                    and not f.startswith('__MACOSX/')  # Skip macOS metadata
                    and not Path(f).name.startswith('.')  # Skip hidden files
                ]
                
                if not supported_files:
                    st.warning("No supported documents found in zip file. Supported formats: PDF, TXT, DOCX")
                    return []
                
                # Extract supported files
                for file_name in supported_files:
                    try:
                        zip_ref.extract(file_name, extract_to)
                        extracted_path = os.path.join(extract_to, file_name)
                        if os.path.isfile(extracted_path):
                            extracted_files.append(extracted_path)
                    except Exception as e:
                        st.warning(f"Failed to extract {file_name}: {str(e)}")
                        continue
                
                return extracted_files
                
        except zipfile.BadZipFile:
            st.error("Invalid zip file format.")
            return []
        except Exception as e:
            st.error(f"Error extracting zip file: {str(e)}")
            return []
    
    def read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.warning(f"Error reading PDF {os.path.basename(file_path)}: {str(e)}")
            return ""
    
    def read_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            st.warning(f"Error reading DOCX {os.path.basename(file_path)}: {str(e)}")
            return ""
    
    def read_txt(self, file_path: str) -> str:
        """Read text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read().strip()
        except Exception as e:
            st.warning(f"Error reading TXT {os.path.basename(file_path)}: {str(e)}")
            return ""
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single document and return its content and metadata."""
        file_extension = Path(file_path).suffix.lower()
        filename = os.path.basename(file_path)
        
        # Read content based on file type
        content = ""
        if file_extension == '.pdf':
            content = self.read_pdf(file_path)
        elif file_extension == '.docx':
            content = self.read_docx(file_path)
        elif file_extension == '.txt':
            content = self.read_txt(file_path)
        
        if not content.strip():
            st.warning(f"No content extracted from {filename}")
            return {
                'filename': filename,
                'content': '',
                'file_type': file_extension,
                'word_count': 0
            }
        
        return {
            'filename': filename,
            'content': content,
            'file_type': file_extension,
            'word_count': len(content.split())
        }
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks with specified size and overlap."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                last_period = text.rfind('.', end - 100, end)
                last_newline = text.rfind('\n', end - 100, end)
                
                # Use the latest sentence boundary found
                boundary = max(last_period, last_newline)
                if boundary > start:
                    end = boundary + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def process_documents(self, zip_file_bytes: bytes) -> List[Dict[str, Any]]:
        """Process all documents from zip file bytes and return processed documents."""
        processed_documents = []
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save zip file to temporary location
            zip_path = os.path.join(temp_dir, "uploaded.zip")
            with open(zip_path, 'wb') as f:
                f.write(zip_file_bytes)
            
            # Extract files
            extracted_files = self.extract_zip(zip_path, temp_dir)
            
            if not extracted_files:
                return []
            
            # Process each extracted file
            progress_bar = st.progress(0)
            for i, file_path in enumerate(extracted_files):
                progress_bar.progress((i + 1) / len(extracted_files))
                
                document = self.process_document(file_path)
                if document:
                    # Create chunks for the document
                    chunks = self.chunk_text(document['content'])
                    
                    # Add chunks to the document
                    document['chunks'] = chunks
                    document['num_chunks'] = len(chunks)
                    
                    processed_documents.append(document)
            
            progress_bar.empty()
        
        return processed_documents
