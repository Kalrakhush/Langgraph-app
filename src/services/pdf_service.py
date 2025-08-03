# src/services/pdf_service.py
import PyPDF2
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from ..config import Config

class PDFService:
    """Service for processing PDF documents."""
    
    def __init__(self):
        # Use sentence-transformers for embeddings instead of OpenAI
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text
        
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """
        Create embeddings for text chunks using sentence-transformers.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(chunks)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return []
    
    def create_query_embedding(self, query: str) -> List[float]:
        """
        Create embedding for a single query.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embedding_model.encode([query])
            return embedding[0].tolist()
        except Exception as e:
            print(f"Error creating query embedding: {e}")
            return []
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete PDF processing pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with chunks and embeddings
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {"chunks": [], "embeddings": []}
        
        # Create chunks
        chunks = self.chunk_text(text)
        if not chunks:
            return {"chunks": [], "embeddings": []}
        
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        
        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "metadata": {
                "source": pdf_path,
                "total_chunks": len(chunks)
            }
        }