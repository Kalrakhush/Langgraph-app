# tests/test_pdf.py
import pytest
from unittest.mock import Mock, patch, mock_open
from src.services.pdf_service import PDFService

class TestPDFService:
    
    def setup_method(self):
        self.pdf_service = PDFService()
    
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake pdf content')
    @patch('PyPDF2.PdfReader')
    def test_extract_text_from_pdf(self, mock_pdf_reader, mock_file):
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF text content"
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader
        
        result = self.pdf_service.extract_text_from_pdf("test.pdf")
        
        assert "Sample PDF text content" in result
    
    def test_chunk_text(self):
        text = "This is a long text " * 100  # Create long text
        
        chunks = self.pdf_service.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 1000 for chunk in chunks)
    
    def test_chunk_empty_text(self):
        chunks = self.pdf_service.chunk_text("")
        assert chunks == []
    
    @patch.object(PDFService, 'extract_text_from_pdf')
    @patch.object(PDFService, 'create_embeddings')
    def test_process_pdf_complete(self, mock_embeddings, mock_extract):
        # Mock methods
        mock_extract.return_value = "Sample text content for testing"
        mock_embeddings.return_value = [[0.1, 0.2, 0.3]] * 2  # Mock embeddings
        
        result = self.pdf_service.process_pdf("test.pdf")
        
        assert "chunks" in result
        assert "embeddings" in result
        assert "metadata" in result
        assert len(result["chunks"]) > 0