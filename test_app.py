"""Test suite for RAG PDF Chat App components."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import os
from io import BytesIO

# Import our modules
from app import PDFProcessor, TextChunker, EmbeddingManager
from config import AppConfig
from utils import clean_text, validate_api_key, calculate_file_hash

class TestPDFProcessor(unittest.TestCase):
    """Test PDF processing functionality."""
    
    def setUp(self):
        self.processor = PDFProcessor()
    
    def test_extract_text_empty_file(self):
        """Test handling of empty PDF file."""
        # This would require a mock PDF file
        # In a real scenario, you'd create test PDF files
        pass
    
    def test_extract_text_fallback(self):
        """Test fallback mechanism between PDF libraries."""
        # Mock the extraction methods
        with patch.object(PDFProcessor, 'extract_text_pdfplumber', return_value=""):
            with patch.object(PDFProcessor, 'extract_text_pypdf2', return_value="Test content"):
                # This would test the fallback mechanism
                pass

class TestTextChunker(unittest.TestCase):
    """Test text chunking functionality."""
    
    def setUp(self):
        self.chunker = TextChunker(chunk_size=100, overlap=20)
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is a test document. It has multiple sentences. Each sentence should be preserved."
        chunks = self.chunker.chunk_text(text)
        
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]['id'], 0)
        self.assertIn('text', chunks[0])
        self.assertIn('start_pos', chunks[0])
        self.assertIn('end_pos', chunks[0])
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = self.chunker.chunk_text("")
        self.assertEqual(len(chunks), 0)
    
    def test_chunk_text_short(self):
        """Test chunking text shorter than chunk size."""
        text = "Short text."
        chunks = self.chunker.chunk_text(text)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['text'], text)
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        text = "A" * 200  # Long enough to create multiple chunks
        chunks = self.chunker.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that there's overlap between consecutive chunks
            chunk1_end = chunks[0]['end_pos']
            chunk2_start = chunks[1]['start_pos']
            overlap = chunk1_end - chunk2_start
            self.assertGreater(overlap, 0)

class TestEmbeddingManager(unittest.TestCase):
    """Test embedding and vector operations."""
    
    def setUp(self):
        # Use a smaller model for testing or mock it
        self.manager = EmbeddingManager("all-MiniLM-L6-v2")
    
    @patch('app.SentenceTransformer')
    def test_model_loading(self, mock_transformer):
        """Test model loading with mocking."""
        mock_model = Mock()
        mock_transformer.return_value = mock_model
        
        self.manager.initialize()
        self.assertEqual(self.manager.model, mock_model)
    
    def test_create_embeddings_empty(self):
        """Test creating embeddings with empty chunks."""
        with self.assertRaises(ValueError):
            self.manager.create_embeddings([])

class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AppConfig()
        
        self.assertEqual(config.get("chunk_size"), 1000)
        self.assertEqual(config.get("chunk_overlap"), 200)
        self.assertIn("all-MiniLM-L6-v2", config.get("embedding_model"))
    
    def test_config_get_set(self):
        """Test getting and setting configuration values."""
        config = AppConfig()
        
        config.set("test_key", "test_value")
        self.assertEqual(config.get("test_key"), "test_value")
    
    @patch.dict(os.environ, {'CHUNK_SIZE': '500'})
    def test_env_loading(self):
        """Test loading configuration from environment variables."""
        config = AppConfig()
        self.assertEqual(config.get("chunk_size"), 500)

class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "  This   is   a    messy   text!  "
        clean = clean_text(dirty_text)
        self.assertEqual(clean, "This is a messy text!")
    
    def test_validate_api_key_valid(self):
        """Test API key validation with valid key."""
        valid_key = "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567"
        self.assertTrue(validate_api_key(valid_key))
    
    def test_validate_api_key_invalid(self):
        """Test API key validation with invalid keys."""
        self.assertFalse(validate_api_key(""))
        self.assertFalse(validate_api_key("short"))
        self.assertFalse(validate_api_key("has spaces in it"))
    
    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        content1 = b"test content"
        content2 = b"test content"
        content3 = b"different content"
        
        hash1 = calculate_file_hash(content1)
        hash2 = calculate_file_hash(content2)
        hash3 = calculate_file_hash(content3)
        
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)

if __name__ == '__main__':
    unittest.main()