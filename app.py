import streamlit as st
import google.generativeai as genai
import PyPDF2
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import io
import traceback
from config import config

# Configure Streamlit page
st.set_page_config(
    page_title="RAG PDF Chat App",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PDFProcessor:
    """Handle PDF text extraction with multiple methods for reliability."""
    
    @staticmethod
    def extract_text_pypdf2(pdf_file) -> str:
        """Extract text using PyPDF2."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"PyPDF2 extraction failed: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_pdfplumber(pdf_file) -> str:
        """Extract text using pdfplumber (more reliable for complex PDFs)."""
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            st.error(f"pdfplumber extraction failed: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text(pdf_file) -> str:
        """Extract text with fallback methods."""
        # Reset file pointer
        pdf_file.seek(0)
        
        # Try pdfplumber first (more reliable)
        text = PDFProcessor.extract_text_pdfplumber(pdf_file)
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not text.strip():
            pdf_file.seek(0)
            text = PDFProcessor.extract_text_pypdf2(pdf_file)
        
        # Clean and validate text
        text = text.strip()
        if not text:
            raise ValueError("No text could be extracted from the PDF")
        
        return text

class TextChunker:
    """Handle intelligent text chunking with overlap."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """Split text into overlapping chunks with metadata."""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Define chunk boundaries
            end = start + self.chunk_size
            
            # If we're not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 chars
                last_period = text.rfind('.', end - 100, end)
                last_question = text.rfind('?', end - 100, end)
                last_exclamation = text.rfind('!', end - 100, end)
                
                sentence_end = max(last_period, last_question, last_exclamation)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'length': len(chunk_text)
                })
                chunk_id += 1
            
            # Move start position (with overlap)
            start = end - self.overlap
            
            # Avoid infinite loop
            if start >= end:
                break
        
        return chunks

class EmbeddingManager:
    """Handle text embeddings and vector operations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
    
    @st.cache_resource
    def load_model(_self):
        """Load sentence transformer model (cached)."""
        try:
            return SentenceTransformer(_self.model_name)
        except Exception as e:
            st.error(f"Failed to load embedding model: {str(e)}")
            return None
    
    def initialize(self):
        """Initialize the embedding model."""
        self.model = self.load_model()
        if self.model is None:
            raise ValueError("Failed to initialize embedding model")
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        if not self.model:
            raise ValueError("Model not initialized")
        
        texts = [chunk['text'] for chunk in chunks]
        
        with st.spinner("Generating embeddings..."):
            embeddings = self.model.encode(texts, show_progress_bar=True)
        
        return embeddings
    
    def build_index(self, chunks: List[Dict]):
        """Build FAISS index from chunks."""
        self.chunks = chunks
        embeddings = self.create_embeddings(chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        st.success(f"Built FAISS index with {len(chunks)} chunks")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar chunks."""
        if not self.model or not self.index:
            raise ValueError("Model or index not initialized")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results

class GeminiChat:
    """Handle Gemini API interactions."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.initialize()
    
    def initialize(self, model_name="gemini-1.5-flash"):
        """Initialize Gemini model."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {str(e)}")
            self.model = None
    
    def generate_response(self, query: str, context_chunks: List[Tuple[Dict, float]], 
                         chat_history: List[Dict]) -> str:
        """Generate response using retrieved context."""
        if not self.model:
            raise ValueError("Gemini model not initialized")
        
        # Build context from retrieved chunks
        context = "\n\n".join([
            f"Source {i+1} (Relevance: {score:.3f}):\n{chunk['text']}"
            for i, (chunk, score) in enumerate(context_chunks)
        ])
        
        # Build conversation history
        history = ""
        if chat_history:
            recent_history = chat_history[-5:]  # Last 5 exchanges
            for msg in recent_history:
                history += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
        
        # Create comprehensive prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided PDF document context. 

CONTEXT FROM DOCUMENT:
{context}

CONVERSATION HISTORY:
{history}

CURRENT QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based primarily on the provided context
2. If the context doesn't contain sufficient information, clearly state this
3. Be specific and cite relevant parts of the context when possible
4. Maintain conversation continuity when appropriate
5. If asked about previous parts of the conversation, refer to the conversation history

RESPONSE:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg.upper():
                return "Error: Invalid API key. Please check your Gemini API key."
            elif "QUOTA" in error_msg.upper() or "LIMIT" in error_msg.upper():
                return "Error: API quota exceeded. Please try again later."
            else:
                return f"Error generating response: {error_msg}"

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = None
    if 'gemini_chat' not in st.session_state:
        st.session_state.gemini_chat = None
    if 'current_pdf_name' not in st.session_state:
        st.session_state.current_pdf_name = None
    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = False
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None

def sidebar():
    """Create sidebar for configuration and controls."""
    st.sidebar.title("🔧 Configuration")
    
    # API Key Configuration
    st.sidebar.subheader("Gemini API Key")
    
    api_key = st.sidebar.text_input(
        "Enter your Gemini API key:",
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    # Gemini Model Selection
    st.sidebar.subheader("🤖 Gemini Model")
    available_models = config.get("gemini_models")
    default_model = config.get("default_gemini_model")
    
    selected_model = st.sidebar.selectbox(
        "Choose Gemini Model:",
        available_models,
        index=available_models.index(default_model),
        help="Select the Gemini model for chat responses"
    )
    
    if api_key:
        # Check if model or API key changed
        if (not st.session_state.api_key_configured or 
            st.session_state.gemini_chat is None or
            getattr(st.session_state, 'current_model', None) != selected_model):
            try:
                st.session_state.gemini_chat = GeminiChat(api_key)
                st.session_state.gemini_chat.initialize(selected_model)
                if st.session_state.gemini_chat.model:
                    st.session_state.api_key_configured = True
                    st.session_state.current_model = selected_model
                    st.sidebar.success(f"✅ {selected_model} configured successfully!")
                else:
                    st.sidebar.error("❌ Failed to configure model")
            except Exception as e:
                st.sidebar.error(f"❌ API Key/Model error: {str(e)}")
    
    # PDF Processing Options
    st.sidebar.subheader("📄 PDF Processing")
    chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100)
    overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200, 50)
    
    # Retrieval Options
    st.sidebar.subheader("🔍 Retrieval Settings")
    num_results = st.sidebar.slider("Number of chunks to retrieve", 1, 10, 5)
    
    # Chat Controls
    st.sidebar.subheader("💬 Chat Controls")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.sidebar.button("Export Chat History"):
        if st.session_state.chat_history:
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.sidebar.download_button(
                "Download Chat History",
                chat_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.sidebar.warning("No chat history to export")
    
    return chunk_size, overlap, num_results

def main():
    """Main application function."""
    st.title("📚 RAG PDF Chat App with Gemini")
    st.markdown("Upload a PDF document and chat with its content using Google's Gemini AI!")
    
    initialize_session_state()
    chunk_size, overlap, num_results = sidebar()
    
    # Check API key
    if not st.session_state.api_key_configured:
        st.warning("⚠️ Please configure your Gemini API key in the sidebar to continue.")
        st.info("💡 Get your free API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with its content"
        )
        
        if uploaded_file is not None:
            # Check if we need to reprocess
            if (st.session_state.current_pdf_name != uploaded_file.name or 
                not st.session_state.pdf_processed):
                
                with st.spinner("Processing PDF..."):
                    try:
                        # Extract text
                        text = PDFProcessor.extract_text(uploaded_file)
                        st.success(f"✅ Extracted {len(text)} characters")
                        
                        # Chunk text
                        chunker = TextChunker(chunk_size, overlap)
                        chunks = chunker.chunk_text(text)
                        st.success(f"✅ Created {len(chunks)} chunks")
                        
                        # Initialize embedding manager
                        embedding_manager = EmbeddingManager()
                        embedding_manager.initialize()
                        
                        # Build embeddings and index
                        embedding_manager.build_index(chunks)
                        
                        # Update session state
                        st.session_state.embedding_manager = embedding_manager
                        st.session_state.pdf_processed = True
                        st.session_state.current_pdf_name = uploaded_file.name
                        st.session_state.chat_history = []  # Clear previous chats
                        
                        st.success("🎉 PDF processed successfully! You can now start chatting.")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
                        st.error("Please try a different PDF or check the file format.")
                        return
            
            # Show PDF info
            if st.session_state.pdf_processed:
                st.info(f"📋 Current PDF: {uploaded_file.name}")
                if st.session_state.embedding_manager:
                    num_chunks = len(st.session_state.embedding_manager.chunks)
                    st.info(f"📊 Chunks: {num_chunks}")
    
    with col2:
        st.subheader("💬 Chat Interface")
        
        if not st.session_state.pdf_processed:
            st.info("👆 Upload a PDF document to start chatting!")
            return
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for i, exchange in enumerate(st.session_state.chat_history):
                # User message
                with st.chat_message("user"):
                    st.write(exchange['user'])
                
                # Assistant message
                with st.chat_message("assistant"):
                    st.write(exchange['assistant'])
                    
                    # Show sources if available
                    if 'sources' in exchange:
                        with st.expander("📖 View Sources"):
                            for j, (chunk, score) in enumerate(exchange['sources']):
                                st.markdown(f"**Source {j+1}** (Relevance: {score:.3f})")
                                st.text(chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'])
                                st.markdown("---")
        
        # Chat input
        user_input = st.chat_input("Ask a question about your PDF...")
        
        if user_input:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Retrieve relevant chunks
                        results = st.session_state.embedding_manager.search(user_input, num_results)
                        
                        if not results:
                            response = "I couldn't find relevant information in the document to answer your question."
                            sources = []
                        else:
                            # Generate response using Gemini
                            response = st.session_state.gemini_chat.generate_response(
                                user_input, results, st.session_state.chat_history
                            )
                            sources = results
                        
                        # Display response
                        st.write(response)
                        
                        # Show sources
                        if sources:
                            with st.expander("📖 View Sources"):
                                for j, (chunk, score) in enumerate(sources):
                                    st.markdown(f"**Source {j+1}** (Relevance: {score:.3f})")
                                    st.text(chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'])
                                    st.markdown("---")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'user': user_input,
                            'assistant': response,
                            'sources': sources,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        error_response = f"❌ Error: {str(e)}"
                        st.error(error_response)
                        
                        # Add error to chat history
                        st.session_state.chat_history.append({
                            'user': user_input,
                            'assistant': error_response,
                            'sources': [],
                            'timestamp': datetime.now().isoformat()
                        })

if __name__ == "__main__":
    main()