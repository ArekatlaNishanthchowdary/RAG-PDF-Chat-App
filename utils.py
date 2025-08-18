"""Utility functions for the RAG PDF Chat App."""

import re
import json
import hashlib
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st
from config import config

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s.,!?;:()\-\'"\"]', ' ', text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"['']", "'", text)
    
    return text.strip()

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate MD5 hash of file content for caching."""
    return hashlib.md5(file_content).hexdigest()

def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def validate_api_key(api_key: str) -> bool:
    """Basic validation for API key format."""
    if not api_key:
        return False
    
    # Basic checks for Gemini API key format
    if len(api_key) < 20:
        return False
    
    # Should not contain spaces
    if ' ' in api_key:
        return False
    
    return True

def export_chat_history(chat_history: List[Dict], filename_prefix: str = "chat_history") -> str:
    """Export chat history to JSON format."""
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "total_exchanges": len(chat_history),
        "chat_history": []
    }
    
    for exchange in chat_history:
        export_exchange = {
            "user_message": exchange.get("user", ""),
            "assistant_message": exchange.get("assistant", ""),
            "timestamp": exchange.get("timestamp", ""),
            "num_sources": len(exchange.get("sources", []))
        }
        export_data["chat_history"].append(export_exchange)
    
    return json.dumps(export_data, indent=2)

def create_download_link(content: str, filename: str, mime_type: str = "text/plain") -> None:
    """Create a download link in Streamlit."""
    st.download_button(
        label=f"Download {filename}",
        data=content,
        file_name=filename,
        mime=mime_type
    )

def display_metrics(metrics: Dict[str, Any]) -> None:
    """Display metrics in Streamlit columns."""
    cols = st.columns(len(metrics))
    
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(label=key, value=value)

def safe_execute(func, *args, default_return=None, **kwargs):
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"Error in {func.__name__}: {str(e)}")
        return default_return

def validate_gemini_response(response: str) -> Dict[str, Any]:
    """Validate and process Gemini API response."""
    if "Error:" in response:
        return {
            "success": False,
            "error": response.replace("Error:", "").strip(),
            "response": None
        }
    return {
        "success": True,
        "error": None,
        "response": response
    }

def prepare_chat_prompt(query: str, context_chunks: List[Dict], chat_history: List[Dict]) -> str:
    """Prepare a formatted chat prompt with context."""
    # Format context
    context = "\n\n".join([
        f"Source {i+1} (Relevance: {score:.3f}):\n{chunk['text']}"
        for i, (chunk, score) in enumerate(context_chunks)
    ])
    
    # Format recent chat history
    history = ""
    if chat_history:
        recent_history = chat_history[-config.get("max_chat_history"):]
        for msg in recent_history:
            history += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n\n"
    
    # Build prompt template
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
    
    return prompt

class ProgressTracker:
    """Track progress for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
    
    def update(self, step_description: str = ""):
        """Update progress."""
        self.current_step += 1
        progress = self.current_step / self.total_steps
        self.progress_bar.progress(progress)
        
        if step_description:
            self.status_text.text(f"{self.description}: {step_description}")
        else:
            self.status_text.text(f"{self.description}: {self.current_step}/{self.total_steps}")
    
    def complete(self, message: str = "Complete!"):
        """Mark as complete."""
        self.progress_bar.progress(1.0)
        self.status_text.text(message)
    
    def cleanup(self):
        """Clean up progress indicators."""
        self.progress_bar.empty()
        self.status_text.empty()

def initialize_session_state() -> None:
    """Initialize or reset Streamlit session state variables."""
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
        st.session_state.current_model = config.get('default_gemini_model')

def display_chat_interface(chat_container: Any, exchange: Dict[str, Any]) -> None:
    """Display a single chat exchange in the interface."""
    # User message
    with chat_container.chat_message("user"):
        st.write(exchange['user'])
    
    # Assistant message
    with chat_container.chat_message("assistant"):
        st.write(exchange['assistant'])
        
        # Show sources if available
        if 'sources' in exchange:
            with st.expander("📖 View Sources"):
                for j, (chunk, score) in enumerate(exchange['sources']):
                    st.markdown(f"**Source {j+1}** (Relevance: {score:.3f})")
                    st.text(truncate_text(chunk['text'], 200))
                    st.markdown("---")