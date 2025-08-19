"""UI utility functions for the RAG PDF Chat App."""

import streamlit as st
from datetime import datetime
import json
from config import config
from utils import create_download_link

def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(**config.get_streamlit_config())

def setup_sidebar() -> tuple[int, int, int]:
    """Create and manage the sidebar configuration."""
    st.sidebar.title("🔧 Configuration")
    
    # API Key Configuration
    st.sidebar.subheader("Gemini API Key")
    api_key = st.sidebar.text_input(
        "Enter your Gemini API key:",
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    # Model Selection
    st.sidebar.subheader("🤖 Gemini Model")
    available_models = config.get("gemini_models")
    selected_model = st.sidebar.selectbox(
        "Choose Gemini Model:",
        available_models,
        index=available_models.index(config.get("default_gemini_model")),
        help="Select the Gemini model for chat responses"
    )
    
    # PDF Processing Options
    st.sidebar.subheader("📄 PDF Processing")
    chunk_settings = config.get_chunk_settings()
    chunk_size = st.sidebar.slider(
        "Chunk Size",
        chunk_settings["chunk_size_min"],
        chunk_settings["chunk_size_max"],
        chunk_settings["chunk_size_default"],
        chunk_settings["chunk_size_step"]
    )
    
    overlap = st.sidebar.slider(
        "Chunk Overlap",
        chunk_settings["overlap_min"],
        chunk_settings["overlap_max"],
        chunk_settings["overlap_default"],
        chunk_settings["overlap_step"]
    )
    
    # Retrieval Settings
    st.sidebar.subheader("🔍 Retrieval Settings")
    retrieval_settings = config.get_retrieval_settings()
    num_results = st.sidebar.slider(
        "Number of chunks to retrieve",
        retrieval_settings["num_results_min"],
        retrieval_settings["num_results_max"],
        retrieval_settings["num_results_default"]
    )
    
    setup_chat_controls()
    
    return api_key, selected_model, chunk_size, overlap, num_results

def setup_chat_controls():
    """Set up chat control buttons in the sidebar."""
    st.sidebar.subheader("💬 Chat Controls")
    
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.sidebar.button("Export Chat History"):
        if st.session_state.chat_history:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_exchanges": len(st.session_state.chat_history),
                "chat_history": st.session_state.chat_history
            }
            
            chat_json = json.dumps(export_data, indent=2)
            filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            create_download_link(
                chat_json,
                filename,
                "application/json",
                "Download Chat History"
            )
        else:
            st.sidebar.warning("No chat history to export")

def show_file_info():
    """Display information about the currently loaded PDF."""
    if st.session_state.current_pdf_name:
        st.info(f"📋 Current PDF: {st.session_state.current_pdf_name}")
        if st.session_state.embedding_manager:
            num_chunks = len(st.session_state.embedding_manager.chunks)
            st.info(f"📊 Chunks: {num_chunks}")

def show_api_configuration_warning():
    """Display warning about missing API configuration."""
    st.warning("⚠️ Please configure your Gemini API key in the sidebar to continue.")
    st.info("💡 Get your free API key from: https://makersuite.google.com/app/apikey")
