"""UI utility functions for the RAG PDF Chat App."""

import streamlit as st
from datetime import datetime
import json
from config import config
from utils import create_download_link

def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(**config.get_streamlit_config())

def setup_sidebar() -> tuple[str, str, str, int, int, int]:
    """Create and manage the sidebar configuration.
    Returns:
        tuple[str, str, str, int, int, int]: (provider, api_key, model, chunk_size, overlap, num_results)
    """
    st.sidebar.title("🔧 Configuration")
    
    # API Provider Selection
    st.sidebar.subheader("🤖 AI Provider")
    available_providers = config.get_available_providers()
    default_provider = config.get("default_provider", "OpenAI")
    
    # Ensure default provider is in available providers
    if default_provider not in available_providers:
        default_provider = available_providers[0] if available_providers else "OpenAI"
    
    selected_provider = st.sidebar.selectbox(
        "Choose AI Provider:",
        available_providers,
        index=available_providers.index(default_provider),
        help="Select the AI provider for chat responses"
    )
    
    # API Key Configuration
    st.sidebar.subheader("🔑 API Key")
    from api_providers import get_api_input_help
    api_key = st.sidebar.text_input(
        f"Enter your {selected_provider} API key:",
        type="password",
        help=get_api_input_help(selected_provider)
    )
    
    # Model Selection based on provider
    st.sidebar.subheader("📝 Model Selection")
    available_models = config.get_provider_models(selected_provider)
    default_model = config.get_default_model(selected_provider)
    
    if not available_models:
        st.sidebar.error(f"No models available for {selected_provider}")
        selected_model = None
    else:
        selected_model = st.sidebar.selectbox(
            f"Choose {selected_provider} Model:",
            available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            help=f"Select the {selected_provider} model for chat responses"
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
    
    return selected_provider, api_key, selected_model, chunk_size, overlap, num_results

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

def show_api_configuration_warning(provider: str = None):
    """Display warning about missing API configuration."""
    provider = provider or "the selected provider"
    st.warning(f"⚠️ Please configure your {provider} API key in the sidebar to continue.")
    
    # Show provider-specific API key link
    if "gemini" in provider.lower():
        st.info("💡 Get your API key from: https://makersuite.google.com/app/apikey")
    elif "openai" in provider.lower():
        st.info("💡 Get your API key from: https://platform.openai.com/api-keys")
    elif "claude" in provider.lower():
        st.info("💡 Get your API key from: https://console.anthropic.com/settings/keys")
    elif "grok" in provider.lower():
        st.info("💡 Get your API key from: https://x.ai/api")
    else:
        st.info("💡 Please visit the provider's website to get your API key.")
