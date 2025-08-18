"""Configuration settings for the RAG PDF Chat App."""

import os
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    # API Providers
    "available_providers": [
        "OpenAI",
        "Google Gemini",
        "Anthropic Claude",
        "Grok"
    ],
    "default_provider": "Google Gemini",
    
    # Provider-specific Models
    "openai_models": [
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k"
    ],
    "gemini_models": [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash-8b",
        "gemini-2.5-flash",
        "gemini-2.5-pro"
    ],
    "claude_models": [
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-2.1",
        "claude-instant"
    ],
    "grok_models": [
        "grok-1",
        "grok-0.5"
    ],
    
    # Default Models for each provider
    "default_openai_model": "gpt-3.5-turbo",
    "default_gemini_model": "gemini-1.5-pro",
    "default_claude_model": "claude-3-sonnet",
    "default_grok_model": "grok-1",
    
    # Embedding Model
    "embedding_model": "all-MiniLM-L6-v2",
    
    # Text Processing
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_chunks_retrieve": 5,
    
    # Chat Settings
    "max_chat_history": 5,
    "max_context_chunks": 5,
    
    # File Settings
    "supported_file_types": ["pdf"],
    "max_file_size_mb": 10,
    
    # UI Settings
    "page_title": "RAG PDF Chat App",
    "page_icon": "📚",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    
    # Processing Settings
    "chunk_size_min": 500,
    "chunk_size_max": 2000,
    "chunk_size_step": 100,
    "overlap_min": 50,
    "overlap_max": 500,
    "overlap_step": 50,
    "num_results_min": 1,
    "num_results_max": 10,
    "num_results_default": 5
}

class AppConfig:
    """Application configuration manager."""
    
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            "EMBEDDING_MODEL": "embedding_model",
            "CHUNK_SIZE": "chunk_size",
            "CHUNK_OVERLAP": "chunk_overlap",
            "MAX_CHUNKS_RETRIEVE": "max_chunks_retrieve",
            "DEFAULT_PROVIDER": "default_provider",
            "MAX_CHAT_HISTORY": "max_chat_history",
            "MAX_FILE_SIZE": "max_file_size_mb",
            "PAGE_TITLE": "page_title",
            "PAGE_ICON": "page_icon",
            "LAYOUT": "layout"
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert to appropriate type
                if config_key in ["chunk_size", "chunk_overlap", "max_chunks_retrieve", 
                                "max_chat_history", "max_file_size_mb"]:
                    try:
                        self.config[config_key] = int(env_value)
                    except ValueError:
                        pass
                else:
                    self.config[config_key] = env_value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return self.config.copy()
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit-specific configuration."""
        return {
            "page_title": self.get("page_title"),
            "page_icon": self.get("page_icon"),
            "layout": self.get("layout"),
            "initial_sidebar_state": self.get("initial_sidebar_state")
        }
    
    def get_chunk_settings(self) -> Dict[str, int]:
        """Get text chunking settings."""
        return {
            "chunk_size_min": self.get("chunk_size_min"),
            "chunk_size_max": self.get("chunk_size_max"),
            "chunk_size_step": self.get("chunk_size_step"),
            "chunk_size_default": self.get("chunk_size"),
            "overlap_min": self.get("overlap_min"),
            "overlap_max": self.get("overlap_max"),
            "overlap_step": self.get("overlap_step"),
            "overlap_default": self.get("chunk_overlap")
        }
    
    def get_retrieval_settings(self) -> Dict[str, int]:
        """Get retrieval settings."""
        return {
            "num_results_min": self.get("num_results_min"),
            "num_results_max": self.get("num_results_max"),
            "num_results_default": self.get("num_results_default")
        }
    
    def validate_file_size(self, file_size_bytes: int) -> bool:
        """Validate file size against configured maximum."""
        max_size_bytes = self.get("max_file_size_mb") * 1024 * 1024
        return file_size_bytes <= max_size_bytes
    
    def validate_file_type(self, file_extension: str) -> bool:
        """Validate file type against supported types."""
        return file_extension.lower().lstrip('.') in self.get("supported_file_types")
    
    def get_available_providers(self) -> list:
        """Get list of available AI providers."""
        return self.get("available_providers", [])
    
    def get_provider_models(self, provider: str) -> list:
        """Get available models for a specific provider."""
        provider = provider.lower()
        if "gemini" in provider:
            return self.get("gemini_models", [])
        elif "openai" in provider:
            return self.get("openai_models", [])
        elif "claude" in provider:
            return self.get("claude_models", [])
        elif "grok" in provider:
            return self.get("grok_models", [])
        return []
    
    def get_default_model(self, provider: str) -> str:
        """Get default model for a specific provider."""
        provider = provider.lower()
        if "gemini" in provider:
            return self.get("default_gemini_model")
        elif "openai" in provider:
            return self.get("default_openai_model")
        elif "claude" in provider:
            return self.get("default_claude_model")
        elif "grok" in provider:
            return self.get("default_grok_model")
        return None

# Global configuration instance
config = AppConfig()