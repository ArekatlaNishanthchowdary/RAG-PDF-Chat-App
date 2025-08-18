"""API Provider handlers for the RAG PDF Chat App."""

from abc import ABC, abstractmethod
import google.generativeai as genai
import anthropic
import openai
from typing import List, Dict, Any, Optional

class APIProvider(ABC):
    """Abstract base class for API providers."""
    
    @abstractmethod
    def initialize(self, api_key: str) -> bool:
        """Initialize the API client with the provided key."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider."""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, model: str) -> str:
        """Generate a response using the specified model."""
        pass

class OpenAIProvider(APIProvider):
    """OpenAI API provider implementation."""
    
    def initialize(self, api_key: str) -> bool:
        try:
            openai.api_key = api_key
            # Test the API key
            openai.models.list()
            return True
        except Exception as e:
            return False
    
    def get_available_models(self) -> List[str]:
        return [
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
    
    def generate_response(self, prompt: str, model: str) -> str:
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

class GeminiProvider(APIProvider):
    """Google Gemini API provider implementation."""
    
    def initialize(self, api_key: str) -> bool:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            return False
    
    def get_available_models(self) -> List[str]:
        return [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash-8b",
            "gemini-2.5-flash",
            "gemini-2.5-pro"
        ]
    
    def generate_response(self, prompt: str, model: str) -> str:
        try:
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

class ClaudeProvider(APIProvider):
    """Anthropic Claude API provider implementation."""
    
    def initialize(self, api_key: str) -> bool:
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            return True
        except Exception as e:
            return False
    
    def get_available_models(self) -> List[str]:
        return [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-2.1",
            "claude-instant"
        ]
    
    def generate_response(self, prompt: str, model: str) -> str:
        try:
            message = self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content
        except Exception as e:
            return f"Error: {str(e)}"

class GrokProvider(APIProvider):
    """Grok API provider implementation."""
    
    def initialize(self, api_key: str) -> bool:
        # TODO: Implement Grok API when it becomes available
        return False
    
    def get_available_models(self) -> List[str]:
        return [
            "grok-1",
            "grok-0.5"
        ]
    
    def generate_response(self, prompt: str, model: str) -> str:
        return "Grok API is not yet available for public use."

def get_provider(provider_name: str) -> APIProvider:
    """Factory function to get the appropriate provider instance."""
    providers = {
        "OpenAI": OpenAIProvider(),
        "Google Gemini": GeminiProvider(),
        "Anthropic Claude": ClaudeProvider(),
        "Grok": GrokProvider()
    }
    return providers.get(provider_name)

def get_api_input_help(provider: str) -> str:
    """Get provider-specific help text for API key input."""
    help_texts = {
        "OpenAI": "Get your API key from: https://platform.openai.com/api-keys",
        "Google Gemini": "Get your API key from: https://makersuite.google.com/app/apikey",
        "Anthropic Claude": "Get your API key from: https://console.anthropic.com/account/keys",
        "Grok": "Grok API is not yet available for public use."
    }
    return help_texts.get(provider, "Please enter your API key")
