"""
A module for providing different LLM implementations.
"""
import os
from abc import ABC, abstractmethod
from llama_cpp import Llama
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(ABC):
    """Abstract base class for a LLM provider."""
    @abstractmethod
    def create_chat_completion(self, messages, max_tokens):
        """Generate a chat completion."""
        pass

class LlamaProvider(LLMProvider):
    """Provider for the local Llama model."""
    def __init__(self, model_path, chat_format="llama-2", n_gpu_layers=-1, n_ctx=4096):
        self.llm = Llama(
            model_path=model_path,
            chat_format=chat_format,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx
        )

    def create_chat_completion(self, messages, max_tokens):
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"]

class GeminiProvider(LLMProvider):
    """Provider for Google Gemini."""
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def create_chat_completion(self, messages, max_tokens):
        # Gemini uses a different message format, so we need to adapt it.
        # This is a simplified adaptation.
        # It assumes the last message is from the user and concatenates previous messages.
        
        # Find the system prompt
        system_prompt = ""
        user_prompts = []
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            else:
                user_prompts.append(message["content"])

        full_prompt = system_prompt + "\n" + "\n".join(user_prompts)

        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens
            )
        )
        return response.text

def get_provider(provider_name, **kwargs):
    """Factory function to get a LLM provider."""
    if provider_name == "llama":
        return LlamaProvider(
            model_path=kwargs.get("model_path", "models/llama-2-7b-chat.Q4_K_M.gguf")
        )
    elif provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        return GeminiProvider(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
