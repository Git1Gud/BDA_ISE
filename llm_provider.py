"""
A module for providing different LLM implementations.
"""
import os
from abc import ABC, abstractmethod
from llama_cpp import Llama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load configuration from llm_config.py
try:
    from llm_config import PROVIDER, MODEL_NAME, MODEL_PATH, MAX_TOKENS, TEMPERATURE
except ImportError:
    # Fallback defaults if config file doesn't exist
    PROVIDER = os.getenv("PROVIDER", "gemini")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-pro")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/llama-2-7b-chat.Q4_K_M.gguf")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

class LangChainLlamaWrapper(LLM):
    """LangChain wrapper for llama-cpp-python."""
    
    model_path: str = MODEL_PATH
    chat_format: str = "llama-2"
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm = Llama(
            model_path=self.model_path,
            chat_format=self.chat_format,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx
        )
    
    @property
    def _llm_type(self) -> str:
        return "llama-cpp"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt."""
        messages = [{"role": "user", "content": prompt}]
        
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop
        )
        return response["choices"][0]["message"]["content"]

class LLMProvider(ABC):
    """Abstract base class for a LLM provider."""
    @abstractmethod
    def create_chat_completion(self, messages, max_tokens):
        """Generate a chat completion."""
        pass

class LlamaProvider(LLMProvider):
    """Provider for the local Llama model."""
    def __init__(self, model_path=MODEL_PATH, chat_format="llama-2", n_gpu_layers=-1, n_ctx=4096):
        self.llm = Llama(
            model_path=model_path,
            chat_format=chat_format,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx
        )

    def create_chat_completion(self, messages, max_tokens=MAX_TOKENS):
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"]

class GeminiProvider(LLMProvider):
    """Provider for Google Gemini using LangChain."""
    def __init__(self, api_key=None, model_name=MODEL_NAME):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=TEMPERATURE,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key,
        )

    def create_chat_completion(self, messages, max_tokens=MAX_TOKENS):
        # Convert messages to LangChain format
        from langchain.schema import HumanMessage, SystemMessage
        
        langchain_messages = []
        for message in messages:
            if message["role"] == "system":
                langchain_messages.append(SystemMessage(content=message["content"]))
            elif message["role"] == "user":
                langchain_messages.append(HumanMessage(content=message["content"]))
        
        # Invoke the model
        response = self.llm.invoke(langchain_messages)
        return response.content

def get_provider(provider_name=None, **kwargs):
    """Factory function to get a LLM provider."""
    if provider_name is None:
        provider_name = PROVIDER.lower()
    
    if provider_name == "llama":
        return LangChainLlamaWrapper(
            model_path=kwargs.get("model_path", MODEL_PATH)
        )
    elif provider_name == "gemini":
        api_key = kwargs.get("api_key")
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model_name", MODEL_NAME),
            temperature=TEMPERATURE,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: llama, gemini")
