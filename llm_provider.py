"""
A module for providing different LLM implementations.
Adds logging around Llama generations so you can see when queries are generated.
"""
import os
import time
from abc import ABC, abstractmethod
from llama_cpp import Llama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional
from dotenv import load_dotenv
from logger import logger

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
        def _preview(txt: str, limit: int = 300) -> str:
            return txt if len(txt) <= limit else txt[:limit] + "..."

        messages = [{"role": "user", "content": prompt}]

        logger.info("LLM[llama] generation started; input preview: %s", _preview(prompt))
        t0 = time.perf_counter()
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop
        )
        out = response["choices"][0]["message"]["content"]
        dur = time.perf_counter() - t0
        logger.info("LLM[llama] generation finished in %.2fs; output preview: %s", dur, _preview(out))
        return out

class LLMProvider(ABC):
    """Abstract base class for a LLM provider."""
    @abstractmethod
    def create_chat_completion(self, messages, max_tokens):
        """Generate a chat completion."""
        pass

class LlamaProvider(LLMProvider):
    """Provider for the local Llama model."""
    def __init__(self, model_path=MODEL_PATH, chat_format="llama-2", n_gpu_layers=1, n_ctx=4096):
        self.llm = Llama(
            model_path=model_path,
            chat_format=chat_format,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx
        )

    def create_chat_completion(self, messages, max_tokens=MAX_TOKENS):
        # Log a short preview of the last user message
        user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        preview = user_msg if len(user_msg) <= 300 else user_msg[:300] + "..."
        logger.info("LLM[llama] generation started; input preview: %s", preview)

        t0 = time.perf_counter()
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens
        )
        out = response["choices"][0]["message"]["content"]
        dur = time.perf_counter() - t0
        out_preview = out if len(out) <= 300 else out[:300] + "..."
        logger.info("LLM[llama] generation finished in %.2fs; output preview: %s", dur, out_preview)
        return out

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
