# LLM Provider Configuration

This project supports multiple LLM providers. You can easily switch between them by editing the `llm_config.py` file.

## Supported Providers

1. **Gemini** (Google's AI model)
2. **Llama** (Local model using llama-cpp-python)
3. **Groq** (via LangChain)

## Configuration

Edit `llm_config.py` to change your provider:

```python
# LLM Configuration
PROVIDER="gemini"  # Change this to "llama" or "groq"

# Model settings
MODEL_NAME="gemini-pro"  # For Gemini
MODEL_PATH="models/llama-2-7b-chat.Q4_K_M.gguf"  # For Llama

# Other settings
MAX_TOKENS=2048
TEMPERATURE=0.7
```

## Environment Variables

Set these in your `.env` file:

```bash
# For Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# For Groq
GROQ_API_KEY=your_groq_api_key_here

# For Qdrant (vector database)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Usage

### In Streamlit App
The Streamlit app will automatically use the provider specified in `llm_config.py`. You can also override it using the radio buttons in the sidebar.

### In Code
```python
from llm_provider import get_provider

# Uses the provider from llm_config.py
llm = get_provider()

# Or specify a provider
llm = get_provider("gemini")
```

## Switching Providers

To switch providers:

1. Edit `llm_config.py` and change the `PROVIDER` variable
2. Set the appropriate API keys in your `.env` file
3. For Llama, ensure your model file exists at the specified `MODEL_PATH`
4. Restart your application

## Troubleshooting

- **Gemini errors**: Check your `GEMINI_API_KEY` in `.env`
- **Llama errors**: Ensure the model file exists at `MODEL_PATH`
- **Groq errors**: Check your `GROQ_API_KEY` in `.env`
- **Import errors**: Make sure all required packages are installed:
  ```bash
  pip install llama-cpp-python langchain-google-genai langchain-groq
  ```
