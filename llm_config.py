# LLM Configuration
# This is the single source of configuration for LLM providers
# Change these values to switch between different providers

# Available providers: "gemini", "llama", "groq"
PROVIDER="gemini"

# Model settings
MODEL_NAME="gemini-2.5-flash"  # For Gemini: "gemini-pro", "gemini-1.5-pro", etc.
MODEL_PATH=r"models/llama-2-7b-chat.Q4_K_M.gguf"  # For Llama: path to your model file

# API Keys (set these in your .env file)
# GEMINI_API_KEY=your_gemini_api_key_here
# GROQ_API_KEY=your_groq_api_key_here

# Other settings
MAX_TOKENS=2048
TEMPERATURE=0.7
