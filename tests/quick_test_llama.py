from llm_provider import get_provider

# provider_name = "llama"  # or "gemini"
provider_name = "llama"

llm_provider = get_provider(
    provider_name,
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf" # Only for llama
)

messages = [
    {"role": "system", "content": "You are an assistant who creates study material."},
    {
        "role": "user",
        "content": "What is system design."
    }
]

response = llm_provider.create_chat_completion(
    messages=messages,
    max_tokens=2096
)

print(response)