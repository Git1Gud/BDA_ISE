from llama_cpp import Llama
llm = Llama(
      model_path="models\llama-2-7b-chat.Q4_K_M.gguf",
      chat_format="llama-2",
      n_gpu_layers=-1,
      n_ctx=4096
)

response = llm.create_chat_completion(
      messages = [
          {"role": "system", "content": "You are an assistant who creates study material."},
          {
              "role": "user",
              "content": "Give me a mermiad diagram code of a simple client and server architecture."
          }
      ],
      max_tokens=512  # Increase this value for longer answers
)
print(response["choices"][0]["message"]["content"])