import os
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")  # any string works

response = client.chat.completions.create(
    model="openai-gpt-oss-20b-obliterated-uncensored-neo-imatrix",  # <-- put your model ID here
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a poem about the sea."},
    ],
    temperature=0.5,
)

print(response.choices[0].message.content)
