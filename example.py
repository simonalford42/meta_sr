from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client.responses.create(
    model="gpt-5-nano",
    input="Write a one-sentence bedtime story about a unicorn."
)
print(response.output_text)

# completion = client.chat.completions.create(
#   model="openai/gpt-5-nano",
#   messages=[
#     {
#       "role": "user",
#       "content": "Write a one-sentence bedtime story about a unicorn."
#     }
#   ]
# )
# print(completion.choices[0].message.content)
