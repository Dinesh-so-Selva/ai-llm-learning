import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load .env from the same folder as this Python file
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

# Read the API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

client = genai.Client(api_key=api_key)

question = input("Ask a question: ")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=question
)

print("\nAI Answer:")
print(response.text)