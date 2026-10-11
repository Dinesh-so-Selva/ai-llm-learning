import os
from google import genai

# Read the API key from an environment variable
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

# Connect to the Gemini API
client = genai.Client(api_key=api_key)

# Ask the model a question
question = input("Ask a question: ")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=question
)

# Display the answer
print("\nAI Answer:")
print(response.text)