import os
from dotenv import load_dotenv

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

if gemini_key:
    print("Connected!")
else:
    print("Oops, ERROR")