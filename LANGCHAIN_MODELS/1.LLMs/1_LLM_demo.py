from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


response = llm.invoke("What is the capital of India")

print(response.content)

# What is the capital of India

# Running this file: python 1_LLM_demo.py
