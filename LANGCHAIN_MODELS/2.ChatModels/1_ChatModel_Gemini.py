from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.8)

response = chat.invoke("Write a 5 line poem on water")

print(response.content)

#Clear and cool, it gently flows,
#A life-giver, where all life grows.
#It takes the shape of any hold,
#A whispered story, ages old.
#Essential, pure, a world untold. (temperature=0)

#Clear and cool, a gentle flow,
#It nourishes all life below.
#From soaring clouds to ocean deep,
#Earth's silent promise it does keep.
#A quench for thirst, a world to steep. (temperature=1.8)

