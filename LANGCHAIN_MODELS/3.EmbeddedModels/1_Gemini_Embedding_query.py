from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=32
)

result = embedding.embed_query("Delhi is the capital of India")

#[-0.017813325, -0.026862552, -0.010524046, -0.04956995, -0.019413924, -0.0052635353, 0.0023144784, 
# 0.016923562, -0.011439316, -0.014441509, -0.029422855, -0.0072889524, -0.00046775318, 0.045691412, 
# 0.11118747, -0.002808599, 0.007386223, 0.00437319, 0.0032116529, -0.032878906, -0.025440896, -0.004567422, 
# 0.03580175, -0.006551662, -0.0031417941, 0.01964157, -0.008390214, 0.015124783, 0.018800875, 0.0030167722, 
# -0.002450314, -0.034173224]

print(result)


