from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load the API key from .env
load_dotenv()

# Create Gemini embedding model
# The API key is automatically picked up from the environment
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=300
)

# Documents that we want to search through
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# The question/query given by the user
query = "tell me about bumrah"

# Convert all documents into 300-dimensional vectors
doc_embeddings = embedding.embed_documents(documents)

# Convert the query into a 300-dimensional vector
query_embedding = embedding.embed_query(query)

# Compare the query vector with every document vector
# Higher score means greater semantic similarity
scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

# Find the document with the highest similarity score
index, score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]

# Print the query
print(query)

# Print the most similar document
print(documents[index])

# Print how similar the query and document are
print("Similarity score is:", score)