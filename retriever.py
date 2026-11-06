import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("movies.index")

with open("movies.pkl", "rb") as f:
    movies = pickle.load(f)

def retrieve_movies(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return movies.iloc[indices[0]]
