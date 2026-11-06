# RAG-Based-Movie-Recommender-System

A content-based movie recommendation application that uses sentence embeddings and FAISS vector similarity search to retrieve movies
matching a user's preferences. The application provides a simple interactive interface built with Streamlit.

##Features
- Collects movie preferences through an interactive Streamlit interface.
  
- Creates semantic embeddings from each movie's genre and overview.
 
- Uses FAISS with L2 distance search to retrieve the three closest matches.
  
- Stores the movie metadata and vector index for reuse.
  
- Displays recommended movie titles with preference-based explanations.

##Tech Stack

- Python
  
- Streamlit - interactive web interface
  
- Sentence Transformers - text embedding generation

- FAISS - vector similarity search

- Pandas - movie metadata processing

- Pickle - persistence of the processed movie DataFrame

###Project Structure

RAG-Movie-Recommender/

├── app.py

├── data/

│   └── movies_data.csv

├── rag/

│   ├── __init__.py

│   ├── embeddings.py

│   ├── retriever.py

│   └── recommender.py

├── movies.index

├── movies.pkl

├── requirements.txt

└── README.md


###How It Works

1. Prepare movie text: The embedding pipeline combines each movie's Genre and Overview fields.

2. Generate embeddings: The all-MiniLM-L6-v2 Sentence Transformer converts the combined text into dense vectors.

3. Build the FAISS index: The vectors are added to a FAISS IndexFlatL2 index. The index is saved as movies.index, while the movie DataFrame is saved as movies.pkl.

4. Retrieve movies: A user's selected preferences are encoded with the same embedding model. FAISS searches the index and returns the top three nearest movie records.

5. Present recommendations: Streamlit displays the retrieved movietitles and a short explanation based on the submitted preferences.


###Dataset

The embedding script uses the Genre and Overview columns to create searchable text. The recommender uses the Series_Title field to
display movie names. The supplied serialized metadata file contains 1,000 movie records.

Installation
1. Clone the repository:
   git clone <your-repository-url>
   cd RAG-Movie-Recommender
   
3. Create and activate a virtual environment:
   Windows
   
   python -m venv .venv
   .venv\Scripts\activate
   macOS / Linux
   python -m venv .venv
   source .venv/bin/activate
   
5. Install dependencies:
   pip install -r requirements.txt
   
Build the Embeddings and Index
run:
python embeddings.py
This creates:
- movies.index - FAISS vector index
- movies.pkl - processed movie metadata
  
The embedding model is downloaded by Sentence Transformers the first time it is used, so an internet connection may be required on the first
run.

###Run the Application
From the project root, run:
streamlit run app.py



###Future Improvements

- Add free-text preference input alongside the multiple-choice interface.

- Improve recommendation explanations using retrieved movie metadata and an LLM.

- Evaluate retrieval quality with a small test set and suitable ranking metrics.

- Add filters such as release year, genre, and IMDb rating.

- Add error handling for missing or incompatible index and metadata files.


