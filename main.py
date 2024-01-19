import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from fastapi import FastAPI

app = FastAPI()

origins = [
    "http://127.0.0.1:8000",
    "http://localhost:5173",
    # "https://workshala-navy.vercel.app",
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your product data
# Make sure to replace 'path/to/your/product_data.csv' with the actual path to your product data CSV file
corpus = pd.read_csv('path/to/your/product_data.csv')

# Preprocess your product data
corpus['clean_description'] = corpus['description'].str.replace(r"<[a-z/]+>", " ") 
corpus['clean_description'] = corpus['clean_description'].str.replace(r"[^A-Za-z]+", " ") 
corpus['clean_description'] = corpus['clean_description'].str.lower()
corpus["clean_tokens"] = corpus["clean_description"].apply(lambda x: x.split())
corpus["clean_document"] = [" ".join(x) for x in corpus['clean_tokens']]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus["clean_document"])

# Perform clustering using DBSCAN
clustering = DBSCAN(eps=0.7, min_samples=3, metric="cosine", algorithm="brute")
clustering.fit(X)

# Assign cluster labels to the corpus
corpus['cluster_id'] = clustering.labels_

# Function to get product recommendations
def get_product_recommendations(product_description: str):
    # Preprocess the input description
    clean_description = preprocess_description(product_description)

    # Vectorize the input description
    new_vector = vectorizer.transform([clean_description])

    # Predict cluster for the input description
    cluster_id = clustering.predict(new_vector)[0]

    # Get product recommendations from the same cluster
    recommendations = corpus.loc[corpus['cluster_id'] == cluster_id, 'product_name'].tolist()

    return recommendations

def preprocess_description(description: str):
    # Implement your description preprocessing logic here (similar to what you did before)
    description = description.replace(r"<[a-z/]+>", " ") 
    description = description.replace(r"[^A-Za-z]+", " ") 
    description = description.lower()
    tokens = description.split()
    clean_description = " ".join(tokens)
    return clean_description

@app.get("/product-recommendations/{product_description}")
def recommend_products(product_description: str):
    recommendations = get_product_recommendations(product_description)
    return {"product_recommendations": recommendations}
