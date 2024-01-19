# main.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import wordcloud
import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS

app = FastAPI()

# Mount a directory to serve static files (if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use Jinja2 templates for HTML rendering
templates = Jinja2Templates(directory="templates")


def load_data(file_path):
    return pd.read_csv(file_path)


def clean_data(corpus):
    # Remove HTML elements
    corpus['clean_description'] = corpus['description'].str.replace(r"<[a-z/]+>", " ")
    # Remove special characters and numbers
    corpus['clean_description'] = corpus['clean_description'].str.replace(r"[^A-Za-z]+", " ")
    # Lowercase
    corpus['clean_description'] = corpus['clean_description'].str.lower()

    # Tokenize the cleaned description
    nlp = spacy.load('en_core_web_sm')
    corpus['clean_tokens'] = corpus['clean_description'].apply(lambda x: nlp(x))

    # Remove stop words
    corpus['clean_tokens'] = corpus['clean_tokens'].apply(lambda x: [token.lemma_ for token in x if token.text not in STOP_WORDS])

    # Put back tokens into one single string
    corpus["clean_document"] = [" ".join(x) for x in corpus['clean_tokens']]


def vectorize_text(corpus):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus["clean_document"])
    X = X.toarray()
    X_df = pd.DataFrame(X, columns=vectorizer.get_feature_names_out(), index=["item_{}".format(x) for x in range(corpus.shape[0])])
    return X_df


def cluster_documents(X_df, corpus):
    clustering = DBSCAN(eps=0.7, min_samples=3, metric="cosine", algorithm="brute")
    clustering.fit(X_df)
    corpus['cluster_id'] = clustering.labels_
    X_df['cluster_id'] = clustering.labels_


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # Additional FastAPI route (can be modified according to your needs)
    return templates.TemplateResponse("index.html", {"request": request})


def main():
    # Load data
    file_path = 'C:/Users/ishs4/Desktop/promotheo/sample-data.csv'
    corpus = load_data(file_path)

    # Clean data
    clean_data(corpus)

    # Vectorize text
    X_df = vectorize_text(corpus)

    # Cluster documents
    cluster_documents(X_df, corpus)

    # Display or print results
    print(corpus.head())
    print(corpus['cluster_id'].value_counts())

    # Additional analysis or visualization code can be added here

if __name__ == "__main__":
    main()
