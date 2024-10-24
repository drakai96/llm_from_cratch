import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from embedding import TFIDF
from utils import make_db, find_importance_word, search, pd_document_to_db

app = FastAPI()

# Load data once at startup to avoid reloading during each request
DATA = pd.read_csv("corpus/db.csv")
CORPUS = pd.read_csv("corpus/corpus.csv")


class GetWordRequest(BaseModel):
    top_k: int = Field(default=10, description="Number of words to find")
    query: str = Field(description="Query to find important words in the document")


class ImportantWordResponse(BaseModel):
    document: str
    score_word: dict


@app.post("/embedding_db")
def embed_documents(n_doc: int = 100):
    """
    Endpoint to update the document embeddings in the database.
    Args:
        n_doc: Number of documents to embed.
    Returns:
        JSON response indicating success.
    """
    make_db(n=n_doc)
    return JSONResponse(status_code=200, content="Database updated with embeddings")


@app.post("/get_important_word")
def get_important_words(input_param: GetWordRequest):
    """
    Endpoint to retrieve the most important words from a document.
    Args:
        input_param: GetWordRequest model containing query string and number of top words.
    Returns:
        JSON response with important words and their scores.
    """
    important_words = find_importance_word(sentence=input_param.query)
    response = {"word": important_words}
    return JSONResponse(status_code=200, content=response)


@app.post("/search_text")
def search_text(input_text: str):
    """
    Endpoint to search for relevant text in the corpus based on input text.
    Args:
        input_text: The text to search in the corpus.
    Returns:
        Relevant text from the corpus.
    """
    relevant_text = search(input_text)
    return JSONResponse(status_code=200, content=relevant_text)


@app.post("/training")
def train_model(number_doc: int = 1000):
    """
    Endpoint to train the model on the provided number of documents.
    Args:
        number_doc: Number of documents to train on.
    Returns:
        JSON response indicating training success.
    """
    make_db(number_doc)
    return JSONResponse(status_code=200, content="Training completed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", reload=True)
