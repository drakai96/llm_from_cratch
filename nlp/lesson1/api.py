import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from utils import get_important_word, make_db

app = FastAPI()

data = pd.read_csv("/Users/nguyenvannham/PycharmProjects/llm_from_cratch/nlp/lesson1/corpus/db.csv")
corpus = pd.read_csv("/Users/nguyenvannham/PycharmProjects/llm_from_cratch/nlp/lesson1/corpus/corpus.csv")


class GetWord(BaseModel):
    n: int = Field(default=5, description="Number of word was returned")
    idx: int = Field(description="Index document query")


class ResponseWord(BaseModel):
    document: str
    score_word: dict


@app.post("/embedding_db")
def embedding(n_doc: int = 100):
    """
    Make db when request
    Args:
        n_doc: number of doc to embdding

    Returns:

    """
    make_db(n=n_doc)
    return JSONResponse(status_code=200, content="Update db")


@app.post("/get_important_word")
def get_importance_word(input_param: GetWord):
    """

    Args:
        input_param:
            n: int = Field(default=5, description="Number of word was returned")
            idx: int = Field(description="Index document query")
    Returns: Document and score of importance vocab

    """
    word = get_important_word(idx=input_param.idx, data=data, number_word_return=input_param.n)
    if word:
        word.update({"document": corpus.iloc[input_param.idx, 0]})
    return JSONResponse(status_code=200, content=word)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", reload=True)
