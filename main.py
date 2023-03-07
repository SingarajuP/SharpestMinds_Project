""" main module of the app"""
import sys
import time
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from src.predict import classify_tfidf, classify_bert
from src.utils import tfidf_lr_model, bert_finetune_model

sys.path.append("src/")

app = FastAPI(
    title="Emotions Classification API",
    description="A simple API that use NLP model to predict the emotion of the books reviews",
    version="0.1",
)
templates = Jinja2Templates(directory="templates/")


@app.get("/")
def read_form():
    """ base form for html"""
    return "please give the link http://127.0.0.1:8000/form"


class request_body(BaseModel):
    """input format"""

    query: str


@app.get("/form")
def form_post(request: Request):
    """ Form to get the title in html"""
    result = "Enter the book title"
    return templates.TemplateResponse(
        "form.html", context={"request": request, "result": result}
    )


# Output resonse format
class Outputresponse(BaseModel):
    """output format"""

    query: str
    error_code: int
    error_messages: List[str]
    run_time_in_secs: float
    predictions_tfidf: dict
    predictions_bert: dict


@app.post("/predict")
def get_title(data: request_body):
    """ Main program to get the prediction"""
    title = data.query
    time_of_req = time.time()
    error_code = 1
    error_messages = []
    output_tfidf = {}
    output_bert = {}
    print("Got the title:", title)
    if not title:
        error_messages.append("Error with title:")
        return Outputresponse(
            query=title,
            run_time_in_secs=(time.time() - time_of_req),
            error_code=error_code,
            error_messages=error_messages,
            predictions_tfidf=output_tfidf,
            predictions_bert=output_bert,
        )

    tfidf_model = tfidf_lr_model()
    output_tfidf = classify_tfidf(title, tfidf_model)
    trainer = bert_finetune_model()
    output_bert = classify_bert(title, trainer)
    error_code = 0
    return Outputresponse(
        query=title,
        run_time_in_secs=(time.time() - time_of_req),
        error_code=error_code,
        error_messages=error_messages,
        predictions_tfidf=output_tfidf,
        predictions_bert=output_bert,
    )


@app.post("/form")
def form_post(request: Request, title: str = Form(...)):
    """ To get result in html"""
    tfidf_model = tfidf_lr_model()
    result = classify_tfidf(title, tfidf_model)
    return templates.TemplateResponse(
        "form.html", context={"request": request, "result": result}
    )
