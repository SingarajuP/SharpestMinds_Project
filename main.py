from fastapi import FastAPI
import uvicorn
import sys
import time
from pydantic import BaseModel
from typing import List


sys.path.append("src/")
from src.predict import classify

app = FastAPI(
    title="Emotions Classification API",
    description="A simple API that use NLP model to predict the emotion of the books reviews",
    version="0.1",
)

class request_body(BaseModel):
    query : str

#Output resonse format
class Outputresponse(BaseModel):
   # query: str
  #  error_code: int
  #  error_messages: List[str]
 #   run_time_in_secs: float
  #  api_version: int
  #  source: str
    predictions: dict

@app.post("/predict-emotions_review for the title")

def get_title(data : request_body):
    title = [data.query]
   # time_of_req=time.time()
   # error_code=1
    #error_messages=[]
    output={}
    print("Got the title:", title)
    #if (not (title)):
        #error_messages.append("Error with title:")
       # return Outputresponse(query= title, error_code=error_code,error_messages=error_messages,run_time_in_secs=(time.time() - time_of_req),predictions=output, api_version=1, source="xgb model")

    output=classify(title)
    print("Before output response:", output)
   # return Outputresponse(query= title, error_code=error_code,error_messages=error_messages,run_time_in_secs=(time.time() - time_of_req),predictions=output, api_version=1, source="tfidf logistic regression model")
    return Outputresponse(predictions=output)
