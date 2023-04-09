""" constants and urls"""
import pandas as pd

emotion = pd.read_csv("./labels_prediction/emotions.csv")
dic_emotions = emotion.to_dict("series")
BASE_URL = "http://goodreads.com"
BOOK_URL = "https://www.goodreads.com/search"
MAX_RETRY = 10
