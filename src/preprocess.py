""" preprocessing for prediction"""
import re
import time
from string import punctuation
from bs4 import BeautifulSoup
import requests
import pandas as pd
import datasets
import nltk
import nltk.data
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
from src.config import BASE_URL, BOOK_URL,MAX_RETRY
from src.utils import tokenizer
from typing import Tuple

def get_reviews(title : str) -> Tuple[str, pd.DataFrame]:
    """getting reviews from the book title as a dataframe"""
    i = 0
    
    data = {"q": title}
    GOODREADS_URL = BOOK_URL
    while i <= MAX_RETRY:    
        req = requests.get(GOODREADS_URL, params=data, timeout=60)
        book_soup = BeautifulSoup(req.text, "html.parser")

        titles = book_soup.find_all("a", class_="bookTitle")
        title = []
        link = []
        for bookname in titles:
            title.append(bookname.get_text())
            link.append(bookname["href"])
        first_book = title[0]
        rev = BASE_URL + link[0]
        rev_url = requests.get(rev, timeout=200)
        rev_soup = BeautifulSoup(rev_url.content, "html.parser")
        rev_list = []
        for x in rev_soup.find_all("section", {"class": "ReviewText"}):
            rev_list.append(x.text)
        df = pd.DataFrame(rev_list, columns=["reviews"])
        if len(df)>0:
            return first_book,df
        else:
            i += 1
    return first_book, df


def text_cleaning(text : str) -> str:
    """This function will change the text to lower case, remove tags,
    special characters,digits, punctuation and lemmatization"""

    text = re.sub(r"\(.*?\)", "", text)

    text = re.sub(r"[^A-Za-z]", " ", str(text))

    text = re.sub(r"&lt;/?.*?&gt;", " &lt;&gt; ", text)
    text = re.sub(r"(\\d|\\W)+", " ", text)
    text = "".join([c for c in text if c not in punctuation])
    stopwords = nltk.corpus.stopwords.words("english")
    text = text.split()
    text = [w for w in text if not w in stopwords]
    text = " ".join(text)

    text = text.split()
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmatized_words)
    text = text.lower()
    return text

def dataset_dict_bert(data : pd.DataFrame) -> dict:
    """This function returns the dataset dictionary format required for BERT model to the input data"""
    dataset = datasets.Dataset.from_dict(data)
    return datasets.DatasetDict({"test": dataset})

def tokenize_data(example : pd.DataFrame) -> AutoTokenizer:
    """This function returns the tokenized vectors for bert model"""

    return tokenizer(example["reviews"], truncation=True, padding="max_length")
