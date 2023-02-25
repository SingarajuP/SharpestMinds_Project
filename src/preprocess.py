from bs4 import BeautifulSoup
import requests  
import numpy as np
import pandas as pd
from langdetect import detect
import re
from string import punctuation 
import nltk
import nltk.data
from nltk.stem import WordNetLemmatizer


#getting reviews from the book title as a dataframe

def get_reviews(title):
    print("started the loop to get the reviews")
    data = {'q': title}
    book_url = "https://www.goodreads.com/search"
    req = requests.get(book_url, params=data)

    book_soup = BeautifulSoup(req.text, 'html.parser')

    titles=book_soup.find_all('a', class_ = 'bookTitle')
    title=[]
    link=[]
    for bookname in titles:
        title.append(bookname.get_text())
        link.append(bookname['href'])
        rev="http://goodreads.com"+link[0]
        rev_url = requests.get(rev)
        rev_soup=BeautifulSoup(rev_url.content, 'html.parser')
    rev_list=[]
    for x in rev_soup.find_all("section", {"class": "ReviewText"}):
        rev_list.append(x.text)
    df=pd.DataFrame(rev_list, columns=['reviews'])
    print("got the reviews and leaving the loop")
    return df
# selecting the reviews in english language

def detect_en(text):
    try:
        return detect(text) == 'en'
    except:
        return False
    
#cleaning the text
def text_cleaning(text):
   
    text=re.sub("\(.*?\)","",text)

    text = re.sub(r"[^A-Za-z]", " ", str(text))
    
     #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    stopwords = nltk.corpus.stopwords.words('english')
    text = text.split()
    text = [w for w in text if not w in stopwords]
    text = " ".join(text)
        
    text = text.split()
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(lemmatized_words)
    text=text.lower()
    print("cleaned the text in the reviews")
    return text