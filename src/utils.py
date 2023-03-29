''' functions to load files'''
import pandas as pd
import pickle
from langdetect import detect
from transformers import AutoTokenizer
from transformers import Trainer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification

def load_data()->pd.DataFrame:
    df = pd.read_pickle('./data/raw/emotions_training.pkl')
    return df
def detect_en(text)->bool:
    """This function takes text as input and returns the text if the
    language is english"""
    try:
        return detect(text) == "en"
    except:
        return False


def tfidf_vector()->TfidfVectorizer:
    """loading the tfidf vectors"""
    tfidf_vectorizer = pickle.load(open("./tfidfvectors/tfidf_vect_clean.pkl", "rb"))
    return tfidf_vectorizer


def tfidf_lr_model():
    """loading the logistic regression model"""
    tfidf_model = pickle.load(open("./models/lr_mn_clean_cw.pkl", "rb"))
    return tfidf_model


def bert_finetune_model()->Trainer:
    """loading the bert finetuned model"""
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        "./models/bert_finetuned_model2"
    )
    trainer = Trainer(bert_model)
    return trainer


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def labels(predicted_results:list)->list:
    """predicting labels from the predictions from the bert model"""
    predicted_labels = predicted_results.predictions.argmax(-1)
    predicted_labels = predicted_labels.flatten().tolist()
    return predicted_labels
