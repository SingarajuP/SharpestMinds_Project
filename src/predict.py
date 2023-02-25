import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import get_reviews,detect_en,text_cleaning




tfidf_vectorizer=pickle.load(open('./tfidfvectors/tfidf_vect_clean.pkl','rb'))
model=pickle.load(open('./models/lr_mn_clean_cw.pkl','rb'))
emotion = pd.read_csv('./labels_prediction/emotions.csv')
dic_emotions=emotion.to_dict('series')

def classify(title):
    print("I am classifying the reviews for title:",title)
   
    data=get_reviews(title)
    print("Got the reviews and will check for the language")
    data=data[data['reviews'].apply(detect_en)]
    data['cleaned_review'] = data['reviews'].apply(lambda x: text_cleaning(x))
    data = data[data['cleaned_review'].map(len) > 0]    
    print("cleaned process complete")
    tfidf_vectors = tfidf_vectorizer.transform(data['cleaned_review'])
    predictions=model.predict(tfidf_vectors)
    print("prediction done, starting post processing")
    data['predicted_labels']=predictions
    data['predicted_emotion'] = data['predicted_labels'].map(dic_emotions['emotion'])
    percentage_emotions=(data['predicted_emotion'].value_counts(normalize=True)*100).to_dict()
    percentage_emotions = {k: int(round(v, 0)) for k, v in percentage_emotions.items()}
    print("Got the emotions for the reviews",percentage_emotions)
    return percentage_emotions
