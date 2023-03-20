import pandas as pd
from preprocess import text_cleaning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from utils import load_data
from evaluate import classification_metrics, confusionmatrix
import sys
sys.path.append("src/")

def train():
    df=load_data()
    df=df.reset_index()
    df['labels'] = df['emotions'].factorize()[0]
    df["cleaned_text"] = df["text"].apply(text_cleaning)
    df = df[df['cleaned_text'].map(len) > 0]
    uniquevalues = pd.unique(df[['emotions']].values.ravel())
    df_unique=pd.DataFrame(uniquevalues,columns=['emotion'])
    tfidf_vectorizer = TfidfVectorizer()
    y =df['labels']
    Xtrain, Xtest, ytrain, ytest = train_test_split(df['text'], y, test_size=0.3,random_state=1)
    Xtrain_tfidf = tfidf_vectorizer.fit_transform(Xtrain)
    Xtest_tfidf = tfidf_vectorizer.transform(Xtest)
    lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr_mn.fit(Xtrain_tfidf, ytrain)
    ypred=lr_mn.predict(Xtest_tfidf)
    classification_metrics(ypred, ytest,Xtrain_tfidf,ytrain,lr_mn)
    confusionmatrix(ypred,ytest)

if __name__ == '__main__':
  train()