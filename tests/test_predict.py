import unittest
import pandas as pd
import datasets
from src.utils import tfidf_lr_model, bert_finetune_model
from src.predict import classify_tfidf, classify_bert

class TestPredict(unittest.TestCase):
    def test_tfidfclassifier(self):
        title = "The book thief"
        tfidf_model = tfidf_lr_model()
        book, output_tfidf = classify_tfidf(title, tfidf_model)
        self.assertGreater(len(book),0,"Title length is 0")
        self.assertEqual(type(output_tfidf),dict,"not expected type for the predicted emotions result")
    def test_bertclassifier(self):
        title = "The book thief"
        trainer = bert_finetune_model()
        book, output_bert = classify_bert(title, trainer)
        self.assertGreater(len(book),0,"Title length is 0")
        self.assertEqual(type(output_bert),dict,"not expected type for the predicted emotions result")



