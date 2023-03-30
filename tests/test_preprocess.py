import unittest
import pandas as pd
import datasets
from src.preprocess import dataset_dict_bert, text_cleaning
class TestPreprocess(unittest.TestCase):
    def test_clean_text(self):
        test="And to test & this 1223 function; for cleanings TEXT with utmost carings"
        clean_text=text_cleaning(test)
        self.assertEqual(clean_text,"and test function cleaning text utmost caring","Cleaning incorrect")


    def test_data_dict(self):
        text=["testing for dataframe to convert into dictionary format compatible for bert model"]
        df = pd.DataFrame(text, columns=['reviews'])
        mydict=dataset_dict_bert(df)
        self.assertEqual(type(mydict), datasets.dataset_dict.DatasetDict ,"not a proper dict format for bert")


    

