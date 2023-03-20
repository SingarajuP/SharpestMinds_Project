import unittest
from src.utils import detect_en

class TestLang(unittest.TestCase):

    def test_lang_en(self):
        test="test case to check language"
        lang=detect_en(test)

        self.assertEqual(lang,True,"Not English language")

    def test_lang_noten(self):
        test="cas de test pour vérifier la langue"
        lang=detect_en(test)

        self.assertEqual(lang,True,"Not English language")
