"""
Run this once to download the data that nltk needs for creating bag of words representation
"""
print('downloading nltk data')
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')