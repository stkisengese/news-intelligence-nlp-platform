import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import spacy
sys.path.append('..')

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

def preprocess(text):
    """Preprocess text using spaCy: lemmatization, lowercasing, stopword and punctuation removal."""
    
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)
