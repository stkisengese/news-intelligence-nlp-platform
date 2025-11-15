import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

def clean_text(text, remove_punctuation=True, lowercase=True):
    """
    Clean the input text by removing punctuation and converting to lowercase.

    Args:
        text (str): The input text to clean.
        remove_punctuation (bool, optional): Whether to remove punctuation. Defaults to True.
        lowercase (bool, optional): Whether to convert text to lowercase. Defaults to True.

    Returns:
        str: The cleaned text.
    """
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    return text


if __name__ == '__main__':
    # Example usage of the preprocessing pipeline
    sample_text = "This is a sample sentence, showing off the stop words filtration and stemming."
    
    print("Original Text:")
    print(sample_text)
    
    # 1. Clean text
    cleaned_text = clean_text(sample_text)
    print("\nCleaned Text:")
    print(cleaned_text)
    
    