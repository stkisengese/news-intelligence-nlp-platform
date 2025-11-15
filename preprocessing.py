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

def tokenize_sentences(text):
    """
    Tokenize the text into sentences.

    Args:
        text (str): The input text.

    Returns:
        list: A list of sentences.
    """
    return sent_tokenize(text)

def tokenize_words(text):
    """
    Tokenize the text into words.

    Args:
        text (str): The input text.

    Returns:
        list: A list of words.
    """
    return word_tokenize(text)

def remove_stopwords(tokens, language='english'):
    """
    Remove stopwords from a list of tokens.

    Args:
        tokens (list): A list of word tokens.
        language (str, optional): The language of the stopwords. Defaults to 'english'.

    Returns:
        list: A list of tokens with stopwords removed.
    """
    stop_words = set(stopwords.words(language))
    return [word for word in tokens if word not in stop_words]


if __name__ == '__main__':
    # Example usage of the preprocessing pipeline
    sample_text = "This is a sample sentence, showing off the stop words filtration and stemming."
    
    print("Original Text:")
    print(sample_text)
    
    # 1. Clean text
    cleaned_text = clean_text(sample_text)
    print("\nCleaned Text:")
    print(cleaned_text)
    
    # 2. Tokenize sentences
    sentences = tokenize_sentences(sample_text)
    print("\nTokenized Sentences:")
    print(sentences)
    
    # 3. Tokenize words
    words = tokenize_words(cleaned_text)
    print("\nTokenized Words:")
    print(words)
    
  