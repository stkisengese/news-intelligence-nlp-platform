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

def stem_tokens(tokens, stemmer='porter'):
    """
    Stem a list of tokens.

    Args:
        tokens (list): A list of word tokens.
        stemmer (str, optional): The stemmer to use. Defaults to 'porter'.

    Returns:
        list: A list of stemmed tokens.
    """
    if stemmer == 'porter':
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in tokens]
    return tokens

def preprocess_pipeline(text, steps=['lowercase', 'tokenize', 'remove_stopwords', 'stem']):
    """
    Apply a series of preprocessing steps to the text.

    Args:
        text (str): The input text.
        steps (list, optional): A list of preprocessing steps to apply.
            Defaults to ['lowercase', 'tokenize', 'remove_stopwords', 'stem'].

    Returns:
        list: A list of preprocessed tokens.
    """
    if 'lowercase' in steps:
        text = text.lower()
    
    if 'tokenize' in steps:
        text = clean_text(text)
        tokens = tokenize_words(text)
    else:
        tokens = text.split()

    if 'remove_stopwords' in steps:
        tokens = remove_stopwords(tokens)

    if 'stem' in steps:
        tokens = stem_tokens(tokens)
        
    return tokens

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
    
    # 4. Remove stopwords
    filtered_words = remove_stopwords(words)
    print("\nWords after removing stopwords:")
    print(filtered_words)
    
    # 5. Stem tokens
    stemmed_words = stem_tokens(filtered_words)
    print("\nStemmed Words:")
    print(stemmed_words)
    
    # 6. Full pipeline
    processed_tokens = preprocess_pipeline(sample_text)
    print("\nFull Pipeline Output:")
    print(processed_tokens)
    
    # Example with different pipeline steps
    processed_tokens_custom = preprocess_pipeline(sample_text, steps=['lowercase', 'tokenize'])
    print("\nCustom Pipeline (lowercase, tokenize):")
    print(processed_tokens_custom)
