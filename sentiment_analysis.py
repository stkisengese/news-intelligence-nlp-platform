# sentiment_analysis.py

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

_vader_analyzer = None

def get_vader_analyzer():
    """
    Initializes and returns a VADER SentimentIntensityAnalyzer instance.
    Downloads the 'vader_lexicon' if not already present.
    """
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except nltk.downloader.DownloadError:
            nltk.download('vader_lexicon')
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer

def analyze_and_classify_article_sentiment(headline, body):
    """
    Analyzes the sentiment of an article (headline + body) and classifies it.

    Args:
        headline (str): The headline of the article.
        body (str): The body of the article.

    Returns:
        tuple: A tuple containing (compound_score, sentiment_classification_string).
               sentiment_classification_string is 'positive', 'negative', or 'neutral'.
    """
    analyzer = get_vader_analyzer()
    combined_text = headline + ". " + body
    sentiment_scores = analyzer.polarity_scores(combined_text)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        classification = 'positive'
    elif compound_score <= -0.05:
        classification = 'negative'
    else:
        classification = 'neutral'

    return compound_score, classification

if __name__ == '__main__':
    # The get_vader_analyzer() will download if necessary
    print("VADER lexicon is ready.")
    
    sample_headline = "New AI can write code like a human"
    sample_body = "A new artificial intelligence model has been developed that can write code with surprising accuracy. This could revolutionize the software development industry."
    
    compound_score, sentiment_classification = analyze_and_classify_article_sentiment(sample_headline, sample_body)
    
    print(f"Article: '{sample_headline}'")
    print(f"Compound Score: {compound_score}")
    print(f"The article has a {sentiment_classification} sentiment")

    print("\n-------------------\n")

    sample_headline_2 = "Stock market crashes amidst economic uncertainty"
    sample_body_2 = "The stock market took a nosedive today, with major indices plummeting as investors fear a global recession. Many companies are facing significant losses."

    compound_score_2, sentiment_classification_2 = analyze_and_classify_article_sentiment(sample_headline_2, sample_body_2)

    print(f"Article: '{sample_headline_2}'")
    print(f"Compound Score: {compound_score_2}")
    print(f"The article has a {sentiment_classification_2} sentiment")