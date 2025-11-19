import pickle
import sys
sys.path.append('..')
from results.training_model import preprocess

def load_topic_model(results_path='results'):
    """
    Loads the trained topic classification model and vectorizer.
    
    Args:
        results_path (str): Path to the directory containing the model.
        
    Returns:
        A tuple containing the loaded model and vectorizer.
    """
    try:
        with open(f'{results_path}/topic_classifier.pkl', 'rb') as f:
            model, vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        print(f"Model file not found in '{results_path}'. Please train the model first.")
        return None, None

def predict_topic(text, model, vectorizer):
    """
    Predicts the topic of a given text.
    
    Args:
        text (str): The text to classify.
        model: The trained topic classification model.
        vectorizer: The fitted TF-IDF vectorizer.
        
    Returns:
        The predicted topic label.
    """
    processed_text = preprocess(text)
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)
    return prediction[0]

def classify_article(headline, body):
    """
    Classifies an article into a topic based on its headline and body.
    
    Args:
        headline (str): The headline of the article.
        body (str): The body of the article.
        
    Returns:
        The predicted topic label.
    """
    model, vectorizer = load_topic_model()
    if model and vectorizer:
        text = headline + " " + body
        return predict_topic(text, model, vectorizer)
    return None
