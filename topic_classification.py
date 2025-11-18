
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
