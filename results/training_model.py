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

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def preprocess(text):
    """Preprocess text using spaCy: lemmatization, lowercasing, stopword and punctuation removal."""
    
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def plot_learning_curves(model, X_train, y_train, results_path: str):
    """Plot and save learning curves."""
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curves (Logistic Regression)")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig(f'{results_path}/learning_curves.png')
    print(f"Learning curves plot saved to '{results_path}/learning_curves.png'")

def train_and_evaluate_model(train_link, test_link, results_path='.'):
    """
    Trains and evaluates the topic classification model.
    
    This function loads the training and test data, preprocesses the text,
    trains a Logistic Regression model, evaluates its performance,
    and saves the trained model, vectorizer, and learning curves plot.
    
    Args:
        data_path (str): Path to the directory containing the data.
        results_path (str): Path to the directory to save the results.
    """
    # Load data
    try:
        train_df = pd.read_csv(train_link)
        test_df = pd.read_csv(test_link, index_col=0)
        print("Data loaded successfully.")
    except Exception as e:
        print("Error loading data:", e)
        return

    # Preprocess text
    train_df['processed_text'] = train_df['Text'].apply(preprocess)
    test_df['processed_text'] = test_df['Text'].apply(preprocess)
    print("Text preprocessing completed.")

    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    X_train = vectorizer.fit_transform(train_df['processed_text'])
    X_test = vectorizer.transform(test_df['processed_text'])

    # Prepare labels
    y_train = train_df['Category']
    y_test = test_df['Category']

    # Train a Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Plot learning curves
    plot_learning_curves(model, X_train, y_train, results_path)

    # Save the model and vectorizer
    with open(f'{results_path}/topic_classifier.pkl', 'wb') as f:
        pickle.dump((model, vectorizer), f)
    print(f"Model saved to '{results_path}/topic_classifier.pkl'")


if __name__ == "__main__":
    train_link = "https://learn.zone01kisumu.ke/api/content/root/public/subjects/ai/nlp-scraper/bbc_news_train.csv"
    test_link = "https://learn.zone01kisumu.ke/api/content/root/public/subjects/ai/nlp-scraper/bbc_news_tests.csv"
    results_path = "results"

    metrics = train_and_evaluate_model(train_link, test_link, results_path)
    print("Training completed. Metrics:", metrics)
