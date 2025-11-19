# NLP-enriched News Intelligence Platform
A Python-based NLP platform for scraping, analyzing, and enriching news articles with entity detection, topic classification, sentiment analysis, and scandal detection. Designed to help analysts extract actionable insights from large volumes of news data.

## 📰 Project Overview
The goal of this project is to build an advanced platform for News Intelligence. It connects to a news data source and uses various Natural Language Processing (NLP) techniques to enrich the articles.

Key functionalities include:
1. **Entity Detection:** Identifying organizations (ORG) using spaCy.
2. **Topic Classification:** Classifying articles into categories (e.g., Tech, Business) using a custom-trained model.
3. **Sentiment Analysis:** Determining article sentiment (Positive/Negative/Neutral) using a pre-trained NLTK model.
4. **Scandal Detection:** Flagging potential environmental disaster risks by calculating the semantic distance between article sentences and defined keywords using embeddings.

## 🚀 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/stkisengese/news-intelligence-nlp-platform.git news-nlp
   cd news-nlp
   ```

2. **Setup Environment**
    ```bash
    conda create -n news-nlp-env python=3.10
    conda activate news-nlp-env

    # alternatively
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate   # On Windows
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4. **Download NLP models and datasets**
    ```bash
    # Download the small English model for spaCy NER
    python -m spacy download en_core_web_sm

    # Download NLTK resources (VADER for sentiment, punkt for tokenization)
    python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
    ```

## Topic Classification Model

The topic classification model is a multi-class classifier that categorizes news articles into one of five topics: Tech, Sport, Business, Entertainment, or Politics.

### Model Architecture

The model is a Logistic Regression classifier trained on a TF-IDF representation of the preprocessed text data. The TF-IDF vectorizer is configured to use a maximum of 5000 features.

### Dataset

The model was trained and evaluated on the BBC News dataset, which is split into training and test sets. The dataset is located in the `data` directory.

### Performance

The model achieves an accuracy of over 95% on the test set. The following table shows the precision, recall, and F1-score for each class:

| Category      | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Tech          | 0.98      | 0.97   | 0.98     |
| Sport         | 0.99      | 0.99   | 0.99     |
| Business      | 0.96      | 0.97   | 0.96     |
| Entertainment | 0.98      | 0.98   | 0.98     |
| Politics      | 0.94      | 0.94   | 0.94     |
| **Weighted Avg**  | **0.97**      | **0.97**   | **0.97**     |

### Learning Curves

The learning curves for the model are shown below. The plot shows that the model does not overfit and that the training and cross-validation scores converge.

![Learning Curves](results/learning_curves.png)

## Sentiment Analysis

Sentiment analysis is performed using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner), a pre-trained model optimized for news and social media text.

### Methodology

VADER is effective without extensive preprocessing because it considers punctuation (e.g., "!") and capitalization (e.g., "GREAT") in its sentiment calculations. The `sentiment_analysis.py` module combines the article's headline and body to get a holistic view of its sentiment.

The sentiment is classified based on the `compound` score returned by VADER:
- **Positive:** `compound score >= 0.05`
- **Neutral:** `compound score > -0.05` and `compound score < 0.05`
- **Negative:** `compound score <= -0.05`

## Scandal Detection

Scandal detection is implemented to identify articles that report on environmental disasters linked to specific companies. This system uses semantic similarity to connect organizations with ESG (Environmental, Social, and Governance) risk-related keywords.

### Methodology

The process involves the following steps:
1.  **Keyword Definition**: A predefined list of keywords and phrases related to environmental disasters (e.g., "oil spill," "deforestation," "emissions scandal") is maintained.
2.  **Entity-Sentence Extraction**: For each article, the system extracts sentences that contain the names of organizations identified by the NER module.
3.  **Embedding Generation**: The predefined disaster keywords and the extracted entity-sentences are converted into high-dimensional vectors using a sentence-transformer model.
4.  **Similarity Calculation**: The cosine similarity is computed between the keyword embeddings and the sentence embeddings. A high similarity score indicates a potential link between a company and a disaster event.
5.  **Scandal Scoring**: The similarity scores are aggregated to produce a final "scandal score" for the article. This score reflects the likelihood that the article is reporting on an environmental scandal.

### Embedding Model

The system uses the `all-MiniLM-L6-v2` sentence-transformer model. This model is chosen for its efficiency and strong performance on semantic similarity tasks. It maps sentences and paragraphs to a 384-dimensional dense vector space and is ideal for clustering or semantic search.

### Similarity Metric

**Cosine Similarity** is used to measure the semantic distance between the disaster keywords and the article sentences. This metric is well-suited for high-dimensional spaces and effectively captures the contextual closeness of the texts, regardless of their length.

### Keyword Selection

The disaster keywords were carefully selected to be specific and unambiguous to minimize false positives. The list includes terms that are strongly associated with environmental incidents and corporate malfeasance. The keywords are regularly reviewed and updated to ensure relevance.