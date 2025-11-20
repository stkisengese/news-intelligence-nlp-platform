import json
import pandas as pd
from utils.entity_detection import load_spacy_model, extract_organizations
from utils.topic_classification import classify_article, load_topic_model
from utils.sentiment_analysis import analyze_and_classify_article_sentiment
from utils.scandal_detection import load_embedding_model, detect_scandal

# Define ANSI color codes
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_CYAN = '\033[96m'
COLOR_MAGENTA = '\033[95m'
COLOR_RESET = '\033[0m'

# def load_articles_from_data(data_dir='data'):
#     """
#     Load articles from JSONL files in the data directory.
#     """
#     articles = []
#     for filename in os.listdir(data_dir):
#         if filename.endswith('.jsonl'):
#             with open(os.path.join(data_dir, filename), 'r') as f:
#                 for line in f:
#                     articles.append(json.loads(line))
#     return articles

def load_articles_from_file(filepath):
    """
    Load articles from a single JSONL file.
    """
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles

def main():
    """
    Main function to run the NLP enrichment pipeline.
    """
    nlp_model = load_spacy_model()
    embedding_model = load_embedding_model()
    # articles = load_articles_from_data()
    articles = load_articles_from_file('data/2025-11-13.jsonl')
    model, vectorizer = load_topic_model()
    
    enriched_data = []
    
    for article in articles:
        print(f"{COLOR_GREEN}\nEnriching {article['url']}:{COLOR_RESET}")
        
        # ---------- Detect entities ----------
        print(f"{COLOR_CYAN}---------- Detect entities ----------{COLOR_RESET}")
        organizations = extract_organizations(article['headline'], article['body'], nlp_model)
        print(f"{COLOR_YELLOW}Detected {len(organizations)} companies which are {', '.join(organizations)}{COLOR_RESET}")
        
        # ---------- Topic detection ----------
        print(f"{COLOR_CYAN}---------- Topic detection ----------{COLOR_RESET}")
        topic = classify_article(article['headline'], article['body'], model, vectorizer)
        print(f"{COLOR_YELLOW}The topic of the article is: {COLOR_RESET}{topic}")
        
        # ---------- Sentiment analysis ----------
        print(f"{COLOR_CYAN}---------- Sentiment analysis ----------{COLOR_RESET}")
        compound_score, sentiment_classification = analyze_and_classify_article_sentiment(article['headline'], article['body'])
        print(f"{COLOR_YELLOW}The article '{article['headline']}' has a {COLOR_RESET}{sentiment_classification} sentiment")
        
        # ---------- Scandal detection ----------
        print(f"{COLOR_CYAN}---------- Scandal detection ----------{COLOR_RESET}")
        scandal_score = detect_scandal(article['body'], organizations, embedding_model)
        print(f"{COLOR_MAGENTA}Computing embeddings and distance ...{COLOR_RESET}")
        
        enriched_article = {
            'unique_id': article['unique_id'],
            'url': article['url'],
            'date': article['date'],
            'headline': article['headline'],
            'body': article['body'],
            'org': organizations,
            'topics': [topic],
            'sentiment': compound_score,
            'scandal_distance': scandal_score,
            'top_10': False
        }
        enriched_data.append(enriched_article)

    # Flag top 10 articles with highest scandal scores
    enriched_data.sort(key=lambda x: x['scandal_distance'], reverse=True)
    for i in range(min(10, len(enriched_data))):
        enriched_data[i]['top_10'] = True
        
    df = pd.DataFrame(enriched_data)
    df.to_csv('results/enhanced_news.csv', index=False)
    print(f"{COLOR_GREEN}Saved enriched news to results/enhanced_news.csv{COLOR_RESET}")

if __name__ == '__main__':
    main()
