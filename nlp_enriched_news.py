import json
import os
import pandas as pd
from entity_detection import load_spacy_model, extract_organizations

def load_articles_from_data(data_dir='data'):
    """
    Load articles from JSONL files in the data directory.
    """
    articles = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jsonl'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                for line in f:
                    articles.append(json.loads(line))
    return articles

def main():
    """
    Main function to run the NLP enrichment pipeline.
    """
    nlp_model = load_spacy_model()
    articles = load_articles_from_data()
    
    enriched_data = []
    
    for i, article in enumerate(articles):
        if i >= 10:
            break
        print(f"Enriching {article['url']}:")
        
        # ---------- Detect entities ----------
        organizations = extract_organizations(article['headline'], article['body'], nlp_model)
        print(f"Detected {len(organizations)} companies which are {', '.join(organizations)}")
        
        # ---------- Topic detection ----------
        print("---------- Topic detection ----------")
        # TODO: Implement topic detection
        topic = "Not implemented"
        print(f"The topic of the article is: {topic}")
        
        # ---------- Sentiment analysis ----------
        print("---------- Sentiment analysis ----------")
        # TODO: Implement sentiment analysis
        sentiment = "Not implemented"
        print(f"The article {article['headline']} has a {sentiment} sentiment")
        
        # ---------- Scandal detection ----------
        print("---------- Scandal detection ----------")
        # TODO: Implement scandal detection
        scandal_score = 0.0
        print("Computing embeddings and distance ...")
        
        enriched_article = {
            'unique_id': article['unique_id'],
            'url': article['url'],
            'date': article['date'],
            'headline': article['headline'],
            'body': article['body'],
            'org': organizations,
            'topics': [topic],
            'sentiment': sentiment,
            'scandal_distance': scandal_score,
            'top_10': False # To be implemented
        }
        enriched_data.append(enriched_article)
        
    df = pd.DataFrame(enriched_data)
    df.to_csv('results/enhanced_news.csv', index=False)
    print("Saved enriched news to results/enhanced_news.csv")

if __name__ == '__main__':
    main()
