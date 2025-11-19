import json
import os
import pandas as pd
from entity_detection import load_spacy_model, extract_organizations
from topic_classification import classify_article
from sentiment_analysis import analyze_and_classify_article_sentiment
from scandal_detection import load_embedding_model, detect_scandal

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
    embedding_model = load_embedding_model()
    articles = load_articles_from_data()
    
    enriched_data = []
    
    for article in articles:
        print(f"Enriching {article['url']}:")
        
        # ---------- Detect entities ----------
        organizations = extract_organizations(article['headline'], article['body'], nlp_model)
        print(f"Detected {len(organizations)} companies which are {', '.join(organizations)}")
        
        # ---------- Topic detection ----------
        print("---------- Topic detection ----------")
        topic = classify_article(article['headline'], article['body'])
        print(f"The topic of the article is: {topic}")
        
        # ---------- Sentiment analysis ----------
        print("---------- Sentiment analysis ----------")
        compound_score, sentiment_classification = analyze_and_classify_article_sentiment(article['headline'], article['body'])
        print(f"The article {article['headline']} has a {sentiment_classification} sentiment")
        
        # ---------- Scandal detection ----------
        print("---------- Scandal detection ----------")
        scandal_score = detect_scandal(article['body'], organizations, embedding_model)
        print("Computing embeddings and distance ...")
        
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
    print("Saved enriched news to results/enhanced_news.csv")

if __name__ == '__main__':
    main()
