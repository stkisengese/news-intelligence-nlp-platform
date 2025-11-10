"""
News Web Scraper
Scrapes news articles from BBC News and stores them with structured data.
Implements polite scraping practices and error handling.
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import os
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import uuid
import logging
import random

class NewsScraper:
    """A web scraper for news articles from BBC News."""

    def __init__(self, base_url="Https://www.bbc.com/news", data_dir="scraped_news"):
        self.base_url = base_url
        self.data_dir = data_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.scraped_urls = set()
        self.articles_scraped = 0

        # create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

    def scrape(self, target_count=300, categories=None):
        """Main method to start scraping articles."""
        if categories is None:
            categories = [
                "https://www.bbc.com/news",
                "https://www.bbc.com/news/business",
                "https://www.bbc.com/news/technology",
                "https://www.bbc.com/news/world",
                "https://www.bbc.com/news/uk",
                "https://www.bbc.com/news/science-environment",
                "https://www.bbc.com/news/health",
                "https://www.bbc.com/news/entertainment_and_arts",
                "https://www.bbc.com/sport"    
            ]

        print(f"Starting scraper - Target: {target_count} articles")
        print(f"Data directory: {self.data_dir}")
        print("=" * 60)
        
        article_links = []
        
        # Collect article links from all categories
        for category in categories:
            links = self.get_article_links(category, max_links=100)
            article_links.extend(links)
            time.sleep(random.uniform(1, 3))  # polite delay

            if len(article_links) >= target_count * 2:
                break

        # Remove duplicates
        article_links = list(set(article_links))
        print(f"\nFound {len(article_links)} unique article links")
        print("=" * 60)
        
        # Scrape articles
        for url in article_links:
            if self.articles_scraped >= target_count:
                break
            
            article = self.scrape_article(url)
            if article:
                self.save_article(article)
            
            # Polite delay between requests (1-2 seconds)
            time.sleep(random.uniform(1.0, 2.0))
        
        print("\n" + "=" * 60)
        print(f"Scraping complete!")
        print(f"Total articles scraped: {self.articles_scraped}")
        print(f"Articles saved in: {self.data_dir}/")
        print("=" * 60)


def main():
    scraper = NewsScraper()

    # Start scraping
    try:
        scraper.scrape(target_count=300)
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user")
        print(f"Articles scraped so far: {scraper.articles_scraped}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print(f"Articles scraped so far: {scraper.articles_scraped}")


if __name__ == "__main__":
    main()