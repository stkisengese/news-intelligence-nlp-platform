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

def main():
    scraper = NewsScraper()
    scraper.scrape()

if __name__ == "__main__":
    main()