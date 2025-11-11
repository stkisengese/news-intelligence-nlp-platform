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
from datetime import datetime
from urllib.parse import urljoin, urlparse
import uuid
import random

class NewsScraper:
    """A web scraper for news articles from BBC News."""

    def __init__(self, base_url="Https://www.bbc.com", data_dir="scraped_news"):
        self.base_url = base_url
        self.data_dir = data_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.scraped_urls = set()
        self.articles_scraped = 0

        # create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

    def save_article(self, article):
        """Save the article data to a JSON file."""
        
        # Create date-based directory
        date_dir = os.path.join(self.data_dir, article['date'])
        os.makedirs(date_dir, exist_ok=True)

        # save article as JSON file
        filepath = os.path.join(date_dir, f"{article['unique_id']}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, ensure_ascii=False, indent=2)

        print(f"    saved article: {filepath}")
        self.articles_scraped += 1

    def scrape_article(self, url):
        """Scrape a single article given its URL."""
        
        try:
            print(f"\n{self.articles_scraped + 1}. scraping {url}")
            print("    requesting ...")
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            print("    parsing ...")
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract headline
            headline = None
            headline_tag = soup.find('h1')
            if headline_tag:
                headline = headline_tag.get_text(strip=True)
            
            # Extract article body
            body_paragraphs = []
            
            # BBC News specific selectors for article content
            article_body = soup.find('article')
            if article_body:
                paragraphs = article_body.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 30:  # Filter out short text snippets
                        body_paragraphs.append(text)
            
            # Fallback: try to find all paragraphs
            if not body_paragraphs:
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 30:
                        body_paragraphs.append(text)
            
            body = ' '.join(body_paragraphs)
            
            # Validate article has minimum content
            if not headline or len(body) < 100:
                print(f"    Skipping: Insufficient content")
                return None
            
            # Create article data structure
            article = {
                'unique_id': str(uuid.uuid4()),
                'url': url,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'headline': headline,
                'body': body
            }
            
            self.scraped_urls.add(url)
            return article
            
        except requests.exceptions.RequestException as e:
            print(f"    Error requesting page: {e}")
            return None
        except Exception as e:
            print(f"    Error parsing article: {e}")
            return None

    def get_article_links(self, category_url, max_links=100):
        """fetch article links from a category page."""

        try:
            print(f"Fetching article links from category: {category_url}")
            response = requests.get(category_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []

            # Find article links
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # BBC News article pattern
                if '/news/articles/' in href or ('/news/' in href and len(href.split('/')) > 4):
                    full_url = urljoin(self.base_url, href)

                    # Avoid duplicates
                    if (full_url not in self.scraped_urls and full_url not in links and
                        'video' not in full_url.lower() and 'live' not in full_url.lower()):
                        links.append(full_url)

                        if len(links) >= max_links:
                            break

            return links
        except requests.exceptions.RequestException as e:
            print(f"Error fetching category page {category_url}: {e}")
            return []

    def scrape(self, target_count=300, categories=None):
        """Main method to start scraping articles."""
        if categories is None:
            categories = [
                "https://www.bbc.com/news",
                "https://www.bbc.com/business",
                "https://www.bbc.com/innovation/technology",
                "https://www.bbc.com/innovation/science",
                "https://www.bbc.com/innovation/artificial-intelligence",
                "https://www.bbc.com/business/technology-of-business",
                "https://www.bbc.com/business/future-of-business",
                "https://www.bbc.com/news/world/middle_east",
                "https://www.bbc.com/news/world/australia",
                "https://www.bbc.com/news/world/asia",
                "https://www.bbc.com/news/europe",
                "https://www.bbc.com/news/world/us-canada",
                "https://www.bbc.com/news/world/africa",
                "https://www.bbc.com/news/uk",
                "https://www.bbc.com/culture",
                "https://www.bbc.com/culture/style",
                "https://www.bbc.com/culture/entertainment-news",
                "https://www.bbc.com/arts",
                "https://www.bbc.com/arts/arts-in-motion",
                "https://www.bbc.com/travel/destinations",
                "https://www.bbc.com/travel/worlds-table",
                "https://www.bbc.com/future-planet",
                "https://www.bbc.com/future-planet/solutions",
                "https://www.bbc.com/future-planet/green-living",
                "http://bbc.com/sport/cycling",
                "http://bbc.com/sport/cycling",
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