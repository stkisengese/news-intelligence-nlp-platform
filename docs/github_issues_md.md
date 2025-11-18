# NLP-Enriched News Intelligence Platform - GitHub Issues

This document contains all GitHub issues for the NLP-enriched News Intelligence Platform project created in project repository.

---

## Issue #1: Project Setup and Environment Configuration

**Title:** Initialize project structure and Python environment

**Labels:** `setup`, `environment`, `documentation`

**Depends On:** None

### Description
Set up the foundational project structure and Python environment for the NLP-enriched News Intelligence Platform. This includes creating the directory structure, initializing a git repository, setting up a virtual environment, and defining all required dependencies.

The project requires NLP libraries (spaCy, NLTK), web scraping tools (BeautifulSoup, requests), machine learning frameworks (scikit-learn), and data processing libraries (pandas, numpy). Proper environment setup ensures reproducibility and smooth development workflow.

### Deliverables
- Complete project directory structure:
  ```
  project/
  ├── data/
  ├── results/
  ├── scraper_news.py (empty placeholder)
  ├── nlp_enriched_news.py (empty placeholder)
  ├── requirements.txt
  ├── README.md
  └── .gitignore
  ```
- `requirements.txt` with all necessary dependencies:
  - requests>=2.28.0
  - beautifulsoup4>=4.11.0
  - spacy>=3.5.0
  - nltk>=3.8.0
  - pandas>=1.5.0
  - numpy>=1.23.0
  - scikit-learn>=1.2.0
  - matplotlib>=3.6.0
  - sentence-transformers>=2.2.0 (for embeddings)
- Initial `README.md` with project overview
- `.gitignore` configured for Python projects

### Acceptance Criteria
- [x] All directories (`data/`, `results/`) are created
- [x] `requirements.txt` includes all specified dependencies with version constraints
- [x] Virtual environment can be created and all packages install without errors
- [x] spaCy English model (`en_core_web_sm`) is downloaded successfully
- [x] NLTK required datasets (vader_lexicon, punkt) are downloaded
- [x] README.md contains project title, brief description, and setup instructions
- [x] `.gitignore` excludes `data/`, `results/`, `__pycache__/`, `*.pyc`, virtual environment folders

---

## Issue #2: Implement News Web Scraper

**Title:** Build web scraper to collect news articles from target source

**Labels:** `feature`, `data`, `scraper`

**Depends On:** Issue #1

### Description
Develop a web scraper that extracts news articles from a chosen news website. The scraper must identify an easily scrapable news source and extract structured article data including unique identifiers, URLs, publication dates, headlines, and article bodies.

The scraper should be designed to collect articles from the past week and must handle pagination, rate limiting, and basic error handling. Implement polite scraping practices (user-agent headers, delays between requests) to respect the website's terms of service.

Key implementation considerations:
- Select a news website with stable HTML structure (e.g., BBC News, The Guardian, Reuters)
- Parse HTML using BeautifulSoup
- Extract publication dates and convert to proper date format
- Generate unique IDs for each article (UUID or incremental)
- Implement error handling for failed requests
- Add logging to track scraping progress

### Deliverables
- `scraper_news.py` with complete scraping logic
- Scraper should output progress to console:
  ```
  1. scraping <URL>
      requesting ...
      parsing ...
      saved in <path>
  ```
- At least 300 articles scraped and stored

### Acceptance Criteria
- [x] Scraper successfully identifies and extracts articles from chosen news website
- [x] Each article contains: unique_id, url, date, headline, body
- [x] Script handles HTTP errors gracefully (404, 503, timeouts)
- [x] Implements polite scraping (1-2 second delay between requests)
- [x] Console output shows scraping progress as specified
- [x] Successfully scrapes minimum 300 articles from past week
- [x] Article dates are parsed correctly and stored in consistent format
- [x] No duplicate articles are stored (based on URL uniqueness)

---

## Issue #3: Implement Data Persistence Layer

**Title:** Create data storage solution for scraped articles

**Labels:** `feature`, `data`, `database`

**Depends On:** Issue #2

### Description
Implement a data persistence mechanism to store scraped news articles. Choose between two approaches: (1) file-based storage with one JSON/CSV file per day, or (2) SQLite database with proper schema design.

For file-based approach: Create organized directory structure under `data/` with date-based folders. Store articles in structured JSON or CSV format.

For database approach: Design a normalized schema with articles table containing all required fields. Implement database connection, table creation, and CRUD operations.

The storage solution must support efficient retrieval for the NLP engine and prevent duplicate entries. Include helper functions for saving and loading article data.

### Deliverables
- Data storage implementation in `scraper_news.py` or separate `database.py` module
- Storage schema documentation in README.md
- Helper functions:
  - `save_article(article_data)` - Store single article
  - `load_articles(date_range=None)` - Retrieve articles for processing
  - `get_article_count()` - Return total articles stored
- Updated `scraper_news.py` to use storage functions

### Acceptance Criteria
- [x] Storage mechanism successfully saves all article fields (unique_id, url, date, headline, body)
- [x] No duplicate articles can be stored (enforced by unique_id or URL)
- [x] Data can be retrieved efficiently for batch processing
- [x] File-based: Articles organized in `data/YYYY-MM-DD/` folders OR Database: SQLite file in `data/` directory
- [x] README.md documents the storage structure and schema
- [x] Storage handles special characters and encoding properly (UTF-8)
- [x] `load_articles()` function returns data in consistent format (list of dictionaries)

---

## Issue #4: Build Text Preprocessing Pipeline

**Title:** Implement core NLP preprocessing utilities

**Labels:** `feature`, `nlp`, `preprocessing`

**Depends On:** Issue #1

### Description
Create a comprehensive text preprocessing module that will be used across all NLP tasks. This pipeline forms the foundation for entity detection, topic classification, and sentiment analysis.

Implement the following preprocessing steps:
1. **Lowercase conversion** - Normalize text casing
2. **Punctuation removal** - Strip special characters while preserving sentence boundaries
3. **Tokenization** - Split text into sentences and words using NLTK
4. **Stop word removal** - Remove common words using NLTK stopwords corpus
5. **Stemming** - Apply Porter or Snowball stemmer to reduce words to root forms

Create modular functions that can be selectively applied based on the NLP task requirements. Some tasks (like NER) work better with original text, while others (like topic classification) benefit from full preprocessing.

### Deliverables
- New file: `preprocessing.py` with functions:
  - `clean_text(text, remove_punctuation=True, lowercase=True)`
  - `tokenize_sentences(text)`
  - `tokenize_words(text)`
  - `remove_stopwords(tokens, language='english')`
  - `stem_tokens(tokens, stemmer='porter')`
  - `preprocess_pipeline(text, steps=['lowercase', 'tokenize', 'remove_stopwords', 'stem'])`
- Unit tests demonstrating each function (inline or separate test file)

### Acceptance Criteria
- [x] `clean_text()` successfully removes punctuation and converts to lowercase
- [x] `tokenize_sentences()` correctly splits text into sentence list
- [x] `tokenize_words()` returns list of word tokens
- [x] `remove_stopwords()` filters out NLTK English stopwords
- [x] `stem_tokens()` applies stemming algorithm correctly
- [x] `preprocess_pipeline()` chains operations with configurable steps
- [x] Functions handle empty strings and edge cases gracefully
- [x] Code includes docstrings with parameter descriptions and examples
- [x] Preprocessing preserves text integrity (no information loss unless intended)

---

## Issue #5: Implement Named Entity Recognition (NER)

**Title:** Detect organizations and companies using spaCy NER

**Labels:** `feature`, `nlp`, `ner`

**Depends On:** Issue #4

### Description
Implement entity detection specifically for organizations (ORG entity type) using spaCy's pre-trained NER model. This task extracts company names and organizations mentioned in article headlines and bodies.

Use spaCy's `en_core_web_sm` or `en_core_web_lg` model to process documents and extract entities. The system should:
- Process both headline and body text
- Filter entities to include only ORG type
- Handle entity overlaps and duplicates
- Return a deduplicated list of organizations per article

Since NER works best on original text (not preprocessed), apply minimal cleaning before processing. Store entity information for later use in scandal detection.

### Deliverables
- New file: `entity_detection.py` with functions:
  - `load_spacy_model(model_name='en_core_web_sm')`
  - `detect_entities(text, entity_type='ORG')`
  - `extract_organizations(headline, body)`
- Integration code to process articles and extract organizations
- Console output: "Detected X companies which are company_1, company_2, ..."

### Acceptance Criteria
- [x] spaCy model loads successfully on script execution
- [x] `detect_entities()` correctly identifies ORG entities in text
- [x] Function filters out non-ORG entities (PERSON, GPE, DATE, etc.)
- [x] Duplicate organizations are removed (case-insensitive deduplication)
- [x] Both headline and body are processed for entity extraction
- [x] Returns empty list when no organizations detected
- [x] Console output matches specified format
- [x] Function handles long articles efficiently (no performance issues)
- [x] Code includes example usage and docstrings

---

## Issue #6: Train Topic Classification Model

**Title:** Build and train multi-class topic classifier

**Labels:** `feature`, `ml`, `nlp`, `classification`

**Depends On:** Issue #4

### Description
Develop a supervised machine learning model to classify news articles into five categories: Tech, Sport, Business, Entertainment, or Politics. Use the provided labeled training and test datasets to build a robust classifier.

Implementation steps:
1. Load and explore the provided labeled dataset
2. Implement text preprocessing pipeline (from Issue #4) optimized for classification
3. Create bag-of-words representation using `CountVectorizer` or TF-IDF
4. Train a classifier (Logistic Regression, Naive Bayes, or SVM)
5. Evaluate on test set and achieve >95% accuracy
6. Plot learning curves to verify no overfitting
7. Save trained model as `topic_classifier.pkl`
8. Create `training_model.py` in `results/` folder for auditing

**Note:** If labeled dataset is not provided, create a synthetic dataset or use a public news classification dataset (e.g., AG News, BBC News dataset).

### Deliverables
- `results/training_model.py` - Complete training script
- `results/topic_classifier.pkl` - Serialized trained model
- `results/learning_curves.png` - Plot showing training/validation accuracy
- `topic_classification.py` - Module for loading model and making predictions:
  - `load_topic_model()`
  - `predict_topic(text)`
  - `classify_article(headline, body)`

- [x] Training script loads and preprocesses labeled dataset
- [x] Text vectorization implemented using TF-IDF
- [x] Model achieves >95% accuracy on test set
- [x] Learning curves plot saved showing train/validation performance
- [x] Learning curves demonstrate no overfitting (curves converge)
- [x] Model serialized successfully with pickle
- [x] `predict_topic()` function loads model and returns topic label
- [x] Console output: "The topic of the article is: <topic>"
- [x] README.md updated with model architecture and performance metrics
- [x] Training script is reproducible (random seed set)

---

## Issue #7: Implement Sentiment Analysis

**Title:** Integrate pre-trained sentiment analysis model

**Labels:** `feature`, `nlp`, `sentiment-analysis`

**Depends On:** Issue #4

### Description
Implement sentiment analysis using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) pre-trained model. VADER is specifically designed for social media and news text, making it ideal for this application.

The sentiment analyzer should:
- Classify articles as positive, negative, or neutral
- Return sentiment scores (compound, positive, negative, neutral)
- Process both headline and body (decide on strategy: separate analysis or combined)
- Handle edge cases (very short articles, special characters)

VADER requires minimal preprocessing since it considers punctuation and capitalization for sentiment intensity. Use the compound score to determine overall sentiment: positive (≥0.05), neutral (-0.05 to 0.05), negative (≤-0.05).

### Deliverables
- `sentiment_analysis.py` with functions:
  - `initialize_vader()`
  - `analyze_sentiment(text)`
  - `get_article_sentiment(headline, body)`
  - `classify_sentiment(compound_score)` - Returns 'positive', 'negative', or 'neutral'
- Integration code to process articles
- Console output: "The article <title> has a <sentiment> sentiment"

### Acceptance Criteria
- [ ] NLTK VADER lexicon is downloaded and loaded successfully
- [ ] `analyze_sentiment()` returns dictionary with compound, pos, neg, neu scores
- [ ] `get_article_sentiment()` processes both headline and body appropriately
- [ ] Sentiment classification thresholds correctly implemented (≥0.05, ≤-0.05)
- [ ] Function handles empty or very short text gracefully
- [ ] Console output matches specified format
- [ ] README.md documents sentiment analysis approach and VADER choice rationale
- [ ] Code includes example usage with sample texts
- [ ] Returns consistent data structure (dict or float) for downstream processing

---

## Issue #8: Implement Scandal Detection System

**Title:** Build environmental disaster detection using embeddings and similarity

**Labels:** `feature`, `nlp`, `embeddings`, `similarity`

**Depends On:** Issue #5

### Description
Create a scandal detection system that identifies articles mentioning environmental disasters related to detected companies. Use semantic similarity between predefined disaster keywords and sentences containing organization entities.

Implementation approach:
1. **Define keyword set** - Create list of environmental disaster terms (pollution, oil spill, deforestation, toxic waste, emissions scandal, etc.)
2. **Generate embeddings** - Use sentence-transformers (SBERT) or Word2Vec to create vector representations
3. **Sentence extraction** - Extract sentences from articles that mention detected organizations
4. **Similarity computation** - Calculate cosine similarity between keyword embeddings and sentence embeddings
5. **Scoring** - Aggregate similarity scores per article
6. **Flagging** - Identify top 10 articles with highest scandal scores

**Important:** Avoid ambiguous keywords that have multiple meanings. Document embedding model choice and distance metric rationale in README.md.

### Deliverables
- `scandal_detection.py` with functions:
  - `load_embedding_model()` - Load sentence-transformers model
  - `define_disaster_keywords()` - Return list of environmental disaster terms
  - `extract_entity_sentences(text, entities)` - Get sentences mentioning companies
  - `compute_embeddings(texts)`
  - `calculate_similarity_scores(keyword_embeddings, sentence_embeddings)`
  - `detect_scandal(article_text, entities)` - Return scandal score
- Console output: "Environmental scandal detected for <entity>"
- README.md section explaining:
  - Embedding model choice (e.g., all-MiniLM-L6-v2)
  - Distance/similarity metric (e.g., cosine similarity)
  - Keyword selection rationale

### Acceptance Criteria
- [ ] Keyword list contains at least 15 relevant environmental disaster terms
- [ ] Keywords are specific and non-ambiguous
- [ ] Embedding model loads successfully (sentence-transformers recommended)
- [ ] Sentences containing organizations are extracted correctly
- [ ] Similarity scores calculated between keywords and sentences
- [ ] Article-level scandal score aggregated (e.g., max, mean, or weighted average)
- [ ] Top 10 articles with highest scores identified
- [ ] Console output shows entities with detected scandals
- [ ] README.md includes thorough explanation of approach
- [ ] Code includes comments explaining similarity calculation

---

## Issue #9: Build Main NLP Engine Integration Script

**Title:** Create nlp_enriched_news.py and generate enhanced_news.csv

**Labels:** `feature`, `integration`, `nlp`

**Depends On:** Issues #3, #5, #6, #7, #8

### Description
Develop the main integration script that orchestrates all NLP tasks and produces the final enriched dataset. This script loads articles from storage, applies all NLP analyses (NER, topic classification, sentiment analysis, scandal detection), and outputs results to CSV.

The script should:
1. Load all scraped articles from data storage
2. Process each article through all NLP pipelines sequentially
3. Display progress and results to console as specified
4. Compile results into structured DataFrame
5. Export to `results/enhanced_news.csv`

Output DataFrame schema:
- `unique_id` (str/int) - Article identifier
- `url` (str) - Article URL
- `date` (date) - Scraping date
- `headline` (str) - Article headline
- `body` (str) - Article body
- `org` (list[str]) - Detected organizations
- `topics` (list[str]) - Predicted topic (as list for consistency)
- `sentiment` (float) - Compound sentiment score
- `scandal_distance` (float) - Scandal detection score
- `top_10` (bool) - Flag for top 10 scandal articles

### Deliverables
- `nlp_enriched_news.py` - Main integration script with:
  - Article loading from storage
  - Sequential NLP pipeline execution
  - Progress logging as specified
  - DataFrame compilation
  - CSV export to `results/enhanced_news.csv`
- `results/enhanced_news.csv` - Final enriched dataset with all required columns
- Console output matching specification:
  ```
  Enriching <URL>:
  ---------- Detect entities ----------
  Detected X companies which are company_1, company_2, ...
  ---------- Topic detection ----------
  The topic of the article is: <topic>
  ---------- Sentiment analysis ----------
  The article <title> has a <sentiment> sentiment
  ---------- Scandal detection ----------
  Environmental scandal detected for <entity>
  ```

### Acceptance Criteria
- [ ] Script successfully loads all scraped articles (300+)
- [ ] All NLP modules imported and initialized correctly
- [ ] Each article processed through complete pipeline (NER → Topic → Sentiment → Scandal)
- [ ] Console output matches specified format for each processing step
- [ ] Progress indicator shows article count or percentage
- [ ] DataFrame created with exact schema specified
- [ ] CSV exported to `results/enhanced_news.csv` with proper encoding (UTF-8)
- [ ] Top 10 scandal articles correctly flagged (boolean column)
- [ ] Script handles errors gracefully (logs issues, continues processing)
- [ ] Processing time is reasonable (<5 minutes for 300 articles)
- [ ] CSV is readable and properly formatted (no encoding issues)

---

## Issue #10: Complete Documentation and Project Review

**Title:** Finalize README.md and prepare project for submission

**Labels:** `documentation`, `review`

**Depends On:** Issue #9

### Description
Complete comprehensive project documentation and perform final review of all deliverables. The README should serve as both user guide and technical documentation, explaining the project architecture, setup instructions, usage, and methodology.

Update README.md with:
1. **Project Overview** - Brief description and objectives
2. **Architecture** - System design and component interaction
3. **Setup Instructions** - Environment setup, dependencies, data downloads
4. **Usage Guide** - How to run scraper and NLP engine with examples
5. **NLP Pipeline Details**:
   - Entity detection approach
   - Topic classification model (architecture, training process, performance)
   - Sentiment analysis methodology
   - Scandal detection (embedding model, similarity metric, keyword justification)
6. **Results** - Sample outputs, performance metrics, insights
7. **Project Structure** - Directory tree with descriptions
8. **Dependencies** - Explanation of key libraries
9. **Future Improvements** - Potential enhancements

Verify all project deliverables are present and functional.

### Deliverables
- Comprehensive `README.md` with all sections listed above
- Verification checklist confirming:
  - All required files present
  - All scripts executable without errors
  - Results folder contains required outputs
  - Code follows consistent style
  - No hardcoded paths or credentials
- Final project structure matches specification:
  ```
  project/
  ├── data/
  │   └── [scraped articles]
  ├── nlp_enriched_news.py
  ├── requirements.txt
  ├── README.md
  ├── results/
  │   ├── training_model.py
  │   ├── enhanced_news.csv
  │   └── learning_curves.png
  └── scraper_news.py
  ```

### Acceptance Criteria
- [ ] README.md is comprehensive (minimum 1500 words)
- [ ] Setup instructions are clear and reproducible
- [ ] Architecture section includes workflow diagram or detailed explanation
- [ ] NLP methodology section explains all technical choices with rationale
- [ ] Scandal detection section justifies embedding model and similarity metric
- [ ] Sample outputs included (screenshots or text snippets)
- [ ] All scripts run successfully following README instructions
- [ ] Code includes docstrings and comments for complex logic
- [ ] No sensitive information (API keys, personal data) in repository
- [ ] requirements.txt is complete and tested
- [ ] Learning curves plot demonstrates proper model training
- [ ] enhanced_news.csv contains all required columns with valid data
- [ ] Project structure matches specification exactly

---

## Issue #11: [Optional] Implement Source Analysis and Insights

**Title:** Generate analytical insights and visualizations from collected data

**Labels:** `optional`, `analytics`, `visualization`, `insights`

**Depends On:** Issue #9

### Description
Create an optional analytics module that generates insights about the news source over time. This requires data collected over at least 5 days (ideally a full week) and produces various visualizations exploring temporal patterns, topic distributions, company mentions, and sentiment trends.

**Note:** This issue is optional but adds significant value to the project by demonstrating data analysis capabilities.

Implement visualizations for:
1. **Daily Analysis**:
   - Proportion of topics per day (stacked bar chart)
   - Number of articles published per day (line chart)
   - Number of companies mentioned per day (line chart)
   - Average sentiment per day (line chart with confidence intervals)

2. **Company Analysis**:
   - Top 20 most mentioned companies (horizontal bar chart)
   - Sentiment distribution per top 10 companies (grouped bar chart or heatmap)
   - Company-topic association matrix (heatmap)

3. **Topic Analysis**:
   - Overall topic distribution (pie chart)
   - Topic sentiment correlation (box plots)

All plots should be saved in `results/` folder with descriptive filenames.

### Deliverables
- `source_analysis.py` - Analytics script with:
  - Data loading and aggregation functions
  - Visualization generation functions
  - Main execution flow
- Visualization outputs in `results/`:
  - `topics_per_day.png`
  - `articles_per_day.png`
  - `companies_per_day.png`
  - `sentiment_per_day.png`
  - `top_companies.png`
  - `sentiment_per_company.png`
  - `topic_distribution.png`
- README.md section describing insights and findings

### Acceptance Criteria
- [ ] Script loads data from minimum 5 days of scraping
- [ ] All specified visualizations generated successfully
- [ ] Charts have proper titles, axis labels, and legends
- [ ] Color schemes are professional and accessible
- [ ] Plots saved as high-resolution PNG files (300 DPI)
- [ ] Temporal trends show clear patterns over time
- [ ] Top companies analysis includes at least top 10-20 entities
- [ ] Sentiment analysis shows meaningful distribution (not all neutral)
- [ ] README.md includes interpretation of key findings
- [ ] Code is well-commented and modular
- [ ] Script can be run independently after nlp_enriched_news.py

---

## Summary

**Total Issues:** 11 (10 required + 1 optional)
**Completed Issues:** Issue #1, Issue #2, Issue #3, Issue #4

**Project Timeline:**
1. Setup & Foundation (Issues #1, #4)
2. Data Collection (Issues #2, #3)
3. NLP Components (Issues #5, #6, #7, #8)
4. Integration & Output (Issue #9)
5. Documentation (Issue #10)
6. Optional Analytics (Issue #11)

This systematic decomposition provides a clear development path with measurable milestones and dependencies clearly defined.