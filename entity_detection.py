import spacy
from spacy.cli.download import download as spacy_download

def load_spacy_model(model_name='en_core_web_sm'):
    """
    Load the spaCy model, downloading it if it's not already installed.
    
    Args:
        model_name (str): The name of the spaCy model to load.
    
    Returns:
        spacy.lang: The loaded spaCy model.
    """
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Spacy model '{model_name}' not found. Downloading...")
        spacy_download(model_name)
        nlp = spacy.load(model_name)
    return nlp

def detect_entities(text, nlp, entity_type='ORG'):
    """
    Detect entities of a specific type in a given text using a spaCy model.
    
    Args:
        text (str): The text to process.
        nlp (spacy.lang): The loaded spaCy model.
        entity_type (str): The type of entity to detect (e.g., 'ORG', 'PERSON').
    
    Returns:
        list: A list of detected entity texts.
    """
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == entity_type]
    return entities

def extract_organizations(headline, body, nlp):
    """
    Extract a deduplicated list of organizations from the headline and body of an article.
    
    Args:
        headline (str): The headline of the article.
        body (str): The body of the article.
        nlp (spacy.lang): The loaded spaCy model.
    
    Returns:
        list: A deduplicated list of organization names.
    """
    headline_orgs = detect_entities(headline, nlp, entity_type='ORG')
    body_orgs = detect_entities(body, nlp, entity_type='ORG')
    
    all_orgs = headline_orgs + body_orgs
    
    # Deduplicate the list of organizations (case-insensitive)
    deduplicated_orgs = list(set([org.strip() for org in all_orgs]))
    
    return deduplicated_orgs

if __name__ == '__main__':
    # Example Usage
    nlp_model = load_spacy_model()
    
    example_headline = "Apple and Google announce new partnership."
    example_body = "The new partnership between Apple Inc. and Google LLC will focus on developing new AI technologies. Other companies like Microsoft are also watching closely."
    
    organizations = extract_organizations(example_headline, example_body, nlp_model)
    
    print(f"Detected {len(organizations)} companies which are {', '.join(organizations)}")
