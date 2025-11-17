import spacy
from spacy.cli.download import download as spacy_download

def load_spacy_model(model_name='en_core_web_sm'):
    """
    Loads a spaCy model, downloading it if not found.
    """
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"spaCy model '{model_name}' not found. Downloading...")
        spacy_download(model_name)
        nlp = spacy.load(model_name)
    return nlp

def detect_entities(text, nlp, entity_type='ORG'):
    """
    Detects entities of a specific type in a text.
    """
    if not text or not isinstance(text, str):
        return []
    
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == entity_type]
    return entities

def extract_organizations(headline, body, nlp):
    """
    Extracts a deduplicated list of organizations from headline and body.
    """
    headline_orgs = detect_entities(headline, nlp, 'ORG')
    body_orgs = detect_entities(body, nlp, 'ORG')
    
    all_orgs = headline_orgs + body_orgs
    
    # Deduplicate while preserving order (case-insensitive)
    seen = set()
    deduplicated_orgs = []
    for org in all_orgs:
        if org.lower() not in seen:
            seen.add(org.lower())
            deduplicated_orgs.append(org)
            
    return deduplicated_orgs

if __name__ == '__main__':
    # Load the spaCy model
    nlp_model = load_spacy_model()

    # Example usage
    sample_headline = "Apple and Google are competing for the best talent."
    sample_body = "Google's new AI is impressive. Meanwhile, Apple announced a new product."

    organizations = extract_organizations(sample_headline, sample_body, nlp_model)

    print(f"Detected {len(organizations)} companies which are {', '.join(organizations)}")
