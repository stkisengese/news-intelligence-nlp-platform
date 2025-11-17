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

