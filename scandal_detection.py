from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from entity_detection import extract_organizations
import numpy as np

def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Loads a pre-trained sentence-transformer model.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        SentenceTransformer: The loaded sentence-transformer model.
    """
    return SentenceTransformer(model_name)

def define_disaster_keywords():
    """Defines a list of environmental disaster keywords.

    Returns:
        list: A list of environmental disaster keywords.
    """
    return [
        "environmental disaster",
        "oil spill",
        "chemical spill",
        "toxic waste",
        "deforestation",
        "pollution",
        "emissions scandal",
        "nuclear meltdown",
        "industrial accident",
        "environmental damage",
        "ecological crisis",
        "water contamination",
        "soil contamination",
        "air pollution",
        "wildfire",
        "illegal logging",
        "ocean dumping",
        "hazardous material",
        "environmental negligence",
        "gas leak"
    ]

def extract_entity_sentences(text, entities):
    """Extracts sentences from the text that contain any of the given entities.

    Args:
        text (str): The text to extract sentences from.
        entities (list): A list of entities to look for.

    Returns:
        list: A list of sentences containing the entities.
    """
    sentences = []
    for sentence in text.split('.'):
        for entity in entities:
            if entity in sentence:
                sentences.append(sentence)
                break
    return sentences

def compute_embeddings(texts, model):
    """Computes embeddings for a list of texts.

    Args:
        texts (list): A list of texts to embed.
        model (SentenceTransformer): The sentence-transformer model to use.

    Returns:
        np.ndarray: The computed embeddings.
    """
    return model.encode(texts)

def calculate_similarity_scores(keyword_embeddings, sentence_embeddings):
    """Calculates the cosine similarity between keyword and sentence embeddings.

    Args:
        keyword_embeddings (np.ndarray): The embeddings of the disaster keywords.
        sentence_embeddings (np.ndarray): The embeddings of the sentences.

    Returns:
        np.ndarray: The cosine similarity scores.
    """
    return cosine_similarity(keyword_embeddings, sentence_embeddings)
