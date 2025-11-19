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
