"""
Lazy loading utilities for heavy imports to improve application startup time.
"""
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_sentence_model():
    """Lazy load SentenceTransformer model - only import when first used."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError as e:
        logger.error("Failed to import sentence_transformers: %s", e)
        raise ImportError("sentence_transformers is required but not installed") from e

@lru_cache(maxsize=1)
def get_faiss():
    """Lazy load faiss library - only import when first used."""
    try:
        import faiss
        logger.info("Loaded faiss library")
        return faiss
    except ImportError as e:
        logger.error("Failed to import faiss: %s", e)
        raise ImportError("faiss is required but not installed") from e

@lru_cache(maxsize=1)
def get_sklearn_cosine_similarity():
    """Lazy load sklearn cosine similarity - only import when first used."""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        logger.info("Loaded sklearn cosine_similarity")
        return cosine_similarity
    except ImportError as e:
        logger.error("Failed to import sklearn: %s", e)
        raise ImportError("scikit-learn is required but not installed") from e

@lru_cache(maxsize=1)
def get_transformers():
    """Lazy load transformers library - only import when first used."""
    try:
        import transformers
        logger.info("Loaded transformers library")
        return transformers
    except ImportError as e:
        logger.error("Failed to import transformers: %s", e)
        raise ImportError("transformers is required but not installed") from e

@lru_cache(maxsize=1)
def get_torch():
    """Lazy load PyTorch - only import when first used."""
    try:
        import torch
        logger.info("Loaded PyTorch")
        return torch
    except ImportError as e:
        logger.error("Failed to import torch: %s", e)
        raise ImportError("torch is required but not installed") from e

@lru_cache(maxsize=1)
def get_numpy():
    """Lazy load numpy - only import when first used."""
    try:
        import numpy
        logger.info("Loaded numpy")
        return numpy
    except ImportError as e:
        logger.error("Failed to import numpy: %s", e)
        raise ImportError("numpy is required but not installed") from e

@lru_cache(maxsize=1)
def get_pandas():
    """Lazy load pandas - only import when first used."""
    try:
        import pandas
        logger.info("Loaded pandas")
        return pandas
    except ImportError as e:
        logger.error("Failed to import pandas: %s", e)
        raise ImportError("pandas is required but not installed") from e

# Convenience function to clear all caches if needed
def clear_all_caches():
    """Clear all lazy loading caches - useful for testing or memory cleanup."""
    get_sentence_model.cache_clear()
    get_faiss.cache_clear()
    get_sklearn_cosine_similarity.cache_clear()
    get_transformers.cache_clear()
    get_torch.cache_clear()
    get_numpy.cache_clear()
    get_pandas.cache_clear()
    logger.info("Cleared all lazy loading caches") 