

import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
import logging
from typing import List
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class GeminiEmbeddingModel:
    """Wrapper for Gemini embedding model with caching and batch processing"""
    
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.cache = {}
        
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if text in self.cache:
            return self.cache[text]
            
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embedding = np.array(result['embedding'], dtype=np.float32)
            self.cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(768, dtype=np.float32)  # Default dimension
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.embed_text(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    