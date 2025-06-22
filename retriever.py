

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import faiss

class AdvancedRetriever:
    """Advanced retrieval system combining multiple techniques"""
    
    def __init__(self, chunks: List[Any], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings
        
        # Initialize different retrieval methods
        self.setup_bm25()
        self.setup_tfidf()
        self.setup_faiss_index()
        
    def setup_bm25(self):
        """Setup BM25 for lexical search"""
        corpus = [chunk.content.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus)
        
    def setup_tfidf(self):
        """Setup TF-IDF vectorizer"""
        corpus = [chunk.content for chunk in self.chunks]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
    def setup_faiss_index(self):
        """Setup FAISS index for semantic search"""
        dimension = self.embeddings.shape[1]
        
        # Use HNSW index for better performance
        self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
        self.faiss_index.hnsw.efConstruction = 40
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings)
        
    def hybrid_search(self, query: str, embedding_model: Any, 
                     top_k: int = 20, alpha: float = 0.5) -> List[Tuple[int, float]]:
        """Hybrid search combining semantic and lexical retrieval"""
        
        # Semantic search using embeddings
        query_embedding = embedding_model.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        semantic_scores, semantic_indices = self.faiss_index.search(query_embedding, top_k * 2)
        semantic_scores = semantic_scores[0]
        semantic_indices = semantic_indices[0]
        
        # BM25 lexical search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # TF-IDF search
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # Combine scores
        combined_scores = {}
        
        # Add semantic scores
        for idx, score in zip(semantic_indices, semantic_scores):
            if idx < len(self.chunks):
                combined_scores[idx] = alpha * float(score)
        
        # Add lexical scores
        for idx, (bm25_score, tfidf_score) in enumerate(zip(bm25_scores, tfidf_scores)):
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * 0.5 * bm25_score
                combined_scores[idx] += (1 - alpha) * 0.5 * tfidf_score
            else:
                combined_scores[idx] = (1 - alpha) * 0.5 * bm25_score + (1 - alpha) * 0.5 * tfidf_score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]
