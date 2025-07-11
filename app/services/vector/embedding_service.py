"""
Embedding service using GTE Multilingual models
"""
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from loguru import logger

from app.core.config import settings


class EmbeddingService:
    """
    Embedding service using GTE Multilingual models
    
    Features:
    - Text embedding with gte-multilingual-base
    - Reranking with gte-multilingual-reranker-base
    - Batch processing for efficiency
    - Caching for repeated queries
    """
    
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        reranker_model_name: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize embedding service
        
        Args:
            embedding_model_name: Embedding model name
            reranker_model_name: Reranker model name
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.embedding_model_name = embedding_model_name or settings.embedding_model
        self.reranker_model_name = reranker_model_name or settings.reranker_model
        self.device = device
        
        # Initialize models lazily
        self._embedding_model = None
        self._reranker_model = None
        
        # Simple cache for embeddings
        self._embedding_cache = {}
        self._cache_limit = 1000
        
        logger.info(f"Initialized EmbeddingService with models:")
        logger.info(f"  Embedding: {self.embedding_model_name}")
        logger.info(f"  Reranker: {self.reranker_model_name}")
        logger.info(f"  Device: {self.device}")
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
                trust_remote_code=True
            )
            logger.info("✅ Embedding model loaded successfully")
        
        return self._embedding_model
    
    @property
    def reranker_model(self) -> CrossEncoder:
        """Lazy-load cross-encoder reranker model"""
        if self._reranker_model is None:
            logger.info(f"Loading reranker model: {self.reranker_model_name}")
            self._reranker_model = CrossEncoder(
                self.reranker_model_name,
                device=self.device,
                trust_remote_code=True  
            )
            logger.info("✅ Reranker model loaded successfully")
        return self._reranker_model

    
    async def embed_text(self, text: str) -> List[float]:
        """
        Create embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        # Check cache first
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            logger.debug("Cache hit for embedding")
            return self._embedding_cache[cache_key]
        
        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_tensor=False, trust_remote_code=True)
        embedding_list = embedding.tolist()
        
        # Cache the result (with limit)
        if len(self._embedding_cache) < self._cache_limit:
            self._embedding_cache[cache_key] = embedding_list
        
        return embedding_list
    
    async def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Create embeddings for multiple texts efficiently
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_tensor=False,
                show_progress_bar=False,
                trust_remote_code=True
            )
            
            # Convert to list format
            for embedding in batch_embeddings:
                all_embeddings.append(embedding.tolist())
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        logger.info(f"✅ Created {len(all_embeddings)} embeddings")
        return all_embeddings
    
    async def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents using the reranker model
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return
            
        Returns:
            List of (document_index, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        logger.info(f"Reranking {len(documents)} documents for query")
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get reranking scores
        scores = self.reranker_model.predict(pairs, convert_to_tensor=False)
        
        # Create (index, score) pairs
        scored_docs = [(i, float(score)) for i, score in enumerate(scores)]
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit
        if top_k is not None:
            scored_docs = scored_docs[:top_k]
        
        logger.info(f"✅ Reranked documents, top score: {scored_docs[0][1]:.4f}")
        
        return scored_docs
    
    async def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = await self.embed_texts([text1, text2])
        
        # Calculate cosine similarity
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Convert to 0-1 scale
        similarity = (cosine_sim + 1) / 2
        
        return float(similarity)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        
        Returns:
            Embedding dimension
        """
        # Get dimension from model
        return self.embedding_model.get_sentence_embedding_dimension()
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._embedding_cache),
            "cache_limit": self._cache_limit,
            "embedding_model": self.embedding_model_name,
            "reranker_model": self.reranker_model_name,
            "device": self.device
        } 