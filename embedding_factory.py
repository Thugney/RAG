import logging
from typing import Union
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from chunker import DocumentChunk

logger = logging.getLogger(__name__)

class EmbeddingFactory:
    """Factory class to create embeddings using different providers"""
    
    def __init__(self, provider: str = "huggingface", 
                 huggingface_model: str = "all-MiniLM-L6-v2",
                 ollama_model: str = "mxbai-embed-large:latest"):
        self.provider = provider
        self.huggingface_model = huggingface_model
        self.ollama_model = ollama_model
        self.dimension = None
        
        if provider == "huggingface":
            self._init_huggingface()
        elif provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
    
    def _init_huggingface(self):
        """Initialize HuggingFace sentence-transformers model"""
        logger.info(f"Initializing HuggingFace embedding model: {self.huggingface_model}")
        self.model = SentenceTransformer(self.huggingface_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"HuggingFace model '{self.huggingface_model}' ready (dimension: {self.dimension})")
    
    def _init_ollama(self):
        """Initialize Ollama embedding model"""
        logger.info(f"Initializing Ollama embedding model: {self.ollama_model}")
        self.client = ollama.Client()
        
        # Test the model to get dimension
        try:
            test_response = self.client.embeddings(model=self.ollama_model, prompt="test")
            self.dimension = len(test_response['embedding'])
            logger.info(f"Ollama model '{self.ollama_model}' ready (dimension: {self.dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama model '{self.ollama_model}': {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate a normalized embedding for a single text."""
        try:
            if self.provider == "huggingface":
                embedding = self.model.encode(text, convert_to_numpy=True)
            elif self.provider == "ollama":
                response = self.client.embeddings(model=self.ollama_model, prompt=text)
                embedding = np.array(response['embedding'], dtype=np.float32)
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm != 0 else embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding with {self.provider}: {e}")
            raise
    
    def embed_chunks(self, chunks: list) -> list:
        """Generate embeddings for multiple chunks."""
        if not chunks:
            return []
            
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.provider}...")
        
        texts_to_embed = [chunk.content for chunk in chunks]
        
        if self.provider == "huggingface":
            # Use batch processing for HuggingFace
            embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True, batch_size=32)
        elif self.provider == "ollama":
            # Ollama doesn't support batch embedding, so do one by one
            embeddings = []
            for text in texts_to_embed:
                response = self.client.embeddings(model=self.ollama_model, prompt=text)
                embeddings.append(np.array(response['embedding'], dtype=np.float32))
            embeddings = np.array(embeddings)
        
        # Normalize and assign embeddings back to chunks
        for i, chunk in enumerate(chunks):
            embedding = embeddings[i]
            norm = np.linalg.norm(embedding)
            chunk.embedding = embedding / norm if norm != 0 else embedding
            
        logger.info("Embedding generation complete.")
        return chunks
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension