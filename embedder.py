from time import sleep
import numpy as np
import ollama
import requests
import logging
import os
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential
from chunker import DocumentChunk  # Assuming DocumentChunk is defined elsewhere

logger = logging.getLogger(__name__)

class EmbedderFactory:
    """Factory to create appropriate embedder based on config."""
    @staticmethod
    def create_embedder(config: dict):
        provider = config.get('provider', 'ollama').lower()  # Ensure lowercase
        
        if provider == 'ollama':
            return OllamaEmbedder(
                model_name=config['ollama']['model'],
                url=config['ollama']['url']
            )
        elif provider == 'xai':
            # Handle both 'xai' and 'Xai' cases
            xai_config = config.get('xai') or config.get('Xai')
            if not xai_config:
                raise ValueError("xAI configuration not found in config")
                
            return XAIEmbedder(
                model_name=xai_config['model'],
                url=xai_config['url'],
                api_key=xai_config.get('api_key') or os.getenv('XAI_API_KEY')
        
            )
        raise ValueError(f"Unknown embedding provider: {provider}")

class OllamaEmbedder:
    """Ollama-based embedding service."""
    def __init__(self, model_name: str, url: str):
        self.model_name = model_name
        self.client = ollama.Client(host=url)
        self.dimension = None
        self._verify_model()
    
    def _verify_model(self):
        """Verify that the embedding model is available and get its dimension."""
        try:
            test_embedding = self.embed_text("test")
            self.dimension = len(test_embedding)
            logger.info(f"Embedding model '{self.model_name}' ready (dimension: {self.dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model '{self.model_name}': {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate a normalized embedding for a single text."""
        try:
            response = self.client.embeddings(model=self.model_name, prompt=text)
            embedding = np.array(response['embedding'], dtype=np.float32)
            return self._normalize(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector."""
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm != 0 else embedding

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for multiple chunks using Ollama in a batched manner."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks with Ollama...")
        
        texts_to_embed = [chunk.content for chunk in chunks]
        if not texts_to_embed:
            return []

        embeddings = []
        for text in texts_to_embed:
            try:
                response = self.client.embeddings(model=self.model_name, prompt=text)
                embedding = np.array(response["embedding"], dtype=np.float32)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error generating Ollama embedding for a chunk: {e}")
                raise

        # Normalize and assign embeddings back to chunks
        for i, chunk in enumerate(chunks):
            chunk.embedding = self._normalize(embeddings[i])
            
        logger.info("Ollama embedding generation complete.")
        return chunks

class XAIEmbedder:
    def __init__(self, model_name: str, url: str, api_key: str):
        self.model_name = model_name
        self.url = url
        self.api_key = api_key or os.getenv('XAI_API_KEY')
        self.dimension = 1024  # Adjust based on your model
        
        if not self.api_key:
            raise ValueError("xAI API key is required but not provided")
            
        self._verify_model()

    def _verify_model(self):
        """Verify that the xAI model is available and working."""
        try:
            logger.info(f"Verifying xAI model '{self.model_name}' at {self.url}")
            
            # Test with a simple embedding request
            test_embedding = self.embed_text("test", retries=1)
            
            if test_embedding is not None and len(test_embedding) > 0:
                self.dimension = len(test_embedding)
                logger.info(f"xAI model '{self.model_name}' verified successfully (dimension: {self.dimension})")
            else:
                raise ValueError("Failed to get valid embedding from xAI API")
                
        except Exception as e:
            logger.error(f"Failed to verify xAI model '{self.model_name}': {e}")
            logger.warning("xAI model verification failed - will attempt to use with fallback to Ollama")
            # Don't raise here to allow fallback behavior
            pass

    def embed_text(self, text: str, retries: int = 3, force_retry_test: bool = False) -> np.ndarray:
        """Generate a normalized embedding for a single text with retry logic."""
        if not text:
            raise ValueError("Text input for embedding cannot be empty")
        if not self.model_name:
            raise ValueError("Model name for embedding must be specified")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Updated payload structure for xAI embeddings API
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float"
        }
        
        last_error = None
        for attempt in range(retries):
            try:
                if force_retry_test and attempt < retries - 1:
                    raise Exception("Forced retry for testing backoff timing")
                    
                response = requests.post(self.url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 403:
                    logger.error(f"xAI API authentication failed (attempt {attempt+1}/{retries}): 403 Forbidden")
                    last_error = Exception("xAI API authentication failed: Invalid API key or permissions")
                    continue
                elif response.status_code == 400:
                    logger.error(f"xAI API bad request (attempt {attempt+1}/{retries}): 400 Bad Request - {response.text}")
                    last_error = Exception(f"xAI API bad request: {response.text}")
                    continue
                
                response.raise_for_status()
                
                # Parse response based on xAI API format
                response_data = response.json()
                
                if 'data' in response_data and len(response_data['data']) > 0:
                    # Standard OpenAI-compatible format
                    embedding_data = response_data['data'][0]
                    if 'embedding' in embedding_data:
                        embedding = np.array(embedding_data['embedding'], dtype=np.float32)
                        return self._normalize(embedding)
                
                raise ValueError("Invalid response format from xAI API: 'embedding' field not found in data")
                
            except Exception as e:
                last_error = e
                logger.warning(f"xAI embedding attempt {attempt+1}/{retries} failed: {str(e)}")
                if attempt < retries - 1:
                    backoff_time = 1 * (attempt + 1)
                    logger.info(f"Applying backoff of {backoff_time} seconds before retry {attempt+2}/{retries}")
                    sleep(backoff_time)
                
        logger.error(f"All {retries} xAI embedding attempts failed. Last error: {str(last_error)}")
        raise last_error

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector."""
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm != 0 else embedding

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for multiple chunks with xAI, falling back to Ollama if needed."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        texts_to_embed = [chunk.content for chunk in chunks]
        if not texts_to_embed:
            return []

        # Try xAI first
        try:
            embeddings = []
            for text in texts_to_embed:
                embedding = self.embed_text(text, force_retry_test=False)
                embeddings.append(embedding)
            
            # Assign embeddings back to chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
                
            logger.info("xAI embedding generation complete.")
            return chunks
            
        except Exception as xai_error:
            logger.warning(f"xAI embedding failed, falling back to Ollama: {str(xai_error)}")
            
            # Initialize Ollama fallback
            ollama_config = {
                'ollama': {
                    'model': 'mxbai-embed-large:latest',
                    'url': os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
                }
            }
            ollama_embedder = OllamaEmbedder(
                model_name=ollama_config['ollama']['model'],
                url=ollama_config['ollama']['url']
            )
            
            return ollama_embedder.embed_chunks(chunks)