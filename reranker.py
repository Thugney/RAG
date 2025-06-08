import torch
import logging
from typing import List
from sentence_transformers import CrossEncoder
from vector_store import RetrievalResult

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    A reranker class that uses a Cross-Encoder model to re-order retrieved documents.
    """
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Determine the device automatically
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing CrossEncoderReranker with model '{model_name}' on device '{device}'.")
        
        # Initialize the CrossEncoder model
        try:
            self.model = CrossEncoder(model_name, device=device)
            logger.info("Cross-Encoder model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            raise

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Reranks a list of retrieval results based on a query.
        
        Args:
            query (str): The user's original query.
            results (List[RetrievalResult]): The list of results from the initial retrieval.
            
        Returns:
            List[RetrievalResult]: The reranked list of results, sorted by relevance.
        """
        if not results or not query:
            return results

        # Create pairs of [query, document_content] for the model
        sentence_pairs = [[query, res.chunk.content] for res in results]
        
        # Predict scores for all pairs. show_progress_bar can be useful for debugging.
        logger.info(f"Reranking {len(results)} documents with Cross-Encoder...")
        scores = self.model.predict(sentence_pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        # Assign the new scores to the results
        for i, res in enumerate(results):
            res.score = scores[i] # Overwrite the initial retrieval score with the more accurate rerank score
            
        # Sort the results by the new scores in descending order
        reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        logger.info("Reranking complete.")
        return reranked_results