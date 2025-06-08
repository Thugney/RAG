import logging
from typing import List, Iterator
from vector_store import RetrievalResult
from huggingface_hub import InferenceClient
import os

logger = logging.getLogger(__name__)

class HuggingFaceGenerator:
    """
    A generator class using the Hugging Face Inference API.
    This version uses the API endpoint instead of local model loading.
    """
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """
        Initializes the Hugging Face Inference API client.
        Requires HUGGINGFACE_API_KEY environment variable.
        """
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
            
        self.client = InferenceClient(
            model=self.model_name,
            token=api_key
        )
        logger.info(f"Initialized HuggingFace Inference API client for model: {self.model_name}")


    def generate_response(self, query: str, context_chunks: List[RetrievalResult]) -> Iterator[str]:
        """
        Generates a streamed response using the Hugging Face Inference API.
        """
        if not self.client:
            yield "HuggingFace API client is not available."
            return
            
        if not context_chunks:
            yield "I could not find any relevant information in the uploaded documents to answer your question."
            return

        context_text = self._format_context(context_chunks)
        
        # Format the prompt for the API
        prompt = f"""You are a helpful and factual AI assistant.
Answer the user's query based ONLY on the provided context.

CONTEXT:
{context_text}

QUERY:
{query}

ANSWER:"""
        
        # Stream the response from the API
        for token in self.client.text_generation(
            prompt,
            temperature=self.temperature,
            max_new_tokens=1024,
            stream=True,
            details=False
        ):
            yield token

    def _format_context(self, context_chunks: List[RetrievalResult]) -> str:
        """Formats the context chunks for the prompt."""
        formatted_context = []
        for res in context_chunks:
            source = res.chunk.metadata.get('filename', 'Unknown Source')
            content = res.chunk.content
            formatted_context.append(f"Source: {source}\nContent: {content}")
        return "\n\n---\n\n".join(formatted_context)