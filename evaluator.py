import logging
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Comprehensive RAG evaluation framework with RAGAS-inspired metrics"""
    
    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.metrics_history = []
        
    def evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """Measure how relevant the answer is to the query (0-1)"""
        query_embedding = self.similarity_model.encode(query, convert_to_tensor=True)
        answer_embedding = self.similarity_model.encode(answer, convert_to_tensor=True)
        
        similarity = util.pytorch_cos_sim(query_embedding, answer_embedding).item()
        return max(0, min(1, similarity))  # Clamp to 0-1 range
    
    def evaluate_faithfulness(self, answer: str, context: List[str]) -> float:
        """Measure how faithful the answer is to the provided context (0-1)"""
        if not context:
            return 0.0
            
        # Simple approach: check if key facts from answer are supported by context
        answer_sentences = self._split_into_sentences(answer)
        context_text = " ".join(context)
        
        supported_facts = 0
        total_facts = 0
        
        for sentence in answer_sentences:
            if self._is_factual_sentence(sentence):
                total_facts += 1
                if self._is_supported(sentence, context_text):
                    supported_facts += 1
        
        return supported_facts / total_facts if total_facts > 0 else 1.0
    
    def evaluate_context_relevance(self, query: str, context: List[str]) -> float:
        """Measure how relevant the retrieved context is to the query (0-1)"""
        if not context:
            return 0.0
            
        query_embedding = self.similarity_model.encode(query, convert_to_tensor=True)
        context_embeddings = self.similarity_model.encode(context, convert_to_tensor=True)
        
        similarities = util.pytorch_cos_sim(query_embedding, context_embeddings)
        avg_similarity = similarities.mean().item()
        
        return max(0, min(1, avg_similarity))
    
    def evaluate_comprehensive(self, query: str, answer: str, context: List[str], 
                             latency: float, cost: float = 0.0) -> Dict:
        """Comprehensive evaluation with all metrics"""
        metrics = {
            'answer_relevance': self.evaluate_answer_relevance(query, answer),
            'faithfulness': self.evaluate_faithfulness(answer, context),
            'context_relevance': self.evaluate_context_relevance(query, context),
            'latency_seconds': latency,
            'cost_usd': cost,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer_length': len(answer),
            'context_count': len(context)
        }
        
        # Calculate composite score
        metrics['composite_score'] = (
            metrics['answer_relevance'] * 0.4 +
            metrics['faithfulness'] * 0.4 + 
            metrics['context_relevance'] * 0.2
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from historical metrics"""
        if not self.metrics_history:
            return {}
            
        df = pd.DataFrame(self.metrics_history)
        
        return {
            'total_queries': len(df),
            'avg_composite_score': df['composite_score'].mean(),
            'avg_answer_relevance': df['answer_relevance'].mean(),
            'avg_faithfulness': df['faithfulness'].mean(),
            'avg_context_relevance': df['context_relevance'].mean(),
            'avg_latency': df['latency_seconds'].mean(),
            'p95_latency': df['latency_seconds'].quantile(0.95),
            'total_cost': df['cost_usd'].sum()
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump({
                'metrics': self.metrics_history,
                'summary': self.get_summary_stats(),
                'export_date': datetime.now().isoformat()
            }, f, indent=2)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        return [s.strip() for s in text.split('.') if s.strip()]
    
    def _is_factual_sentence(self, sentence: str) -> bool:
        """Check if sentence contains factual information"""
        # Simple heuristic: sentences with numbers or specific patterns
        has_numbers = any(char.isdigit() for char in sentence)
        has_factual_indicators = any(keyword in sentence.lower() for keyword in 
                                   ['is', 'are', 'was', 'were', 'has', 'have', 'contains'])
        return has_numbers or has_factual_indicators
    
    def _is_supported(self, fact: str, context: str) -> bool:
        """Check if a factual statement is supported by context"""
        try:
            fact_embedding = self.similarity_model.encode(fact, convert_to_tensor=True)
            context_embedding = self.similarity_model.encode(context, convert_to_tensor=True)
            
            similarity = util.pytorch_cos_sim(fact_embedding, context_embedding).item()
            return similarity > 0.4  # Lower threshold for better support detection
        except:
            # Fallback: simple keyword matching if embedding fails
            fact_words = set(fact.lower().split())
            context_words = set(context.lower().split())
            return len(fact_words.intersection(context_words)) / len(fact_words) > 0.3

# Global evaluator instance
evaluator = RAGEvaluator()