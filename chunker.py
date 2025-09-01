import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

import PyPDF2
from docx import Document
import markdown
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

class SmartChunker:
    """Intelligent document chunking with semantic boundaries, from your script."""
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """Chunk text with semantic boundaries and overlap"""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk_words = []
        
        for sentence in sentences:
            sentence_words = sentence.split()
            if len(current_chunk_words) + len(sentence_words) > self.chunk_size and current_chunk_words:
                chunk_content = " ".join(current_chunk_words)
                chunk_metadata = {**metadata, 'chunk_index': len(chunks)}
                chunks.append(DocumentChunk(content=chunk_content, metadata=chunk_metadata))
                
                # Start new chunk with overlap
                overlap_word_count = min(len(current_chunk_words), self.overlap)
                current_chunk_words = current_chunk_words[-overlap_word_count:]

            current_chunk_words.extend(sentence_words)

        if current_chunk_words:
            chunk_content = " ".join(current_chunk_words)
            chunk_metadata = {**metadata, 'chunk_index': len(chunks)}
            chunks.append(DocumentChunk(content=chunk_content, metadata=chunk_metadata))
        
        return chunks


class EnhancedChunker(SmartChunker):
    """Enhanced chunking with entity-aware splitting, dynamic sizing, topic boundary detection, cross-chunk references, and enhanced metadata"""
    def __init__(self, chunk_size: int = 512, overlap: int = 128, use_ner: bool = True, min_chunk_size: int = 256, max_chunk_size: int = 1024, detect_topics: bool = True):
        super().__init__(chunk_size, overlap)
        self.use_ner = use_ner
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.detect_topics = detect_topics
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=['parser'])
        except Exception as e:
            logger.warning(f"Could not load spaCy model for NER: {e}. Falling back to basic chunking.")
            self.use_ner = False
            self.detect_topics = False
            
    def chunk_text(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """Chunk text with entity-aware boundaries, dynamic sizing, topic detection, cross-chunk references, enhanced metadata, and overlap"""
        if not self.use_ner:
            return super().chunk_text(text, metadata)
            
        # Process text with spaCy for entity recognition
        doc = self.nlp(text)
        sentences = list(doc.sents)
        chunks = []
        current_chunk_tokens = []
        current_entities = set()
        current_entity_types = {}
        current_complexity = 0
        current_topic = None
        topic_counter = 0
        document_id = metadata.get('source', 'doc') + '_' + str(hash(text) if len(text) > 0 else 'empty')
        content_types = set()
        
        for i, sent in enumerate(sentences):
            sent_tokens = [token.text for token in sent]
            sent_entities = {ent.text for ent in sent.ents}
            sent_entity_types = {ent.text: ent.label_ for ent in sent.ents}
            # Estimate complexity based on entity density and sentence length
            sent_complexity = len(sent_entities) * 2 + len(sent_tokens) / 10
            
            # Detect content type (rudimentary)
            if any(token.text.lower() in ['chapter', 'section', 'article'] for token in sent):
                content_types.add('structural')
            if any(ent.label_ in ['PERSON', 'ORG'] for ent in sent.ents):
                content_types.add('narrative')
            if any(ent.label_ in ['DATE', 'TIME', 'MONEY', 'QUANTITY'] for ent in sent.ents):
                content_types.add('factual')
            
            # Detect potential topic shift (rudimentary approach based on entities and keywords)
            topic_shift = False
            if self.detect_topics and i > 0 and current_topic is not None:
                prev_sent = sentences[i-1]
                prev_entities = {ent.text for ent in prev_sent.ents}
                common_entities = len(sent_entities.intersection(prev_entities))
                # If few common entities, possible topic shift
                if common_entities < len(sent_entities) * 0.3 and common_entities < len(prev_entities) * 0.3:
                    topic_shift = True
            
            # Check if adding this sentence would exceed dynamic chunk size or if topic shift detected
            current_size = len(current_chunk_tokens)
            dynamic_threshold = min(self.max_chunk_size, max(self.min_chunk_size, self.chunk_size - int(current_complexity)))
            if (current_size + len(sent_tokens) > dynamic_threshold or topic_shift) and current_chunk_tokens:
                # Avoid splitting entities if possible
                if not any(ent in current_entities for ent in sent_entities):
                    chunk_content = " ".join(current_chunk_tokens)
                    chunk_id = f"{document_id}_chunk_{len(chunks)}"
                    chunk_metadata = {
                        **metadata,
                        'chunk_index': len(chunks),
                        'chunk_id': chunk_id,
                        'entities': list(current_entities),
                        'entity_types': current_entity_types,
                        'complexity': current_complexity / max(1, current_size) if current_size > 0 else 0,
                        'topic_id': current_topic if current_topic is not None else f"topic_{topic_counter}",
                        'content_types': list(content_types),
                        'prev_chunk_id': f"{document_id}_chunk_{len(chunks)-1}" if len(chunks) > 0 else None,
                        'next_chunk_id': None  # Will be updated in the next iteration
                    }
                    # Update the next_chunk_id of the previous chunk if it exists
                    if len(chunks) > 0:
                        chunks[-1].metadata['next_chunk_id'] = chunk_id
                    
                    chunks.append(DocumentChunk(content=chunk_content, metadata=chunk_metadata))
                    
                    # Start new chunk with overlap
                    overlap_token_count = min(len(current_chunk_tokens), self.overlap)
                    current_chunk_tokens = current_chunk_tokens[-overlap_token_count:]
                    current_entities = set()
                    current_entity_types = {}
                    current_complexity = 0
                    content_types = set()
                    for token in current_chunk_tokens:
                        for ent in doc.ents:
                            if token in ent.text:
                                current_entities.add(ent.text)
                                if ent.text in sent_entity_types:
                                    current_entity_types[ent.text] = sent_entity_types[ent.text]
                    if topic_shift:
                        topic_counter += 1
                        current_topic = f"topic_{topic_counter}"
                    else:
                        current_topic = chunk_metadata['topic_id']
            
            current_chunk_tokens.extend(sent_tokens)
            current_entities.update(sent_entities)
            current_entity_types.update(sent_entity_types)
            current_complexity += sent_complexity
            if current_topic is None:
                current_topic = f"topic_{topic_counter}"
        
        if current_chunk_tokens:
            chunk_content = " ".join(current_chunk_tokens)
            current_size = len(current_chunk_tokens)
            chunk_id = f"{document_id}_chunk_{len(chunks)}"
            chunk_metadata = {
                **metadata,
                'chunk_index': len(chunks),
                'chunk_id': chunk_id,
                'entities': list(current_entities),
                'entity_types': current_entity_types,
                'complexity': current_complexity / max(1, current_size) if current_size > 0 else 0,
                'topic_id': current_topic if current_topic is not None else f"topic_{topic_counter}",
                'content_types': list(content_types),
                'prev_chunk_id': f"{document_id}_chunk_{len(chunks)-1}" if len(chunks) > 0 else None,
                'next_chunk_id': None
            }
            # Update the next_chunk_id of the previous chunk if it exists
            if len(chunks) > 0:
                chunks[-1].metadata['next_chunk_id'] = chunk_id
            
            chunks.append(DocumentChunk(content=chunk_content, metadata=chunk_metadata))
            
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        text = re.sub(r'\n+', ' ', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s and s.strip()]

    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document file into chunks"""
        file_p = Path(file_path)
        if not file_p.exists():
            raise FileNotFoundError(f"File not found: {file_p}")
        
        text = ""
        try:
            if file_p.suffix.lower() == '.pdf':
                text = self._extract_pdf_text(file_p)
            elif file_p.suffix.lower() == '.docx':
                text = self._extract_docx_text(file_p)
            elif file_p.suffix.lower() == '.md':
                text = self._extract_markdown_text(file_p)
            elif file_p.suffix.lower() == '.txt':
                text = self._extract_text_file(file_p)
            else:
                raise ValueError(f"Unsupported file type: {file_p.suffix}")
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return []

        metadata = {
            'source': str(file_p),
            'filename': file_p.name,
            'processed_at': datetime.now().isoformat()
        }
        return self.chunk_text(text, metadata)
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text
    
    def _extract_docx_text(self, file_path: Path) -> str:
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _extract_markdown_text(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            html = markdown.markdown(file.read())
            return re.sub(r'<[^>]+>', '', html)
    
    def _extract_text_file(self, file_path: Path) -> str:
        return file_path.read_text(encoding='utf-8')