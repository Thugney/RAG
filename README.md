# Local RAG System

A local Retrieval-Augmented Generation (RAG) system for document interaction using various embedding and generation models.

## Project Goal

Create a local RAG system that enables users to upload documents, process them into a searchable vector database, and query the content through a chat interface with AI-generated responses augmented by document retrieval. The system is designed to run locally, ensuring privacy and control over data.

## Key Features

- Document upload and processing (PDF, DOCX, MD, TXT)
- Vector embeddings with multiple provider support (Ollama, xAI)
- Advanced retrieval with hybrid search, query expansion, and re-ranking
- Response generation using configurable LLM models (DeepSeek, HuggingFace)
- Chat history storage and session management
- Streamlit-based user interface for interaction

## Tech Stack

- **Backend**: Python, Streamlit, FAISS (vector store), Ollama (local embeddings/LLM), xAI (cloud embeddings/LLM), DeepSeek API, HuggingFace models
- **Frontend**: Streamlit for web UI, providing chat interface and document management

## Overall Architecture

- Modular Python backend with separated concerns (chunking, embedding, retrieval, generation)
- Streamlit for interactive frontend UI
- Local vector database (FAISS) for document storage
- Configurable embedding and generation providers

## Installation

Instructions for setting up the project will be added here.

## Usage

Instructions for using the application will be added here.

## License

This project is licensed under the terms of the LICENSE file.