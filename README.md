# RAGagument - Local RAG System

A powerful Retrieval-Augmented Generation system that allows you to chat with your documents using local or cloud-based AI models.

## 🚀 Features

- **Document Processing**: Support for PDF, DOCX, TXT, and MD files
- **Flexible Embeddings**: Switch between HuggingFace and Ollama embedding models
- **Vector Search**: FAISS-based semantic search with configurable parameters
- **LLM Integration**: DeepSeek API for intelligent response generation
- **Chat History**: Persistent conversation storage with SQLite
- **Streamlit UI**: Beautiful web interface with real-time updates
- **Evaluation Metrics**: Comprehensive performance tracking and analytics

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:
### Core Components
- **Frontend**: Streamlit web interface with real-time chat and document upload
- **Application**: Python orchestration layer managing the RAG pipeline
- **Processing**: Intelligent document chunking and multi-provider embedding generation
- **Retrieval**: FAISS vector search with configurable parameters and RAG fusion
- **Generation**: DeepSeek LLM integration with streaming responses
- **Data**: SQLite chat history + local file storage for documents and vectors

### Key Features
- **Flexible Embeddings**: Switch between HuggingFace and Ollama providers via config
- **Advanced Retrieval**: RAG fusion, query expansion, and hybrid search capabilities
- **Real-time UI**: Streamlit-based interface with live updates and progress indicators
- **Session Management**: Persistent chat history with automatic cleanup
- **Evaluation System**: Comprehensive performance metrics and quality assessment

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Thugney/rag.git
   cd rag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv rag_venv
   source rag_venv/bin/activate  # On Windows: rag_venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Embedding provider (HuggingFace or Ollama)
- Chunk size and overlap
- Vector store settings
- LLM parameters
- Advanced features like RAG fusion

## 🎯 Usage

1. **Upload Documents**: Use the upload section to add PDF, DOCX, TXT, or MD files
2. **Process Documents**: Click "Process & Index" to make documents searchable
3. **Ask Questions**: Use the chat interface to ask questions about your documents
4. **Manage Sessions**: Switch between chat sessions or start new ones

## 🔧 Advanced Features

- **RAG Fusion**: Generate multiple query variants for comprehensive results
- **Tool Integration**: Built-in tools for enhanced functionality
- **Performance Metrics**: Real-time evaluation and analytics
- **Session Management**: Persistent chat history with cleanup options

## 📊 Evaluation

The system includes comprehensive evaluation metrics:
- Answer relevance
- Faithfulness to source material
- Context relevance
- Response latency
- Cost tracking

## 🛠️ Development

### Project Structure
```
RAGagument/
├── app.py                 # Main application
├── config.yaml           # Configuration file
├── requirements.txt      # Dependencies
├── .gitignore           # Git exclusion rules
├── .env                 # Environment variables (git ignored)
├── uploaded_docs/       # Uploaded documents (git ignored)
└── vector_db/           # FAISS vector store (git ignored)
```

### Key Files
- `app.py` - Main Streamlit application
- `config_loader.py` - Configuration management
- `embedding_factory.py` - Multi-provider embedding system
- `vector_store.py` - FAISS vector database integration
- `retriever.py` - Advanced retrieval with RAG fusion
- `generator.py` - DeepSeek LLM integration
- `chat_history_db.py` - SQLite chat history management
- `chunker.py` - Document processing and chunking
- `evaluator.py` - Performance evaluation system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions, please check:
- [Issue Tracker](https://github.com/Thugney/rag/issues)
- [Discussion Forum](https://github.com/Thugney/rag/discussions)

## 🔗 Links

- [DeepSeek API](https://platform.deepseek.com/)
- [HuggingFace Models](https://huggingface.co/models)
- [Ollama](https://ollama.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
