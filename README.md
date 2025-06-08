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

To install and set up the Local RAG System, follow these steps:

1. **Clone the Repository**:
   ```
   git clone https://github.com/Thugney/RAG.git
   cd RAG
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8 or higher installed. Then, install the required packages:
   ```
   uv pip install -r requirements.txt
   ```

3. **Configuration**:
   Edit `config.yaml` to set your preferred embedding and generation models, API keys, and other settings.

4. **Run the Application**:
   Start the Streamlit app:
   ```
   streamlit run app.py
   ```
   Open your browser and navigate to `http://localhost:8501` to access the interface.

## Usage

1. **Upload Documents**:
   Use the web interface to upload your documents (PDF, DOCX, MD, TXT formats supported).
   
2. **Query Documents**:
   Use the chat interface to ask questions or query content from your uploaded documents. The system will retrieve relevant information and generate responses based on the content.

3. **Manage Chat History**:
   Access previous chat sessions via the sidebar to revisit or continue conversations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact and Social Links

For inquiries or contributions, reach out through the following channels:

- **GitHub**: [Thugney](https://github.com/Thugney)
- **Twitter**: [YourTwitterHandle](https://twitter.com/YourTwitterHandle) (Replace with actual handle if available)
- **LinkedIn**: [YourLinkedInProfile](https://linkedin.com/in/YourLinkedInProfile) (Replace with actual profile if available)
- **Email**: Provide an email address if you wish to be contacted directly.

Feel free to contribute to the project by opening issues or submitting pull requests on GitHub.