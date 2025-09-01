import streamlit as st
import os
import logging
import hashlib
import json
import datetime
from pathlib import Path
from datetime import timedelta

# Streamlit has built-in health check at /_stcore/health

# Import your powerful classes from the refactored modules
from config_loader import Config
from chunker import SmartChunker
from embedder import OllamaEmbedder
from vector_store import FAISSVectorStore
from retriever import AdvancedRetriever
from generator import DeepSeekGenerator
from tools import ToolRegistry
from chat_history_db import ChatHistoryDB

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Configuration ---
st.set_page_config(page_title="Local RAG System", page_icon="⚙️", layout="wide")

# Initialize session state variables at the very start to avoid access issues during rerun
# Must be done right after imports and page config
if "chat_db" not in st.session_state:
    st.session_state.chat_db = ChatHistoryDB()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = st.session_state.chat_db.start_new_session()

# --- Load Configuration ---
@st.cache_resource
def load_configuration():
    return Config(config_path="config.yaml")

config = load_configuration()

# --- Initialize Core Components using Streamlit's Cache ---
@st.cache_resource
def initialize_system(_config): # <--- RENAMED 'config' to '_config'
    """Initialize all RAG components from your classes."""
    logging.info("Initializing RAG system components...")
    
    # Use the new argument name '_config' throughout the function
    embedder = OllamaEmbedder(model_name=str(_config.get('embedding.model')))
    
    try:
        vector_store = FAISSVectorStore(
            dimension=embedder.dimension,
            persist_path=_config.get('vector_store.persist_path'),
            use_gpu=_config.get('vector_store.use_gpu', False)
        )
    except Exception as e:
        st.error(f"Vector store initialization failed: {str(e)}")
        vector_store = None
        return None, None, None, None, None, None
    
    retriever = AdvancedRetriever(
        vector_store=vector_store,
        embedder=embedder,
        llm_model=_config.get('llm.model')
    )
    
    generator = DeepSeekGenerator(
        model_name=_config.get('llm.model'),
        temperature=_config.get('llm.temperature')
    )
    
    chunker = SmartChunker(
        chunk_size=_config.get('system.chunk_size'),
        overlap=_config.get('system.overlap')
    )
    
    tools = ToolRegistry()
    
    logging.info("RAG system initialized successfully.")
    return embedder, vector_store, retriever, generator, chunker, tools

# The function call remains exactly the same
embedder, vector_store, retriever, generator, chunker, tools = initialize_system(config)


# --- UI: Sidebar for Chat History Only ---
with st.sidebar:
    st.header("🕒 Chat History")
    show_history = st.checkbox("Show Chat History", value=True)
    
    if show_history:
        with st.expander("Previous Sessions", expanded=True):
            sessions = st.session_state.chat_db.get_all_sessions()
            if sessions:
                for session_id, title, start_time in sessions:
                    # Parse ISO format manually for compatibility with older Python versions
                    formatted_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d %H:%M")
                    if st.button(f"{formatted_time} - {title}", key=f"session_{session_id}"):
                        # Load selected session
                        st.session_state.current_session_id = session_id
                        history_msgs = st.session_state.chat_db.get_session_history(session_id)
                        st.session_state.messages = [
                            {"role": role, "content": content}
                            for role, content, _ in history_msgs
                        ]
                        st.rerun()
            else:
                st.info("No previous chat sessions found.")
                
        if st.button("Start New Session"):
            st.session_state.current_session_id = st.session_state.chat_db.start_new_session()
            st.session_state.messages = []
            st.rerun()
            
        st.slider("Keep history for (days)", 0, 90, 0, key="history_retention_days")
        if st.button("Clean Old History"):
            deleted_count = st.session_state.chat_db.delete_old_sessions(st.session_state.history_retention_days)
            st.info(f"Deleted {deleted_count} old sessions.")
            st.rerun()
            
        st.markdown("---")
        st.header("ℹ️ About")
        st.image("https://www.eriteach.com/favicon.png", width=50)
        st.markdown("""
        **Local RAG System** is a powerful tool for interacting with your documents using Retrieval-Augmented Generation (RAG). Upload your files, ask questions, and get insightful responses based on the content of your documents.
        """)
        st.markdown("**Connect with EriTeach:**")
        st.markdown("- [YouTube](https://youtube.com/eriteach)")
        st.markdown("- [Facebook](https://facebook.com/eriteach)")
        st.markdown("- [TikTok](https://tiktok.com/eriteach)")

# --- Main Chat Interface ---
st.title("🤖 Local RAG System")
st.markdown("Chat with your documents using Ollama and your powerful RAG engine.")

# Initialize session state variables at the very start to avoid access issues during rerun
# Must be done before any UI elements or other code that might access session state
if "chat_db" not in st.session_state:
    st.session_state.chat_db = ChatHistoryDB()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = st.session_state.chat_db.start_new_session()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Compact Document Upload and Settings directly above Chat Input
with st.container():
    with st.expander("📤 Upload & Settings", expanded=False):
        st.markdown("**Upload Documents**")
        # Use a dedicated folder for uploaded documents
        docs_path = Path("uploaded_docs")
        docs_path.mkdir(exist_ok=True)

        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, MD, or TXT files",
            type=["pdf", "docx", "md", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        def cleanup_old_files(docs_path, days=30):
            """Remove files older than specified days"""
            cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
            for meta_file in docs_path.glob('*.json'):
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                        upload_time = datetime.datetime.fromisoformat(meta['upload_time'])
                        if upload_time < cutoff:
                            # Delete both the document and its metadata
                            doc_file = docs_path / meta['name']
                            if doc_file.exists():
                                doc_file.unlink()
                            meta_file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to process {meta_file}: {e}")

        if st.button("Process and Index Documents"):
            if not uploaded_files:
                st.warning("Please upload at least one document.")
            else:
                with st.spinner("Processing documents... This may take a moment."):
                    # Clean up old files first
                    cleanup_old_files(docs_path)
                    all_chunks = []
                    for up_file in uploaded_files:
                        # Calculate file hash and size
                        file_hash = hashlib.md5(up_file.getvalue()).hexdigest()
                        file_size = up_file.size
                        
                        # Check for existing file with same hash/size
                        existing_files = [f for f in docs_path.glob('*.json')
                                        if f.stem.startswith(up_file.name)]
                        duplicate = False
                        
                        for meta_file in existing_files:
                            with open(meta_file) as f:
                                meta = json.load(f)
                                if meta['hash'] == file_hash and meta['size'] == file_size:
                                    duplicate = True
                                    st.warning(f"Skipping duplicate file: {up_file.name}")
                                    break
                        
                        if duplicate:
                            continue
                            
                        # Save file to a persistent location
                        file_path = docs_path / up_file.name
                        with open(file_path, "wb") as f:
                            f.write(up_file.getbuffer())
                            
                        # Create metadata file
                        meta_path = docs_path / f"{up_file.name}.json"
                        with open(meta_path, "w") as f:
                            json.dump({
                                "name": up_file.name,
                                "hash": file_hash,
                                "size": file_size,
                                "upload_time": datetime.datetime.now().isoformat(),
                                "processed": False
                            }, f)
                        
                        st.write(f"Chunking {up_file.name}...")
                        # Use your advanced chunker
                        file_chunks = chunker.process_document(str(file_path))
                        # Add metadata to chunks
                        upload_time = datetime.datetime.now().isoformat()
                        for chunk in file_chunks:
                            chunk.metadata['upload_time'] = upload_time
                            chunk.metadata['filename'] = up_file.name
                        all_chunks.extend(file_chunks)
                        
                        # Mark file as processed in metadata
                        with open(meta_path, "r+") as f:
                            meta = json.load(f)
                            meta['processed'] = True
                            f.seek(0)
                            json.dump(meta, f)
                            f.truncate()

                    if all_chunks:
                        st.write(f"Embedding {len(all_chunks)} chunks...")
                        # Use your embedder (with batching)
                        embedded_chunks = embedder.embed_chunks(all_chunks)
                        
                        st.write("Adding to FAISS vector store...")
                        # Use your vector store
                        vector_store.add_chunks(embedded_chunks)
                        st.success(f"Successfully indexed {len(uploaded_files)} documents!")
                    else:
                        st.error("Could not extract any content from the documents.")

        st.markdown("**Query Settings**")
        use_fusion = st.checkbox(
            "Enable RAG Fusion",
            value=config.get('advanced.enable_rewriting'),
            help="Generates multiple versions of your query to find more comprehensive results."
        )

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_db.save_message(st.session_state.current_session_id, "user", prompt)
    # Suggest a title based on first user message if session is new
    if len([m for m in st.session_state.messages if m["role"] == "user"]) == 1:
        suggested_title = st.session_state.chat_db.get_session_title_suggestion(st.session_state.current_session_id)
        st.session_state.chat_db.update_session_title(st.session_state.current_session_id, suggested_title)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        with st.spinner("Thinking..."):

            # --- RAG Pipeline Execution ---
            
            # 1. Check for Tool Use
            tool_call = tools.route_query(prompt)
            final_response = None

            if tool_call and config.get('advanced.enable_tools'):
                tool_name = tool_call['name']
                tool_input = tool_call['input']
                logging.info(f"Tool detected: '{tool_name}' with input '{tool_input}'")
                tool_result = tools.execute_tool(tool_name, tool_input)
                
                # For simple tools, we can just display the result directly.
                final_response = f"**Tool Executed: `{tool_name}`**\n\n{tool_result}"
                response_container.markdown(final_response)

            else:
                # 2. Retrieve Context from Vector Store
                logging.info(f"Retrieving context for query: '{prompt}'")
                retrieved_chunks = retriever.retrieve(
                    query=prompt,
                    top_k=config.get('system.top_k'),
                    use_fusion=use_fusion,
                    num_variants=config.get('advanced.fusion_queries')
                )

                # Display retrieved context in an expander for transparency
                with st.expander("🔍 View Retrieved Context"):
                    if retrieved_chunks:
                        for i, res in enumerate(retrieved_chunks):
                            st.info(f"**Source:** `{res.chunk.metadata.get('filename')}` (Score: {res.score:.2f})\n\n---\n{res.chunk.content}")
                    else:
                        st.warning("No relevant context was found in the knowledge base.")

                # 3. Generate Response using the LLM
                logging.info("Generating response from LLM...")
                response_stream = generator.generate_response(prompt, retrieved_chunks)
                
                # Stream the response to the UI
                final_response = response_container.write_stream(response_stream)

    # Add the final assistant response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.session_state.chat_db.save_message(st.session_state.current_session_id, "assistant", final_response)