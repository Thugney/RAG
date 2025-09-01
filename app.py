import streamlit as st
import os
import logging
import hashlib
import json
import datetime
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit has built-in health check at /_stcore/health

# Import your powerful classes from the refactored modules
from config_loader import Config
from chunker import SmartChunker
from embedding_factory import EmbeddingFactory
from vector_store import FAISSVectorStore
from retriever import AdvancedRetriever
from generator import DeepSeekGenerator
from tools import ToolRegistry
from chat_history_db import ChatHistoryDB
from evaluator import evaluator

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Configuration ---
st.set_page_config(
    page_title="Local RAG System", 
    page_icon="⚙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .stChatMessage[data-testid="stChatMessage"] {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stButton button {
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .stSuccess {
        background-color: #f0f9f0;
        border-left: 4px solid #28a745;
    }
    
    .stInfo {
        background-color: #f0f8ff;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

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
    embedder = EmbeddingFactory(
        provider=str(_config.get('embedding.provider')),
        huggingface_model=str(_config.get('embedding.huggingface_model')),
        ollama_model=str(_config.get('embedding.ollama_model'))
    )
    
    try:
        vector_store = FAISSVectorStore(
            dimension=embedder.get_dimension(),
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
    st.header("🕒 Chat Sessions")
    
    # Session Management
    with st.expander("📋 Current Session", expanded=True):
        sessions = st.session_state.chat_db.get_all_sessions()
        if sessions:
            session_options = [f"{datetime.datetime.strptime(stime, '%Y-%m-%dT%H:%M:%S.%f').strftime('%m/%d %H:%M')} - {title}" 
                             for _, title, stime in sessions]
            selected_session = st.selectbox(
                "Select Session", 
                options=session_options,
                index=0,
                help="Switch between different chat sessions"
            )
            
            if selected_session:
                selected_index = session_options.index(selected_session)
                session_id, title, start_time = sessions[selected_index]
                if session_id != st.session_state.current_session_id:
                    st.session_state.current_session_id = session_id
                    history_msgs = st.session_state.chat_db.get_session_history(session_id)
                    st.session_state.messages = [
                        {"role": role, "content": content}
                        for role, content, _ in history_msgs
                    ]
                    st.rerun()
        else:
            st.info("No chat sessions yet. Start chatting to create one!")
    
    # Session Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Session", use_container_width=True):
            st.session_state.current_session_id = st.session_state.chat_db.start_new_session()
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Current", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.markdown("---")
    
    # History Management
    with st.expander("⚙️ History Settings", expanded=False):
        retention_days = st.slider(
            "Keep history for (days)", 
            0, 90, 7,
            help="Sessions older than this will be deleted during cleanup"
        )
        
        if st.button("🧹 Clean Old History", use_container_width=True):
            deleted_count = st.session_state.chat_db.delete_old_sessions(retention_days)
            if deleted_count > 0:
                st.success(f"Cleaned {deleted_count} old sessions")
            else:
                st.info("No old sessions to clean")
            st.rerun()
    
    st.markdown("---")
    
    # Evaluation Metrics
    with st.expander("📊 Performance", expanded=False):
        if st.button("📈 Show Metrics", use_container_width=True):
            summary = evaluator.get_summary_stats()
            if summary:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Queries", summary['total_queries'])
                    st.metric("Avg Score", f"{summary['avg_composite_score']:.3f}")
                with col2:
                    st.metric("Avg Latency", f"{summary['avg_latency']:.2f}s")
                    st.metric("Context Relevance", f"{summary['avg_context_relevance']:.3f}")
            else:
                st.info("No evaluation data yet")
        
        if st.button("📤 Export Data", use_container_width=True):
            export_path = "rag_metrics_export.json"
            evaluator.export_metrics(export_path)
            st.success(f"Metrics exported to {export_path}")
    
    st.markdown("---")
    
    # About Section
    with st.expander("ℹ️ About", expanded=False):
        st.image("https://www.eriteach.com/favicon.png", width=50)
        st.markdown("""
        **Local RAG System**  
        Interact with your documents using Retrieval-Augmented Generation.
        """)
        
        st.markdown("**Connect:**")
        st.markdown("- [YouTube](https://youtube.com/eriteach)")
        st.markdown("- [Facebook](https://facebook.com/eriteach)")
        st.markdown("- [TikTok](https://tiktok.com/eriteach)")

# --- Main Chat Interface ---
st.title("🤖 Local RAG System")
st.markdown("Chat with your documents using Ollama and your powerful RAG engine.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Compact Document Upload and Settings directly above Chat Input
with st.container():
    with st.expander("📤 Document Management", expanded=False):
        
        # Upload Section
        st.subheader("📁 Upload Documents")
        docs_path = Path("uploaded_docs")
        docs_path.mkdir(exist_ok=True)

        uploaded_files = st.file_uploader(
            "Drag and drop PDF, DOCX, MD, or TXT files here",
            type=["pdf", "docx", "md", "txt"],
            accept_multiple_files=True,
            help="Upload multiple documents to build your knowledge base"
        )

        # Show upload status
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) ready for processing")
            for file in uploaded_files:
                st.caption(f"📄 {file.name} ({file.size // 1024} KB)")

        def cleanup_old_files(docs_path, days=30):
            """Remove files older than specified days"""
            cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
            for meta_file in docs_path.glob('*.json'):
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                        upload_time = datetime.datetime.fromisoformat(meta['upload_time'])
                        if upload_time < cutoff:
                            doc_file = docs_path / meta['name']
                            if doc_file.exists():
                                doc_file.unlink()
                            meta_file.unlink()
                except Exception as e:
                    logging.warning(f"Failed to process {meta_file}: {e}")

        # Processing Button with better UX
        if uploaded_files:
            process_col1, process_col2 = st.columns([2, 1])
            with process_col1:
                if st.button("🚀 Process & Index Documents", use_container_width=True, type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        with st.spinner("Processing documents..."):
                            # Clean up old files first
                            cleanup_old_files(docs_path)
                            all_chunks = []
                            processed_count = 0
                            
                            for i, up_file in enumerate(uploaded_files):
                                status_text.text(f"Processing {up_file.name}...")
                                progress_bar.progress((i) / len(uploaded_files))
                                
                                # Check for duplicates
                                file_hash = hashlib.md5(up_file.getvalue()).hexdigest()
                                file_size = up_file.size
                                duplicate = False
                                
                                existing_files = [f for f in docs_path.glob('*.json') 
                                                if f.stem.startswith(up_file.name)]
                                for meta_file in existing_files:
                                    with open(meta_file) as f:
                                        meta = json.load(f)
                                        if meta['hash'] == file_hash and meta['size'] == file_size:
                                            duplicate = True
                                            st.warning(f"⏭️ Skipping duplicate: {up_file.name}")
                                            break
                                
                                if duplicate:
                                    continue
                                
                                # Save and process file
                                file_path = docs_path / up_file.name
                                with open(file_path, "wb") as f:
                                    f.write(up_file.getbuffer())
                                
                                # Create metadata
                                meta_path = docs_path / f"{up_file.name}.json"
                                with open(meta_path, "w") as f:
                                    json.dump({
                                        "name": up_file.name,
                                        "hash": file_hash,
                                        "size": file_size,
                                        "upload_time": datetime.datetime.now().isoformat(),
                                        "processed": False
                                    }, f)
                                
                                # Process document
                                file_chunks = chunker.process_document(str(file_path))
                                upload_time = datetime.datetime.now().isoformat()
                                for chunk in file_chunks:
                                    chunk.metadata['upload_time'] = upload_time
                                    chunk.metadata['filename'] = up_file.name
                                all_chunks.extend(file_chunks)
                                
                                # Mark as processed
                                with open(meta_path, "r+") as f:
                                    meta = json.load(f)
                                    meta['processed'] = True
                                    f.seek(0)
                                    json.dump(meta, f)
                                    f.truncate()
                                
                                processed_count += 1
                            
                            if all_chunks:
                                status_text.text("Generating embeddings...")
                                progress_bar.progress(0.8)
                                
                                embedded_chunks = embedder.embed_chunks(all_chunks)
                                
                                status_text.text("Updating vector store...")
                                progress_bar.progress(0.9)
                                
                                vector_store.add_chunks(embedded_chunks)
                                progress_bar.progress(1.0)
                                
                                st.success(f"✅ Successfully processed {processed_count} documents!")
                                st.balloons()
                            else:
                                st.error("❌ No content could be extracted from the documents")
                                
                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                        logging.error(f"Document processing error: {e}")
            
            with process_col2:
                if st.button("🗑️ Clear Uploads", use_container_width=True):
                    uploaded_files = None
                    st.rerun()
        
        st.markdown("---")
        
        # Query Settings
        st.subheader("⚙️ Query Settings")
        use_fusion = st.checkbox(
            "Enable RAG Fusion",
            value=config.get('advanced.enable_rewriting'),
            help="Generate multiple query variations for more comprehensive results"
        )
        
        # Show current vector store stats
        if vector_store and hasattr(vector_store, 'index') and vector_store.index.ntotal > 0:
            st.caption(f"📊 Vector store contains {vector_store.index.ntotal} chunks")

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
        
        # Enhanced thinking animation
        thinking_placeholder = st.empty()
        thinking_placeholder.info("💭 Analyzing your question and searching documents...")
        
        try:
            # --- RAG Pipeline Execution ---
            
            # 1. Check for Tool Use
            tool_call = tools.route_query(prompt)
            final_response = None

            if tool_call and config.get('advanced.enable_tools'):
                thinking_placeholder.info("🛠️ Executing tool...")
                tool_name = tool_call['name']
                tool_input = tool_call['input']
                logging.info(f"Tool detected: '{tool_name}' with input '{tool_input}'")
                tool_result = tools.execute_tool(tool_name, tool_input)
                
                # For simple tools, we can just display the result directly.
                final_response = f"**Tool Executed: `{tool_name}`**\n\n{tool_result}"
                thinking_placeholder.empty()
                response_container.markdown(final_response)

            else:
                # 2. Retrieve Context from Vector Store
                thinking_placeholder.info("🔍 Searching through your documents...")
                logging.info(f"Retrieving context for query: '{prompt}'")
                retrieved_chunks = retriever.retrieve(
                    query=prompt,
                    top_k=config.get('system.top_k'),
                    use_fusion=use_fusion,
                    num_variants=config.get('advanced.fusion_queries')
                )

                # Display retrieved context in an expander for transparency
                with st.expander("📚 Retrieved Sources", expanded=False):
                    if retrieved_chunks:
                        st.success(f"Found {len(retrieved_chunks)} relevant passages")
                        for i, res in enumerate(retrieved_chunks):
                            with st.expander(f"Source {i+1}: {res.chunk.metadata.get('filename', 'Unknown')} (Score: {res.score:.2f})", expanded=False):
                                st.markdown(f"```\n{res.chunk.content}\n```")
                    else:
                        st.warning("No relevant context found in your documents.")

                # 3. Generate Response using the LLM
                thinking_placeholder.info("🧠 Generating response...")
                logging.info("Generating response from LLM...")
                response_stream = generator.generate_response(prompt, retrieved_chunks)
                
                # Stream the response to the UI with better animation
                thinking_placeholder.empty()
                final_response = ""
                for chunk in response_stream:
                    final_response += chunk
                    response_container.markdown(final_response + "▌")
                response_container.markdown(final_response)
                
        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
            response_container.error(error_msg)
            final_response = error_msg
            logging.error(f"Chat error: {e}")

    # Add the final assistant response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.session_state.chat_db.save_message(st.session_state.current_session_id, "assistant", final_response)
    
    # Evaluate RAG performance if not a tool call
    if not (tool_call and config.get('advanced.enable_tools')):
        try:
            # Extract context text for evaluation
            context_texts = [chunk.chunk.content for chunk in retrieved_chunks] if retrieved_chunks else []
            
            # Perform comprehensive evaluation
            eval_metrics = evaluator.evaluate_comprehensive(
                query=prompt,
                answer=final_response,
                context=context_texts,
                latency=0.0,  # Would need actual timing measurement
                cost=0.0      # Would need actual cost calculation
            )
            
            # Log evaluation results
            logging.info(f"RAG Evaluation - Composite Score: {eval_metrics['composite_score']:.3f}")
            logging.info(f"  Answer Relevance: {eval_metrics['answer_relevance']:.3f}")
            logging.info(f"  Faithfulness: {eval_metrics['faithfulness']:.3f}")
            logging.info(f"  Context Relevance: {eval_metrics['context_relevance']:.3f}")
            
        except Exception as e:
            logging.warning(f"RAG evaluation failed: {e}")