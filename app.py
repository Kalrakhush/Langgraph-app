# streamlit_app.py
import streamlit as st
import os
import tempfile
from src.graph.graph import AIProcessingGraph
from src.services.pdf_service import PDFService
from src.services.vector_store import VectorStore
from src.config import Config

# Page config
st.set_page_config(
    page_title="AI Pipeline Demo",
    page_icon="🤖",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

def setup_ai_pipeline():
    """Setup AI pipeline components."""
    try:
        Config.validate()
        graph = AIProcessingGraph()
        return graph, None
    except Exception as e:
        return None, str(e)

def process_pdf(uploaded_file):
    """Process uploaded PDF file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process PDF
        pdf_service = PDFService()
        result = pdf_service.process_pdf(tmp_path)
        
        if result["chunks"]:
            # Store in vector database
            vector_store = VectorStore()
            success = vector_store.store_embeddings(
                result["chunks"],
                result["embeddings"],
                {"filename": uploaded_file.name}
            )
            
            if success:
                st.session_state.pdf_processed = True
                return f"✅ Successfully processed {len(result['chunks'])} chunks from {uploaded_file.name}"
            else:
                return "❌ Failed to store PDF embeddings"
        else:
            return "❌ No text extracted from PDF"
        
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"
    finally:
        # Clean up temporary file
        if 'tmp_path' in locals():
            os.unlink(tmp_path)

def main():
    """Main Streamlit application."""
    st.title("🤖 AI Pipeline Demo")
    st.markdown("### Weather API + PDF RAG System with LangGraph")
    
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check API keys
        api_keys_valid = all([
            Config.GROQ_API_KEY,
            Config.OPENWEATHERMAP_API_KEY
        ])
        
        if api_keys_valid:
            st.success("✅ API keys configured")
        else:
            st.error("❌ Missing API keys")
            st.markdown("""
            Please set the following environment variables:
            - `GROQ_API_KEY`
            - `OPENWEATHERMAP_API_KEY`
            - `QDRANT_URL` and `QDRANT_API_KEY` (optional - uses in-memory fallback)
            - `LANGSMITH_API_KEY` (optional)
            """)
            return
        
        # PDF Upload
        st.header("📄 PDF Upload")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file and not st.session_state.pdf_processed:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    result = process_pdf(uploaded_file)
                    st.write(result)
        
        if st.session_state.pdf_processed:
            st.success("✅ PDF ready for queries")
        
        # Vector Store Info
        if st.session_state.pdf_processed:
            try:
                vector_store = VectorStore()
                info = vector_store.get_collection_info()
                if info:
                    st.info(f"📊 Vector DB: {info.get('points_count', 0)} embeddings")
            except:
                st.warning("Could not connect to vector database")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Initialize AI pipeline
    if st.session_state.graph is None:
        with st.spinner("Initializing AI pipeline..."):
            graph, error = setup_ai_pipeline()
            if graph:
                st.session_state.graph = graph
                st.success("✅ AI pipeline ready")
            else:
                st.error(f"❌ Failed to initialize: {error}")
                return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("🔍 View Details"):
                    metadata = message["metadata"]
                    st.json({
                        "Intent": metadata.get("intent", "unknown"),
                        "Weather Data": bool(metadata.get("weather_data")),
                        "Retrieved Docs": metadata.get("retrieved_docs_count", 0),
                        "Processing Info": metadata.get("metadata", {})
                    })
    
    # Chat input
    if prompt := st.chat_input("Ask about weather or PDF content..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    result = st.session_state.graph.process_query(prompt)
                    response = result["response"]
                    
                    st.markdown(response)
                    
                    # Store assistant message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "metadata": result
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Sample queries
    st.header("💡 Sample Queries")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌤️ Weather Queries")
        weather_samples = [
            "What's the weather in London?",
            "How's the temperature in New York?",
            "Is it raining in Tokyo?"
        ]
        for sample in weather_samples:
            if st.button(sample, key=f"weather_{sample}"):
                st.session_state.messages.append({"role": "user", "content": sample})
                st.rerun()
    
    with col2:
        st.subheader("📄 PDF Queries")
        if st.session_state.pdf_processed:
            pdf_samples = [
                "What is the main topic of the document?",
                "Summarize the key points",
                "What are the conclusions?"
            ]
            for sample in pdf_samples:
                if st.button(sample, key=f"pdf_{sample}"):
                    st.session_state.messages.append({"role": "user", "content": sample})
                    st.rerun()
        else:
            st.info("Upload a PDF to see sample queries")

if __name__ == "__main__":
    main()