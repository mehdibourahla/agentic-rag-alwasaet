import streamlit as st
import os
import asyncio
from pathlib import Path
import tempfile
from typing import List, Dict

from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine


# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize all system components"""
    config = Config()
    vector_store = VectorStore(config)
    doc_processor = DocumentProcessor()
    rag_engine = RAGEngine(config, vector_store)
    return config, vector_store, doc_processor, rag_engine


def format_citations(citations: List[Dict]) -> str:
    if not citations:
        return ""
    
    formatted = "\n\n**Citations:**\n"
    for i, citation in enumerate(citations, 1):
        formatted += f"[{i}] {citation['filename']}, Page {citation['page']}\n"
    return formatted


async def process_uploaded_files(uploaded_files, doc_processor, vector_store):
    results = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            # Process the PDF
            documents = await doc_processor.process_pdf(
                tmp_path, 
                uploaded_file.name
            )
            
            # Add to vector store
            success = await vector_store.add_documents(documents)
            
            results.append({
                "filename": uploaded_file.name,
                "success": success,
                "chunks": len(documents)
            })
        except Exception as e:
            results.append({
                "filename": uploaded_file.name,
                "success": False,
                "error": str(e)
            })
        finally:
            os.unlink(tmp_path)
    
    return results


async def handle_query(query: str, rag_engine, conversation_history):
    # Get last 5 exchanges for context
    context = conversation_history[-10:] if conversation_history else []
    
    # Query the RAG engine
    result = await rag_engine.query(query, context)
    
    return result


def main():
    st.title("🤖 Agentic RAG System")
    st.markdown("Upload PDFs and ask questions - get answers with precise citations")
    
    # Initialize components
    try:
        config, vector_store, doc_processor, rag_engine = init_components()
        
        # Connection status
        if asyncio.run(vector_store.check_connection()):
            st.success("✅ Connected to Vector Database")
        else:
            st.error("❌ Failed to connect to Vector Database")
            st.stop()
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        st.stop()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Upload")
        
        # Multiple file upload
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents to query"
        )
        
        if uploaded_files:
            if st.button("📤 Process Uploaded Files"):
                with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                    try:
                        results = asyncio.run(
                            process_uploaded_files(uploaded_files, doc_processor, vector_store)
                        )
                        
                        # Display results
                        success_count = sum(1 for r in results if r["success"])
                        st.success(f"✅ Successfully processed {success_count}/{len(results)} files")
                        
                        # Show details in expander
                        with st.expander("Processing Details"):
                            for result in results:
                                if result["success"]:
                                    st.write(f"✅ **{result['filename']}**: {result['chunks']} chunks")
                                else:
                                    st.write(f"❌ **{result['filename']}**: {result.get('error', 'Unknown error')}")
                        
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
        
        # Display indexed documents count
        doc_count = asyncio.run(vector_store.get_document_count())
        st.metric("Documents in Database", doc_count)
        
        # Clear database button
        if st.button("🗑️ Clear All Documents"):
            if asyncio.run(vector_store.clear_collection()):
                st.success("Database cleared")
                st.rerun()
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Initialize conversation history in session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Chat history display
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Query the RAG system
                    result = asyncio.run(
                        handle_query(
                            prompt, 
                            rag_engine, 
                            st.session_state.conversation_history
                        )
                    )
                    
                    # Format response
                    if result["answer"] == "No answer found in the uploaded documents.":
                        response = result["answer"]
                    else:
                        response = result["answer"] + format_citations(result["citations"])
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to history
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Show confidence score in expander
                    with st.expander("📊 Response Details"):
                        st.metric("Confidence Score", f"{result['confidence']:.2%}")
                        
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()