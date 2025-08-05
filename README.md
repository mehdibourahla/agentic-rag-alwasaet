# 🤖 Agentic RAG System

A sophisticated Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents, ask questions, and receive accurate answers with precise citations. Features advanced multi-language support, OCR capabilities, and hybrid search for optimal retrieval performance.

## ✨ Key Features

### 📄 Document Processing
- **Multi-format PDF Support**: Handles text-based and image-based PDFs
- **Advanced OCR**: Automatic fallback to OCR for garbled or image-based text
- **Multi-language Support**: Native Arabic and English processing with language detection
- **Smart Text Extraction**: Automatic quality detection and extraction method selection
- **Duplicate Prevention**: Content-based hashing to prevent duplicate indexing

### 🔍 Intelligent Search & Retrieval
- **Hybrid Search**: Combines semantic similarity (OpenAI embeddings) and keyword matching (BM25)
- **Language-Aware Retrieval**: Prioritizes documents matching query language
- **Confidence Scoring**: Adaptive confidence thresholds to prevent hallucinations
- **Citation Tracking**: Precise page-level citations with source filenames

### 💬 Conversational Interface
- **Context-Aware**: Maintains conversation history for follow-up questions
- **Multi-language Queries**: Supports Arabic and English questions
- **Streaming Responses**: Real-time response generation
- **Error Handling**: Graceful handling of edge cases and errors

### 🚀 Production-Ready Deployment
- **Docker Support**: Complete containerization with Docker Compose
- **Scalable Architecture**: Modular design for easy scaling
- **Health Monitoring**: Connection status and system health checks
- **Configuration Management**: Environment-based configuration

## 🏗️ Architecture Overview

```
┌─────────────────────┐
│  Streamlit Web UI   │ ← User Interface & Chat
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Document Processor │ ← PDF parsing, OCR, language detection
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Hybrid Search     │ ← Semantic (OpenAI) + BM25 keyword search
│   Vector Store      │ ← Qdrant vector database
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│    RAG Engine       │ ← LlamaIndex + OpenAI LLM
│  Language-Aware     │ ← Context building & response generation
└─────────────────────┘
```

### 🔧 Core Components

1. **Document Processor** (`src/document_processor.py`)
   - PDF text extraction with pdfplumber
   - OCR fallback using Tesseract for Arabic/English
   - Language detection and quality assessment
   - Smart chunking with metadata preservation

2. **Vector Store** (`src/vector_store.py`)
   - Qdrant vector database integration
   - OpenAI embeddings for semantic search
   - BM25 index for keyword matching
   - Hybrid result ranking and deduplication

3. **RAG Engine** (`src/rag_engine.py`)
   - Multi-language query processing
   - Context-aware response generation
   - Citation extraction and formatting
   - Hallucination prevention through grounding verification

4. **Configuration** (`src/config.py`)
   - Environment-based settings management
   - Model and parameter configuration
   - Deployment flexibility

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key

### 🐳 Docker Deployment (Recommended)

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd agentic-rag
   cp .env.example .env
   ```

2. **Configure Environment**
   Edit `.env` and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Deploy with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Access Application**
   Open your browser to `http://localhost:8501`

### 💻 Local Development

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Install System Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng poppler-utils
   
   # macOS
   brew install tesseract tesseract-lang poppler
   ```

3. **Start Services**
   ```bash
   # Start Qdrant
   docker run -p 6333:6333 qdrant/qdrant
   
   # Run application
   streamlit run app.py
   ```

## 📖 Usage Guide

### 1. Document Upload
- Use the sidebar to upload PDF files (supports multiple files)
- The system automatically detects text quality and applies OCR when needed
- View processing status and chunk count for each document

### 2. Asking Questions
- Type questions in Arabic or English
- The system automatically detects query language
- Responses include precise citations with filename and page numbers

### 3. Follow-up Questions
- Continue conversations naturally
- The system maintains context from previous exchanges
- Reference previous answers using "it", "that", etc.

### 4. Database Management
- View total indexed document count in sidebar
- Clear all documents using the "Clear All Documents" button
- Check connection status in the main interface

## ⚙️ Configuration

### Environment Variables

```env
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo                    # or gpt-4 for better accuracy
EMBEDDING_MODEL=text-embedding-ada-002

# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=documents

# RAG Parameters
CHUNK_SIZE=1000                               # Text chunk size for processing
CHUNK_OVERLAP=200                             # Overlap between chunks
TOP_K_RETRIEVAL=5                             # Number of chunks to retrieve
CONFIDENCE_THRESHOLD=0.4                      # Minimum confidence for responses

# Document Processing
MAX_FILE_SIZE_MB=10                           # Maximum PDF file size
```

### 🎛️ Performance Tuning

- **Higher Accuracy**: Use `gpt-4` model, increase `CONFIDENCE_THRESHOLD` to 0.6+
- **Faster Processing**: Reduce `CHUNK_SIZE` to 512, lower `TOP_K_RETRIEVAL` to 3
- **Better Arabic Support**: Ensure OCR is properly configured with Arabic language packs
- **Memory Optimization**: Adjust Docker memory limits for large document collections
