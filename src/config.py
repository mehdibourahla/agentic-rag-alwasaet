import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    
    def __init__(self):
        # OpenAI settings
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        
        # Qdrant settings
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "documents")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        
        # RAG settings
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.top_k_retrieval = int(os.getenv("TOP_K_RETRIEVAL", "5"))
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
        
        # Document processing settings
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
        
        # Validate required settings
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    def get_qdrant_url(self):
        """Get Qdrant connection URL"""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"