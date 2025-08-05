from typing import List, Dict
import uuid
import hashlib
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from rank_bm25 import BM25Okapi
import pickle
import os
import re


class VectorStore:
    
    def __init__(self, config):
        self.config = config
        self.client = QdrantClient(
            url=config.qdrant_host,
            api_key=config.qdrant_api_key
        )
        self.collection_name = config.qdrant_collection
        
        self.embedding_model = OpenAIEmbedding(
            api_key=config.openai_api_key,
            model=config.embedding_model
        )
        self.embedding_dim = 1536
        
        # BM25 storage
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadata = []
        self.bm25_path = "bm25_index.pkl"
        
        # Track document hashes to prevent duplicates
        self.document_hashes = set()
        self.hashes_path = "document_hashes.pkl"
        
        # Initialize collection and load indices
        self._initialize_collection()
        self._load_bm25_index()
        self._load_document_hashes()
    
    def _initialize_collection(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"Error initializing collection: {e}")
    
    def _load_bm25_index(self):
        try:
            if os.path.exists(self.bm25_path):
                with open(self.bm25_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25_index = data['index']
                    self.bm25_documents = data['documents']
                    self.bm25_metadata = data['metadata']
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
    
    def _save_bm25_index(self):
        try:
            with open(self.bm25_path, 'wb') as f:
                pickle.dump({
                    'index': self.bm25_index,
                    'documents': self.bm25_documents,
                    'metadata': self.bm25_metadata
                }, f)
        except Exception as e:
            print(f"Error saving BM25 index: {e}")
    
    def _load_document_hashes(self):
        try:
            if os.path.exists(self.hashes_path):
                with open(self.hashes_path, 'rb') as f:
                    self.document_hashes = pickle.load(f)
        except Exception as e:
            print(f"Error loading document hashes: {e}")
    
    def _save_document_hashes(self):
        try:
            with open(self.hashes_path, 'wb') as f:
                pickle.dump(self.document_hashes, f)
        except Exception as e:
            print(f"Error saving document hashes: {e}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        # Check if text contains Arabic
        has_arabic = any('\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' for char in text)
        
        if has_arabic:
            tokens = re.findall(r'[\u0600-\u06FF\u0750-\u077F]+|[a-zA-Z]+|\d+', text.lower())
            return [token for token in tokens if len(token) > 1]  # Filter out single characters
        else:
            # For non-Arabic text, use simple split
            return text.lower().split()
    
    def _generate_chunk_hash(self, text: str, filename: str, page: int, chunk_index: int) -> str:
        content = f"{filename}_{page}_{chunk_index}_{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _hash_to_uuid(self, hash_string: str) -> str:
        # Use a namespace UUID for document chunks
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # Standard namespace
        return str(uuid.uuid5(namespace, hash_string))
    
    async def check_connection(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    async def add_documents(self, documents: List[Document]) -> bool:
        try:
            points = []
            new_documents_added = 0
            duplicates_skipped = 0
            
            for doc in documents:
                # Generate hash for this chunk
                chunk_hash = self._generate_chunk_hash(
                    doc.text,
                    doc.metadata.get("filename", ""),
                    doc.metadata.get("page", 0),
                    doc.metadata.get("chunk_index", 0)
                )
                
                # Skip if duplicate
                if chunk_hash in self.document_hashes:
                    duplicates_skipped += 1
                    continue
                
                # Generate embedding
                embedding = self.embedding_model.get_text_embedding(doc.text)
                
                # Convert hash to UUID for Qdrant
                point_id = self._hash_to_uuid(chunk_hash)
                
                # Create point with UUID as ID
                point = PointStruct(
                    id=point_id,  # Use UUID instead of raw hash
                    vector=embedding,
                    payload={
                        "text": doc.text,
                        "filename": doc.metadata.get("filename", ""),
                        "page": doc.metadata.get("page", 0),
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                        "hash": chunk_hash,  # Store original hash in payload
                        "language": doc.metadata.get("language", "unknown"),
                        "is_mixed": doc.metadata.get("is_mixed", False)
                    }
                )
                points.append(point)
                
                # Add to BM25 index
                tokenized_text = self._tokenize_text(doc.text)
                self.bm25_documents.append(tokenized_text)
                self.bm25_metadata.append({
                    "text": doc.text,
                    "filename": doc.metadata.get("filename", ""),
                    "page": doc.metadata.get("page", 0),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "hash": chunk_hash,
                    "language": doc.metadata.get("language", "unknown"),
                    "is_mixed": doc.metadata.get("is_mixed", False)
                })
                
                # Track hash
                self.document_hashes.add(chunk_hash)
                new_documents_added += 1
            
            if duplicates_skipped > 0:
                print(f"Skipped {duplicates_skipped} duplicate chunks")
            
            if points:
                # Upload points in batches
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                
                # Rebuild BM25 index
                if self.bm25_documents:
                    self.bm25_index = BM25Okapi(self.bm25_documents)
                    self._save_bm25_index()
                
                # Save document hashes
                self._save_document_hashes()
                
                print(f"Added {new_documents_added} new chunks")
            
            return True
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        try:
            # Semantic search
            semantic_results = await self._semantic_search(query, top_k * 2)
            
            # BM25 search
            bm25_results = await self._bm25_search(query, top_k * 2)
            
            # Combine and rank results
            combined_results = self._combine_search_results(
                semantic_results, bm25_results, top_k
            )
            
            return combined_results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    async def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform semantic similarity search"""
        try:
            # Generate query embedding based on model type
            query_embedding = self.embedding_model.get_query_embedding(query)
            
            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Format results
            documents = []
            for result in results:
                documents.append({
                    "text": result.payload["text"],
                    "filename": result.payload["filename"],
                    "page": result.payload["page"],
                    "score": result.score,
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "language": result.payload.get("language", "unknown"),
                    "is_mixed": result.payload.get("is_mixed", False),
                    "search_type": "semantic"
                })
            
            return documents
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    async def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform BM25 keyword search"""
        try:
            if not self.bm25_index or not self.bm25_documents:
                return []
            
            # Tokenize query
            query_tokens = self._tokenize_text(query)
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top results
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            documents = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include non-zero scores
                    documents.append({
                        "text": self.bm25_metadata[idx]["text"],
                        "filename": self.bm25_metadata[idx]["filename"],
                        "page": self.bm25_metadata[idx]["page"],
                        "score": scores[idx],
                        "chunk_index": self.bm25_metadata[idx]["chunk_index"],
                        "language": self.bm25_metadata[idx].get("language", "unknown"),
                        "is_mixed": self.bm25_metadata[idx].get("is_mixed", False),
                        "search_type": "bm25"
                    })
            
            return documents
            
        except Exception as e:
            print(f"Error in BM25 search: {e}")
            return []
    
    def _combine_search_results(self, semantic_results: List[Dict], 
                               bm25_results: List[Dict], top_k: int) -> List[Dict]:
        """Combine and rank semantic and BM25 search results"""
        try:
            # Normalize scores
            if semantic_results:
                max_semantic = max(doc["score"] for doc in semantic_results)
                for doc in semantic_results:
                    doc["normalized_score"] = doc["score"] / max_semantic if max_semantic > 0 else 0
            
            if bm25_results:
                max_bm25 = max(doc["score"] for doc in bm25_results)
                for doc in bm25_results:
                    doc["normalized_score"] = doc["score"] / max_bm25 if max_bm25 > 0 else 0
            
            # Combine results with weighted scoring (0.6 semantic + 0.4 BM25)
            all_results = {}
            
            # Add semantic results
            for doc in semantic_results:
                key = f"{doc['filename']}_{doc['page']}_{doc['chunk_index']}"
                doc["final_score"] = doc["normalized_score"] * 0.6
                all_results[key] = doc
            
            # Add or update with BM25 results
            for doc in bm25_results:
                key = f"{doc['filename']}_{doc['page']}_{doc['chunk_index']}"
                if key in all_results:
                    # Combine scores
                    all_results[key]["final_score"] += doc["normalized_score"] * 0.4
                    all_results[key]["search_type"] = "hybrid"
                else:
                    # New result from BM25
                    doc["final_score"] = doc["normalized_score"] * 0.4
                    all_results[key] = doc
            
            # Sort by final score and return top_k
            sorted_results = sorted(all_results.values(), 
                                  key=lambda x: x["final_score"], reverse=True)
            
            return sorted_results[:top_k]
            
        except Exception as e:
            print(f"Error combining search results: {e}")
            return semantic_results[:top_k]
    
    async def get_document_count(self) -> int:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception:
            return 0
    
    async def clear_collection(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            
            # Clear BM25 index
            self.bm25_index = None
            self.bm25_documents = []
            self.bm25_metadata = []
            if os.path.exists(self.bm25_path):
                os.remove(self.bm25_path)
            
            # Clear document hashes
            self.document_hashes.clear()
            if os.path.exists(self.hashes_path):
                os.remove(self.hashes_path)
            
            return True
        except Exception:
            return False