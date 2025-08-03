# src/services/vector_store.py
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .pdf_service import PDFService
from ..config import Config
import uuid
import numpy as np

class VectorStore:
    """Qdrant Cloud vector database service."""
    
    def __init__(self):
        # Initialize Qdrant client with cloud credentials
        if Config.QDRANT_URL and Config.QDRANT_API_KEY:
            self.client = QdrantClient(
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY,
            )
            self.use_cloud = True
        else:
            # Fallback to in-memory storage
            print("Warning: Using in-memory vector storage. Data will not persist.")
            self.client = None
            self.use_cloud = False
            self._memory_store = []
        
        self.collection_name = Config.COLLECTION_NAME
        self.pdf_service = PDFService()
        
        if self.use_cloud:
            self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self.use_cloud:
            return
            
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Get embedding dimension from sentence-transformers model
                embedding_dim = 384  # all-MiniLM-L6-v2 dimension
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            print(f"Error ensuring collection: {e}")
            # Fallback to memory storage
            self.use_cloud = False
            self._memory_store = []
    
    def store_embeddings(self, chunks: List[str], embeddings: List[List[float]], 
                        metadata: Dict[str, Any] = None) -> bool:
        """
        Store text chunks and embeddings in vector database.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        if self.use_cloud:
            return self._store_cloud(chunks, embeddings, metadata)
        else:
            return self._store_memory(chunks, embeddings, metadata)
    
    def _store_cloud(self, chunks: List[str], embeddings: List[List[float]], 
                    metadata: Dict[str, Any] = None) -> bool:
        """Store in Qdrant Cloud."""
        try:
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "chunk_index": i,
                        **(metadata or {})
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Stored {len(points)} embeddings in Qdrant Cloud")
            return True
            
        except Exception as e:
            print(f"Error storing embeddings in cloud: {e}")
            # Fallback to memory
            self.use_cloud = False
            return self._store_memory(chunks, embeddings, metadata)
    
    def _store_memory(self, chunks: List[str], embeddings: List[List[float]], 
                     metadata: Dict[str, Any] = None) -> bool:
        """Store in memory as fallback."""
        try:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                self._memory_store.append({
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "text": chunk,
                    "chunk_index": i,
                    "metadata": metadata or {}
                })
            
            print(f"Stored {len(chunks)} embeddings in memory")
            return True
            
        except Exception as e:
            print(f"Error storing embeddings in memory: {e}")
            return False
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of similar documents
        """
        if self.use_cloud:
            return self._search_cloud(query, limit)
        else:
            return self._search_memory(query, limit)
    
    def _search_cloud(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search in Qdrant Cloud."""
        try:
            # Create query embedding
            query_embedding = self.pdf_service.create_query_embedding(query)
            if not query_embedding:
                return []
            
            # Search in vector database
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "text": result.payload.get("text", ""),
                    "score": result.score,
                    "metadata": {
                        k: v for k, v in result.payload.items() 
                        if k != "text"
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching in cloud: {e}")
            # Fallback to memory search
            self.use_cloud = False
            return self._search_memory(query, limit)
    
    def _search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search in memory storage."""
        try:
            if not self._memory_store:
                return []
            
            # Create query embedding
            query_embedding = self.pdf_service.create_query_embedding(query)
            if not query_embedding:
                return []
            
            # Calculate similarities
            similarities = []
            for item in self._memory_store:
                # Simple cosine similarity
                similarity = self._cosine_similarity(query_embedding, item["vector"])
                similarities.append((similarity, item))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for score, item in similarities[:limit]:
                results.append({
                    "text": item["text"],
                    "score": score,
                    "metadata": item["metadata"]
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching in memory: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if self.use_cloud:
            try:
                info = self.client.get_collection(self.collection_name)
                return {
                    "name": self.collection_name,  # Use the collection name we know
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value,  # Add .value for enum
                    "points_count": info.points_count,
                    "storage": "Qdrant Cloud"
                }
            except Exception as e:
                print(f"Error getting collection info: {e}")
                return {
                    "name": self.collection_name,
                    "storage": "Qdrant Cloud (error)",
                    "error": str(e)
                }
        else:
            return {
                "name": self.collection_name,
                "storage": "In-Memory",
                "points_count": len(self._memory_store),
                "vector_size": 384
            }