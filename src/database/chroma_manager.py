"""
ChromaDB integration for semantic search and content versioning
"""
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from loguru import logger

from ..models import ScrapedContent, ProcessedContent, SearchQuery, VersionControl
from ..config.settings import settings


class ChromaManager:
    """ChromaDB manager for semantic search and content versioning"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.initialize_chroma()
    
    def initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Book publication workflow content"}
            )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_scraped_content(self, content: ScrapedContent) -> str:
        """Add scraped content to ChromaDB"""
        try:
            # Create unique ID
            content_id = str(content.id)
            
            # Prepare metadata
            metadata = {
                "type": "scraped",
                "url": content.url,
                "title": content.title,
                "word_count": len(content.content.split()),
                "quality_score": content.metadata.get("quality_score", 0.0),
                "scraped_at": content.scraped_at.isoformat(),
                "screenshot_path": content.screenshot_path or "",
                "status": content.status.value,
                "content_hash": self._calculate_content_hash(content.content)
            }
            
            # Add to collection
            self.collection.add(
                documents=[content.content],
                metadatas=[metadata],
                ids=[content_id]
            )
            
            logger.info(f"Added scraped content to ChromaDB: {content_id}")
            return content_id
            
        except Exception as e:
            logger.error(f"Failed to add scraped content: {e}")
            raise
    
    def add_processed_content(self, content: ProcessedContent, 
                            original_content: ScrapedContent) -> str:
        """Add processed content to ChromaDB"""
        try:
            # Create unique ID
            content_id = str(content.id)
            
            # Prepare metadata
            metadata = {
                "type": "processed",
                "original_content_id": str(content.original_content_id),
                "original_url": original_content.url,
                "original_title": original_content.title,
                "writer_output_length": len(content.writer_output),
                "editor_output_length": len(content.editor_output),
                "quality_score": content.quality_score,
                "processed_at": content.processed_at.isoformat(),
                "status": content.status.value,
                "content_hash": self._calculate_content_hash(content.editor_output)
            }
            
            # Combine all content for embedding
            combined_content = f"{content.writer_output}\n\n{content.editor_output}"
            
            # Add to collection
            self.collection.add(
                documents=[combined_content],
                metadatas=[metadata],
                ids=[content_id]
            )
            
            logger.info(f"Added processed content to ChromaDB: {content_id}")
            return content_id
            
        except Exception as e:
            logger.error(f"Failed to add processed content: {e}")
            raise
    
    def add_version_control(self, version: VersionControl) -> str:
        """Add version control entry to ChromaDB"""
        try:
            # Create unique ID
            version_id = f"{str(version.content_id)}_v{version.version_number}"
            
            # Prepare metadata
            metadata = {
                "type": "version",
                "content_id": str(version.content_id),
                "version_number": version.version_number,
                "change_description": version.change_description,
                "changed_by": version.changed_by,
                "changed_at": version.changed_at.isoformat(),
                "parent_version": str(version.parent_version) if version.parent_version else "",
                "content_hash": self._calculate_content_hash(version.content_snapshot)
            }
            
            # Add to collection
            self.collection.add(
                documents=[version.content_snapshot],
                metadatas=[metadata],
                ids=[version_id]
            )
            
            logger.info(f"Added version control to ChromaDB: {version_id}")
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to add version control: {e}")
            raise
    
    def semantic_search(self, query: str, filters: Dict[str, Any] = None, 
                       limit: int = 10, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """Perform semantic search on content"""
        try:
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause if where_clause else None,
                include=["metadatas", "documents", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    result = {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "distance": results["distances"][0][i],
                        "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                    }
                    
                    if include_metadata and results["metadatas"]:
                        result["metadata"] = results["metadatas"][0][i]
                    
                    formatted_results.append(result)
            
            logger.info(f"Semantic search completed: {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    def search_by_content_type(self, content_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search content by type (scraped, processed, version)"""
        return self.semantic_search(
            query="",  # Empty query to get all documents
            filters={"type": content_type},
            limit=limit
        )
    
    def search_by_quality_score(self, min_score: float = 0.0, max_score: float = 1.0, 
                               limit: int = 10) -> List[Dict[str, Any]]:
        """Search content by quality score range"""
        return self.semantic_search(
            query="",
            filters={
                "quality_score": {
                    "$gte": min_score,
                    "$lte": max_score
                }
            },
            limit=limit
        )
    
    def search_by_date_range(self, start_date: datetime, end_date: datetime, 
                           date_field: str = "scraped_at", limit: int = 10) -> List[Dict[str, Any]]:
        """Search content by date range"""
        return self.semantic_search(
            query="",
            filters={
                date_field: {
                    "$gte": start_date.isoformat(),
                    "$lte": end_date.isoformat()
                }
            },
            limit=limit
        )
    
    def get_content_by_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get content by ID"""
        try:
            results = self.collection.get(
                ids=[content_id],
                include=["metadatas", "documents"]
            )
            
            if results["ids"] and results["ids"][0]:
                return {
                    "id": results["ids"][0][0],
                    "content": results["documents"][0][0],
                    "metadata": results["metadatas"][0][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get content by ID: {e}")
            return None
    
    def get_version_history(self, content_id: str) -> List[Dict[str, Any]]:
        """Get version history for a content item"""
        try:
            results = self.semantic_search(
                query="",
                filters={"type": "version", "content_id": content_id},
                limit=100  # Get all versions
            )
            
            # Sort by version number
            results.sort(key=lambda x: x["metadata"]["version_number"])
            return results
            
        except Exception as e:
            logger.error(f"Failed to get version history: {e}")
            return []
    
    def update_content(self, content_id: str, new_content: str, 
                      metadata_updates: Dict[str, Any] = None) -> bool:
        """Update existing content"""
        try:
            # Get existing content
            existing = self.get_content_by_id(content_id)
            if not existing:
                logger.warning(f"Content not found for update: {content_id}")
                return False
            
            # Prepare updated metadata
            updated_metadata = existing["metadata"].copy()
            if metadata_updates:
                updated_metadata.update(metadata_updates)
            
            # Update content hash
            updated_metadata["content_hash"] = self._calculate_content_hash(new_content)
            updated_metadata["updated_at"] = datetime.utcnow().isoformat()
            
            # Update in collection
            self.collection.update(
                ids=[content_id],
                documents=[new_content],
                metadatas=[updated_metadata]
            )
            
            logger.info(f"Updated content in ChromaDB: {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update content: {e}")
            return False
    
    def delete_content(self, content_id: str) -> bool:
        """Delete content from ChromaDB"""
        try:
            self.collection.delete(ids=[content_id])
            logger.info(f"Deleted content from ChromaDB: {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete content: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            
            # Get sample of documents for analysis
            sample_results = self.collection.get(
                limit=min(count, 100),
                include=["metadatas"]
            )
            
            # Analyze content types
            content_types = {}
            quality_scores = []
            
            if sample_results["metadatas"]:
                for metadata in sample_results["metadatas"]:
                    content_type = metadata.get("type", "unknown")
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                    
                    if "quality_score" in metadata:
                        quality_scores.append(metadata["quality_score"])
            
            return {
                "total_documents": count,
                "content_types": content_types,
                "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                "min_quality_score": min(quality_scores) if quality_scores else 0.0,
                "max_quality_score": max(quality_scores) if quality_scores else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash for content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def backup_collection(self, backup_path: str = None) -> str:
        """Backup the collection"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backups/chroma_backup_{timestamp}"
        
        try:
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Export collection data
            all_data = self.collection.get(include=["metadatas", "documents"])
            
            backup_data = {
                "collection_name": settings.chroma_collection_name,
                "backup_timestamp": datetime.utcnow().isoformat(),
                "total_documents": len(all_data["ids"]),
                "documents": []
            }
            
            for i, doc_id in enumerate(all_data["ids"]):
                backup_data["documents"].append({
                    "id": doc_id,
                    "content": all_data["documents"][i],
                    "metadata": all_data["metadatas"][i]
                })
            
            # Save backup
            with open(f"{backup_path}.json", "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Collection backed up to: {backup_path}.json")
            return f"{backup_path}.json"
            
        except Exception as e:
            logger.error(f"Failed to backup collection: {e}")
            raise
    
    def restore_collection(self, backup_path: str) -> bool:
        """Restore collection from backup"""
        try:
            with open(backup_path, "r", encoding="utf-8") as f:
                backup_data = json.load(f)
            
            # Clear existing collection
            self.collection.delete(where={})
            
            # Restore documents
            documents = []
            metadatas = []
            ids = []
            
            for doc in backup_data["documents"]:
                documents.append(doc["content"])
                metadatas.append(doc["metadata"])
                ids.append(doc["id"])
            
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"Collection restored from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore collection: {e}")
            return False


# Global ChromaDB manager instance
chroma_manager = ChromaManager() 