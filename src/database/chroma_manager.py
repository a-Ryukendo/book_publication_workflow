import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from loguru import logger
from models import ScrapedContent, ProcessedContent, SearchQuery, VersionControl
from config.settings import settings

class ChromaManager:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._init_chroma()

    def _init_chroma(self):
        try:
            self.client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.client.get_or_create_collection(
                name=settings.chroma_collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Book publication workflow content"}
            )
            logger.info("ChromaDB ready")
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            raise

    def add_scraped_content(self, content: ScrapedContent) -> str:
        try:
            content_id = str(content.id)
            metadata = {
                "type": "scraped",
                "url": content.url,
                "title": content.title,
                "word_count": len(content.content.split()),
                "quality_score": content.metadata.get("quality_score", 0.0),
                "scraped_at": content.scraped_at.isoformat(),
                "screenshot_path": content.screenshot_path or "",
                "status": content.status.value,
                "content_hash": self._hash(content.content)
            }
            self.collection.add(
                documents=[content.content],
                metadatas=[metadata],
                ids=[content_id]
            )
            logger.info(f"Added scraped: {content_id}")
            return content_id
        except Exception as e:
            logger.error(f"Add scraped failed: {e}")
            raise

    def add_processed_content(self, content: ProcessedContent, original_content: ScrapedContent) -> str:
        try:
            content_id = str(content.id)
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
                "content_hash": self._hash(content.editor_output)
            }
            combined = f"{content.writer_output}\n\n{content.editor_output}"
            self.collection.add(
                documents=[combined],
                metadatas=[metadata],
                ids=[content_id]
            )
            logger.info(f"Added processed: {content_id}")
            return content_id
        except Exception as e:
            logger.error(f"Add processed failed: {e}")
            raise

    def add_version_control(self, version: VersionControl) -> str:
        try:
            version_id = f"{str(version.content_id)}_v{version.version_number}"
            metadata = {
                "type": "version",
                "content_id": str(version.content_id),
                "version_number": version.version_number,
                "change_description": version.change_description,
                "changed_by": version.changed_by,
                "changed_at": version.changed_at.isoformat(),
                "parent_version": str(version.parent_version) if version.parent_version else "",
                "content_hash": self._hash(version.content_snapshot)
            }
            self.collection.add(
                documents=[version.content_snapshot],
                metadatas=[metadata],
                ids=[version_id]
            )
            logger.info(f"Added version: {version_id}")
            return version_id
        except Exception as e:
            logger.error(f"Add version failed: {e}")
            raise

    def semantic_search(self, query: str, filters: Dict[str, Any] = None, limit: int = 10, include_metadata: bool = True) -> List[Dict[str, Any]]:
        try:
            where = {}
            if filters:
                for k, v in filters.items():
                    if isinstance(v, (list, tuple)):
                        where[k] = {"$in": v}
                    else:
                        where[k] = v
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where if where else None,
                include=["metadatas", "documents", "distances"]
            )
            out = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    item = {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "distance": results["distances"][0][i],
                        "similarity_score": 1 - results["distances"][0][i]
                    }
                    if include_metadata and results["metadatas"]:
                        item["metadata"] = results["metadatas"][0][i]
                    out.append(item)
            logger.info(f"Semantic search: {len(out)} results")
            return out
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise

    def get_version_history(self, content_id: str) -> List[Dict[str, Any]]:
        # Example: implement as needed
        return []

    def get_collection_stats(self) -> Dict[str, Any]:
        # Example: implement as needed
        return {}

    def _hash(self, content: str) -> str:
        return hashlib.md5(content.encode('utf-8')).hexdigest()

chroma_manager = ChromaManager() 