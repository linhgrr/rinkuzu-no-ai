import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.interfaces.vector_store import VectorStore, VectorDocument, SearchResult
from app.core.config import settings


class TrustingSentenceTransformerEmbeddingFunction(SentenceTransformerEmbeddingFunction):
    def __init__(self, model_name: str, device: str = "cpu"):
        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self._model.to(device)

    def __call__(self, texts):  # type: ignore[override]
        return self._model.encode(texts, convert_to_tensor=False).tolist()


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "default"
    ):
        self.persist_directory = persist_directory or settings.vector_db_path
        self.default_collection_name = collection_name

        os.makedirs(self.persist_directory, exist_ok=True)

        # Try using default embedding function first to avoid hanging issues
        try:
            logger.info(">>> Trying ChromaDB's default embedding function...")
            self.embedding_function = None  # Use ChromaDB default
            logger.info(">>> Using ChromaDB default embedding function")
        except Exception as e:
            logger.warning(f">>> Default embedding failed, trying custom: {e}")
            try:
                self.embedding_function = TrustingSentenceTransformerEmbeddingFunction(
                    model_name=settings.embedding_model
                )
                logger.info(f">>> Using custom embedding function with model: {settings.embedding_model}")
            except Exception as e2:
                logger.error(f">>> Custom embedding also failed: {e2}")
                self.embedding_function = None

        # Try to initialize ChromaDB client
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f">>> Attempting to initialize ChromaDB client (attempt {attempt + 1}/{max_retries})")
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                logger.info(">>> ChromaDB client initialized successfully")
                break
            except Exception as e:
                logger.error(f">>> ChromaDB initialization failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(">>> Max retries reached, trying to reset ChromaDB...")
                    self._reset_chroma_db()
                    # One final attempt after reset
                    self.client = chromadb.PersistentClient(
                        path=self.persist_directory,
                        settings=ChromaSettings(
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                    )
                    break
                else:
                    import time
                    time.sleep(1)  # Wait 1 second before retry

        self._collections = {}
        logger.info(f"Initialized ChromaVectorStore at {self.persist_directory}")

    def _reset_chroma_db(self):
        """Reset ChromaDB by removing the persistent directory"""
        try:
            import shutil
            if os.path.exists(self.persist_directory):
                logger.warning(f">>> Resetting ChromaDB directory: {self.persist_directory}")
                shutil.rmtree(self.persist_directory)
                os.makedirs(self.persist_directory, exist_ok=True)
                logger.info(">>> ChromaDB directory reset completed")
        except Exception as e:
            logger.error(f">>> Failed to reset ChromaDB directory: {e}")

    def _get_collection(self, collection_name: Optional[str] = None):
        collection_name = collection_name or self.default_collection_name

        if collection_name not in self._collections:
            # Create collection with or without custom embedding function
            if self.embedding_function:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
            else:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            self._collections[collection_name] = collection

        return self._collections[collection_name]

    async def add_documents(
        self,
        documents: List[VectorDocument],
        collection_name: Optional[str] = None
    ) -> bool:
        if not documents:
            return True

        try:
            logger.info(f">>> Starting add_documents with {len(documents)} documents")
            
            collection = self._get_collection(collection_name)
            logger.info(f">>> Got collection: {collection.name}")

            ids = []
            metadatas = []
            documents_text = []

            for i, doc in enumerate(documents):
                if not doc.content:
                    logger.warning(f"Skipping document {i} with empty content")
                    continue

                doc_id = doc.id or str(uuid.uuid4())
                ids.append(doc_id)
                metadatas.append(doc.metadata or {})
                documents_text.append(doc.content)
                logger.debug(f"Prepared document {i}: id={doc_id}, content_length={len(doc.content)}")

            if not documents_text:
                logger.warning("No valid documents to add.")
                return False
            
            logger.info(f">>> About to add {len(documents_text)} documents to ChromaDB")
            logger.info(f">>> Collection embedding function: {type(collection._embedding_function)}")
            
            # Try a simpler synchronous approach first
            try:
                logger.info(">>> Attempting synchronous add...")
                collection.add(
                    ids=ids,
                    metadatas=metadatas,
                    documents=documents_text
                )
                logger.info(">>> Synchronous add completed successfully!")
            except Exception as sync_error:
                logger.error(f">>> Synchronous add failed: {sync_error}")
                logger.info(">>> Attempting asynchronous add with timeout...")
                
                # Try async with timeout
                async def add_with_timeout():
                    return await asyncio.wait_for(
                        asyncio.to_thread(
                            collection.add,
                            ids=ids,
                            metadatas=metadatas,
                            documents=documents_text
                        ),
                        timeout=30.0  # 30 second timeout
                    )
                
                await add_with_timeout()
                logger.info(">>> Asynchronous add completed successfully!")
            
            logger.info(
                f"✅ Added {len(documents_text)} documents to collection '{collection_name or self.default_collection_name}'"
            )
            return True

        except asyncio.TimeoutError:
            logger.error("❌ Operation timed out after 30 seconds")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            logger.error(f"❌ Error type: {type(e)}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False

    async def search(
        self,
        query_text: str,
        collection_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        try:
            collection = self._get_collection(collection_name)

            where_clause = {}
            if filters:
                for key, value in filters.items():
                    where_clause[key] = {"$in": value} if isinstance(value, list) else {"$eq": value}

            results = await asyncio.to_thread(
                collection.query,
                query_texts=[query_text],
                n_results=top_k,
                where=where_clause if filters else None,
                include=["documents", "metadatas", "distances", "embeddings"]
            )

            search_results = []

            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            embeds = results.get("embeddings", [[]])[0] if results.get("embeddings") else []

            for i in range(len(ids)):
                vector_doc = VectorDocument(
                    id=ids[i],
                    content=docs[i],
                    embedding=embeds[i] if embeds else [],
                    metadata=metas[i] or {}
                )
                similarity_score = 1.0 - dists[i]
                search_results.append(SearchResult(document=vector_doc, score=similarity_score))

            logger.info(f"✅ Found {len(search_results)} results in collection '{collection_name or self.default_collection_name}'")
            return search_results

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        if not document_ids:
            return True

        try:
            collection = self._get_collection(collection_name)
            await asyncio.to_thread(collection.delete, ids=document_ids)

            logger.info(f"✅ Deleted {len(document_ids)} documents from collection '{collection_name or self.default_collection_name}'")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete documents: {e}")
            return False

    async def get_collection_stats(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            collection = self._get_collection(collection_name)

            count = await asyncio.to_thread(collection.count)

            sample_size = min(10, count)
            sample_results = await asyncio.to_thread(
                collection.get, limit=sample_size, include=["metadatas"]
            )

            metadata_keys = set()
            for metadata in sample_results.get("metadatas", []):
                if metadata:
                    metadata_keys.update(metadata.keys())

            stats = {
                "collection_name": collection_name or self.default_collection_name,
                "document_count": count,
                "metadata_keys": list(metadata_keys),
                "persist_directory": self.persist_directory
            }

            logger.info(f"Collection stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {"error": str(e)}

    def list_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"❌ Failed to list collections: {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.client.delete_collection(name=collection_name)
            self._collections.pop(collection_name, None)
            logger.info(f"✅ Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete collection {collection_name}: {e}")
            return False
 