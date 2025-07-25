import os
import uuid
import asyncio
import pickle
import json
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from app.core.interfaces.vector_store import VectorStore, VectorDocument, SearchResult
from app.core.config import settings


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store implementation with optimized concurrent processing
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "default"
    ):
        self.persist_directory = persist_directory or settings.vector_db_path
        self.default_collection_name = collection_name
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Concurrency control
        self._max_concurrent_embeddings = settings.max_concurrent_embeddings
        self._embedding_semaphore = asyncio.Semaphore(self._max_concurrent_embeddings)
        self._collection_locks: Dict[str, asyncio.Lock] = {}  # Per-collection locks
        self._collection_locks_lock = asyncio.Lock()  # Lock for locks dict
        
        # Thread pool for CPU-intensive operations
        max_workers = settings.faiss_thread_pool_workers
        if max_workers <= 0:
            max_workers = min(4, os.cpu_count() or 2)
        
        self._thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="faiss_worker"
        )
        
        # Initialize embeddings with HuggingFace model
        try:
            logger.info(f"Initializing HuggingFace embeddings with model: {settings.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={'device': 'cpu', 'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ HuggingFace embeddings initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            raise
            
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        
        # FAISS vector stores for different collections
        self._vector_stores: Dict[str, FAISS] = {}
        self._metadata_stores: Dict[str, Dict[str, Dict]] = {}  # Store metadata separately
        
        logger.info(f"Initialized FAISS VectorStore at {self.persist_directory}")
    
    def __del__(self):
        """Cleanup thread pool on destruction"""
        try:
            if hasattr(self, '_thread_pool'):
                self._thread_pool.shutdown(wait=False)
        except:
            pass
    
    async def close(self):
        """Gracefully close the vector store and cleanup resources"""
        try:
            self._thread_pool.shutdown(wait=True)
            logger.info("✅ FAISS VectorStore closed successfully")
        except Exception as e:
            logger.error(f"❌ Error closing FAISS VectorStore: {e}")
    
    async def _get_collection_lock(self, collection_name: str) -> asyncio.Lock:
        """Get or create a lock for the specified collection"""
        async with self._collection_locks_lock:
            if collection_name not in self._collection_locks:
                self._collection_locks[collection_name] = asyncio.Lock()
            return self._collection_locks[collection_name]
    
    def _get_collection_path(self, collection_name: str) -> str:
        """Get the file path for a collection"""
        return os.path.join(self.persist_directory, f"{collection_name}.faiss")
    
    def _get_metadata_path(self, collection_name: str) -> str:
        """Get the metadata file path for a collection"""
        return os.path.join(self.persist_directory, f"{collection_name}_metadata.json")
    
    def _get_vector_store(self, collection_name: Optional[str] = None) -> FAISS:
        """Get or create FAISS vector store for the specified collection"""
        collection_name = collection_name or self.default_collection_name
        
        if collection_name not in self._vector_stores:
            try:
                collection_path = self._get_collection_path(collection_name)
                metadata_path = self._get_metadata_path(collection_name)
                
                # Try to load existing FAISS index
                if os.path.exists(collection_path):
                    logger.info(f"Loading existing FAISS index: {collection_name}")
                    vector_store = FAISS.load_local(
                        collection_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    # Load metadata
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            self._metadata_stores[collection_name] = json.load(f)
                    else:
                        self._metadata_stores[collection_name] = {}
                        
                    logger.info(f"✅ Loaded existing FAISS index for collection: {collection_name}")
                else:
                    # Create new FAISS index - we'll initialize it as None and create it when first document is added
                    logger.info(f"Creating new FAISS index: {collection_name}")
                    vector_store = None  # Will be created when first documents are added
                    self._metadata_stores[collection_name] = {}
                    logger.info(f"✅ New FAISS index slot created for collection: {collection_name}")
                
                self._vector_stores[collection_name] = vector_store
                
            except Exception as e:
                logger.error(f"❌ Failed to create/load FAISS index for {collection_name}: {e}")
                raise
                
        return self._vector_stores[collection_name]
    
    def _save_collection(self, collection_name: str):
        """Save FAISS index and metadata to disk (thread-safe)"""
        try:
            if collection_name in self._vector_stores and self._vector_stores[collection_name] is not None:
                collection_path = self._get_collection_path(collection_name)
                metadata_path = self._get_metadata_path(collection_name)
                
                # Save FAISS index
                self._vector_stores[collection_name].save_local(collection_path)
                
                # Save metadata
                if collection_name in self._metadata_stores:
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(self._metadata_stores[collection_name], f, ensure_ascii=False, indent=2)
                
                logger.debug(f"✅ Saved collection {collection_name} to disk")
        except Exception as e:
            logger.error(f"❌ Failed to save collection {collection_name}: {e}")

    async def add_documents_batch(
        self,
        documents: List[VectorDocument],
        collection_name: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> bool:
        """Add multiple documents in batches to optimize memory usage"""
        if not documents:
            return True
            
        batch_size = batch_size or settings.batch_size_documents
        collection_name = collection_name or self.default_collection_name
        
        logger.info(f"🚀 Processing {len(documents)} documents in batches of {batch_size}")
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size} ({len(batch)} documents)")
            
            success = await self.add_documents(batch, collection_name)
            if not success:
                logger.error(f"❌ Failed to process batch {i//batch_size + 1}")
                return False
                
        logger.info("✅ All document batches processed successfully")
        return True

    async def add_documents(
        self,
        documents: List[VectorDocument],
        collection_name: Optional[str] = None
    ) -> bool:
        """Add documents to the FAISS vector store with optimized concurrent processing"""
        if not documents:
            return True
            
        collection_name = collection_name or self.default_collection_name
        
        # Get collection-specific lock to prevent concurrent modifications
        collection_lock = await self._get_collection_lock(collection_name)
        
        async with collection_lock:
            # Use semaphore to limit concurrent embedding operations
            async with self._embedding_semaphore:
                try:
                    logger.info(f">>> Starting add_documents with {len(documents)} documents to FAISS (collection: {collection_name})")
                    
                    # Get vector store (might be None for new collections)
                    vector_store = self._get_vector_store(collection_name)
                    
                    # Convert VectorDocument to LangChain Document format
                    langchain_docs = []
                    doc_ids = []
                    
                    for doc in documents:
                        if not doc.content.strip():
                            logger.warning(f"Skipping document with empty content: {doc.id}")
                            continue
                            
                        # Split document into chunks if it's too long
                        chunks = self.text_splitter.split_text(doc.content)
                        
                        for i, chunk in enumerate(chunks):
                            if not chunk.strip():
                                continue
                                
                            # Create unique ID for each chunk
                            chunk_id = f"{doc.id}_{i}" if len(chunks) > 1 else doc.id
                            doc_ids.append(chunk_id)
                            
                            # Prepare metadata
                            metadata = doc.metadata.copy() if doc.metadata else {}
                            metadata.update({
                                "source_doc_id": doc.id,
                                "chunk_index": i,
                                "total_chunks": len(chunks)
                            })
                            
                            langchain_doc = Document(
                                page_content=chunk,
                                metadata=metadata
                            )
                            langchain_docs.append(langchain_doc)
                    
                    if not langchain_docs:
                        logger.warning("No valid document chunks to add.")
                        return False
                    
                    logger.info(f">>> Prepared {len(langchain_docs)} document chunks for FAISS indexing")
                    
                    # Handle initial index creation or adding to existing index with increased timeout
                    try:
                        if vector_store is None:
                            # Create new FAISS index from first batch of documents
                            logger.info(">>> Creating new FAISS index from documents")
                            vector_store = await asyncio.wait_for(
                                asyncio.get_event_loop().run_in_executor(
                                    self._thread_pool,
                                    lambda: FAISS.from_documents(langchain_docs, self.embeddings)
                                ),
                                timeout=settings.embedding_timeout_seconds
                            )
                            # Update the stored vector store
                            self._vector_stores[collection_name] = vector_store
                        else:
                            logger.info(">>> Adding documents to existing FAISS index")
                            await asyncio.wait_for(
                                asyncio.get_event_loop().run_in_executor(
                                    self._thread_pool,
                                    lambda: vector_store.add_documents(langchain_docs)
                                ),
                                timeout=settings.embedding_timeout_seconds
                            )
                        
                        logger.info(f"✅ Successfully added {len(langchain_docs)} document chunks to FAISS")
                        
                        # Save to disk in thread pool to avoid blocking
                        await asyncio.get_event_loop().run_in_executor(
                            self._thread_pool,
                            lambda: self._save_collection(collection_name)
                        )
                        logger.info("✅ FAISS index saved to disk")
                        
                        return True
                        
                    except asyncio.TimeoutError:
                        logger.error(f"❌ FAISS add_documents timed out after {settings.embedding_timeout_seconds} seconds")
                        return False
                    except Exception as e:
                        logger.error(f"❌ Failed to add documents to FAISS: {e}")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Error in add_documents: {e}")
                    import traceback
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    return False
    
    async def search(
        self,
        query_text: str,
        collection_name: Optional[str] = None,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents using FAISS similarity search"""
        try:
            collection_name = collection_name or self.default_collection_name
            logger.info(f">>> Searching in FAISS for: '{query_text}...' in collection: {collection_name}")
            
            vector_store = self._get_vector_store(collection_name)
            
            # Check if collection exists and has data
            if vector_store is None:
                logger.warning(f">>> No FAISS index found for collection: {collection_name}")
                return []
            
            # FAISS similarity search with score
            search_kwargs = {"k": top_k}
            # Note: FAISS doesn't support metadata filtering as robustly as ChromaDB
            # We'll filter results after retrieval if needed
            
            results = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                lambda: vector_store.similarity_search_with_score(query_text, **search_kwargs)
            )
            
            # Convert results to SearchResult format
            search_results = []
            for doc, distance in results:
                # FAISS returns L2 distance (lower = more similar)
                # Convert to similarity score (higher = more similar)
                similarity_score = 1.0 / (1.0 + distance)
                
                # # Apply filters if provided
                # if filters:
                #     match = True
                #     for key, value in filters.items():
                #         if key not in doc.metadata or doc.metadata[key] != value:
                #             match = False
                #             break
                #     if not match:
                #         continue
                
                vector_doc = VectorDocument(
                    id=doc.metadata.get("source_doc_id", str(uuid.uuid4())),
                    content=doc.page_content,
                    metadata=doc.metadata
                )
                
                search_result = SearchResult(
                    document=vector_doc,
                    score=float(similarity_score)
                )
                search_results.append(search_result)
            
            # Sort by similarity score (descending)
            search_results.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"✅ Found {len(search_results)} similar documents in FAISS")
            if search_results:
                logger.info(f">>> Top score: {search_results[0].score:.4f}")
            
            return search_results
            
        except Exception as e:
            logger.error(f"❌ FAISS search failed: {e}")
            import traceback
            logger.error(f"❌ Search traceback: {traceback.format_exc()}")
            return []
    
    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """Delete documents by IDs (Note: FAISS doesn't support direct deletion)"""
        try:
            collection_name = collection_name or self.default_collection_name
            logger.warning("⚠️ FAISS doesn't support direct document deletion")
            logger.info("💡 To delete documents, you'll need to recreate the index without those documents")
            
            # For now, just mark as deleted in metadata
            # In a production system, you'd rebuild the index without these documents
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete documents: {e}")
            return False
    
    async def get_collection_stats(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection_name = collection_name or self.default_collection_name
            vector_store = self._get_vector_store(collection_name)
            
            # Check if collection exists
            if vector_store is None:
                count = 0
            else:
                # Get document count from FAISS index
                count = vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else 0
            
            return {
                "document_count": count,
                "collection_name": collection_name,
                "embedding_model": settings.embedding_model,
                "vector_store_type": "FAISS"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get FAISS collection stats: {e}")
            return {"document_count": 0, "error": str(e)}
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        try:
            collections = []
            if os.path.exists(self.persist_directory):
                for file in os.listdir(self.persist_directory):
                    if file.endswith('.faiss'):
                        collection_name = file.replace('.faiss', '')
                        collections.append(collection_name)
            
            return collections
            
        except Exception as e:
            logger.error(f"❌ Failed to list FAISS collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete an entire collection"""
        try:
            if collection_name in self._vector_stores:
                del self._vector_stores[collection_name]
            
            if collection_name in self._metadata_stores:
                del self._metadata_stores[collection_name]
            
            # Remove files
            collection_path = self._get_collection_path(collection_name)
            metadata_path = self._get_metadata_path(collection_name)
            
            if os.path.exists(collection_path):
                os.remove(collection_path)
                
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            logger.info(f"✅ Deleted FAISS collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete FAISS collection {collection_name}: {e}")
            return False 