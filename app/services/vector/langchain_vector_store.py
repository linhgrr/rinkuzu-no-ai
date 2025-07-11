import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from app.core.interfaces.vector_store import VectorStore, VectorDocument, SearchResult
from app.core.config import settings


class LangChainVectorStore(VectorStore):
    """
    LangChain-based vector store implementation with better error handling
    and abstractions than direct ChromaDB usage
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "default"
    ):
        self.persist_directory = persist_directory or settings.vector_db_path
        self.default_collection_name = collection_name
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
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
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Vector stores for different collections
        self._vector_stores: Dict[str, Chroma] = {}
        
        logger.info(f"Initialized LangChainVectorStore at {self.persist_directory}")
    
    def _get_vector_store(self, collection_name: Optional[str] = None) -> Chroma:
        """Get or create vector store for the specified collection"""
        collection_name = collection_name or self.default_collection_name
        
        if collection_name not in self._vector_stores:
            try:
                persist_path = os.path.join(self.persist_directory, collection_name)
                
                logger.info(f"Creating/loading vector store for collection: {collection_name}")
                vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_path,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                
                self._vector_stores[collection_name] = vector_store
                logger.info(f"✅ Vector store ready for collection: {collection_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create vector store for {collection_name}: {e}")
                raise
                
        return self._vector_stores[collection_name]
    
    async def add_documents(
        self,
        documents: List[VectorDocument],
        collection_name: Optional[str] = None
    ) -> bool:
        """Add documents to the vector store with LangChain's robust handling"""
        if not documents:
            return True
            
        try:
            logger.info(f">>> Starting add_documents with {len(documents)} documents")
            
            vector_store = self._get_vector_store(collection_name)
            
            # Convert VectorDocument to LangChain Document format
            langchain_docs = []
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
            
            logger.info(f">>> Prepared {len(langchain_docs)} document chunks for indexing")
            
            # Use LangChain's async-friendly add_documents method
            try:
                await asyncio.to_thread(
                    vector_store.add_documents,
                    langchain_docs
                )
                logger.info(f"✅ Successfully added {len(langchain_docs)} document chunks")
                
                # Persist the changes
                await asyncio.to_thread(vector_store.persist)
                logger.info("✅ Vector store persisted to disk")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to add documents to vector store: {e}")
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
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents using LangChain's similarity search"""
        try:
            logger.info(f">>> Searching for: '{query_text[:50]}...' in collection: {collection_name}")
            
            vector_store = self._get_vector_store(collection_name)
            
            # Use LangChain's similarity search with score
            search_kwargs = {"k": top_k}
            if filters:
                search_kwargs["filter"] = filters
            
            results = await asyncio.to_thread(
                vector_store.similarity_search_with_score,
                query_text,
                **search_kwargs
            )
            
            # Convert results to SearchResult format
            search_results = []
            for doc, distance in results:
                # ChromaDB returns distance (lower = more similar)
                # Convert to similarity score (higher = more similar)
                similarity_score = 1.0 / (1.0 + distance)
                
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
            
            logger.info(f"✅ Found {len(search_results)} similar documents")
            if search_results:
                logger.info(f">>> Top score: {search_results[0].score:.4f}")
            
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            import traceback
            logger.error(f"❌ Search traceback: {traceback.format_exc()}")
            return []
    
    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """Delete documents by IDs"""
        try:
            vector_store = self._get_vector_store(collection_name)
            
            # LangChain Chroma supports deletion by metadata filter
            for doc_id in document_ids:
                await asyncio.to_thread(
                    vector_store.delete,
                    filter={"source_doc_id": doc_id}
                )
            
            logger.info(f"✅ Deleted {len(document_ids)} documents")
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
            vector_store = self._get_vector_store(collection_name)
            
            # Get collection info
            collection = vector_store._collection
            count = await asyncio.to_thread(collection.count)
            
            return {
                "document_count": count,
                "collection_name": collection_name or self.default_collection_name,
                "embedding_model": settings.embedding_model
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {"document_count": 0, "error": str(e)}
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        try:
            # List subdirectories in persist_directory
            collections = []
            if os.path.exists(self.persist_directory):
                for item in os.listdir(self.persist_directory):
                    item_path = os.path.join(self.persist_directory, item)
                    if os.path.isdir(item_path):
                        collections.append(item)
            
            return collections
            
        except Exception as e:
            logger.error(f"❌ Failed to list collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete an entire collection"""
        try:
            if collection_name in self._vector_stores:
                del self._vector_stores[collection_name]
            
            # Remove the persist directory for this collection
            collection_path = os.path.join(self.persist_directory, collection_name)
            if os.path.exists(collection_path):
                import shutil
                shutil.rmtree(collection_path)
                logger.info(f"✅ Deleted collection: {collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete collection {collection_name}: {e}")
            return False 