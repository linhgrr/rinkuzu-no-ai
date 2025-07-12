"""
DocumentIndexer – single-responsibility helper that handles
loading, chunking and indexing learning materials into the vector store.
This component allows RagTutorService (and any future service) to delegate
all document-ingestion concerns, following SOLID principles.
"""
from __future__ import annotations

import io
import uuid
from typing import Dict, Any, Optional, List

from loguru import logger

from app.core.interfaces.vector_store import VectorStore, VectorDocument
from app.services.file_loaders.factory import FileLoaderFactory
from app.utils.collection_utils import get_subject_collection_name


class DocumentIndexer:
    """Extract, chunk and store documents for RAG pipelines."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        file_loader_factory: FileLoaderFactory,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        collection_prefix: str = "subject",
    ) -> None:
        self.vector_store = vector_store
        self.file_loader_factory = file_loader_factory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_prefix = collection_prefix

    async def index_document(
        self,
        *,
        file_content: bytes,
        filename: str,
        subject_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a file and store its chunks inside the vector store.

        Args:
            file_content: Raw bytes of the file.
            filename: Original filename.
            subject_id: Subject/course identifier (used for collection).
            metadata: Optional additional metadata.

        Returns:
            Dict with success flag, message/error and statistics.
        """
        try:
            logger.info(f"📚 Processing document: {filename} for subject: {subject_id}")

            # 1. Load & split 
            loader = self.file_loader_factory.get_loader(
                filename,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            file_stream = io.BytesIO(file_content)
            chunks = await loader.load(file_stream, filename)

            if not chunks:
                return {
                    "success": False,
                    "error": "No content could be extracted from the document",
                }

            logger.info(f"📄 Extracted {len(chunks)} chunks from {filename}")

            vector_documents: List[VectorDocument] = []
            for i, chunk in enumerate(chunks):
                combined_metadata: Dict[str, Any] = {
                    "subject_id": subject_id,
                    "filename": filename,
                    "chunk_index": i,
                    # short uuid acts as timestamp / unique suffix
                    "upload_time": str(uuid.uuid4())[:8],
                }
                if metadata:
                    combined_metadata.update(metadata)
                # Preserve metadata extracted by loader (page, etc.)
                combined_metadata.update(chunk.metadata)

                vector_documents.append(
                    VectorDocument(
                        id=f"{subject_id}_{filename}_{i}_{combined_metadata['upload_time']}",
                        content=chunk.content,
                        metadata=combined_metadata,
                    )
                )

            collection_name = get_subject_collection_name(subject_id, self.collection_prefix)
            success = await self.vector_store.add_documents(vector_documents, collection_name)

            if success:
                logger.info(
                    f"✅ Successfully indexed {len(vector_documents)} chunks for {filename}"
                )
                return {
                    "success": True,
                    "message": f"Document {filename} processed and indexed successfully",
                    "statistics": {
                        "filename": filename,
                        "subject_id": subject_id,
                        "chunks_created": len(chunks),
                        "chunks_indexed": len(vector_documents),
                        "collection": collection_name,
                    },
                }
            return {"success": False, "error": "Failed to store document in vector database"}

        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"❌ Failed to process document {filename}: {exc}")
            return {
                "success": False,
                "error": f"Document processing failed: {exc}",
            }
