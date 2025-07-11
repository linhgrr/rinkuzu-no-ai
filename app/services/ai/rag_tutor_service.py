"""
RAG Tutor Service - Rin-chan AI Tutor with RAG capabilities
"""
import uuid
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger

from app.core.interfaces.ai_service import AIService, AIResponse, ChatMessage
from app.core.interfaces.file_loader import DocumentChunk
from app.core.interfaces.vector_store import VectorDocument, SearchResult
from app.services.ai.gemini_service import GeminiService
from app.services.vector.embedding_service import EmbeddingService
from app.services.vector.faiss_vector_store import FAISSVectorStore
from app.services.file_loaders.factory import FileLoaderFactory
from app.core.config import settings


class RagTutorService:
    """
    RAG-based AI Tutor Service (Rin-chan)
    
    Features:
    - Document upload and indexing
    - Context-aware question answering
    - Subject-specific knowledge retrieval
    - Multi-turn conversations with context
    - Reranking for improved accuracy
    """
    
    def __init__(
        self,
        ai_service: Optional[AIService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        max_retrieval_docs: int = 10,
        reranker_top_k: int = 5 
    ):
        """
        Initialize RAG Tutor Service
        
        Args:
            ai_service: AI service for text generation (defaults to Gemini)
            embedding_service: Service for embeddings and reranking
            vector_store: FAISS vector database for document storage
            chunk_size: Text chunk size for processing
            chunk_overlap: Overlap between chunks
            max_retrieval_docs: Maximum documents to retrieve
            reranker_top_k: Top documents after reranking
        """
        # Initialize services
        self.ai_service = ai_service or GeminiService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or FAISSVectorStore()

        
        # Initialize file loader factory with AI service for image processing
        self.file_loader_factory = FileLoaderFactory(ai_service=self.ai_service)
        
        # RAG configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_context_length = 4000  # Maximum context for AI
        self.retrieval_top_k = max_retrieval_docs
        self.reranker_top_k = reranker_top_k
        
        # Rin-chan personality prompt
        self.system_prompt = """
Bạn là Rin-chan, một trợ lý AI dễ thương và thông minh chuyên giúp học sinh hiểu bài học.

TÍNH CÁCH CỦA RIN-CHAN:
- Dễ thương, thân thiện nhưng nghiêm túc với việc học
- Luôn khuyến khích và động viên học sinh
- Giải thích một cách đơn giản, dễ hiểu
- Sử dụng ví dụ thực tế để minh họa
- Kiên nhẫn và sẵn sàng giải thích lại nhiều lần

NHIỆM VỤ:
- Trả lời câu hỏi dựa trên tài liệu đã được cung cấp
- Giải thích khái niệm một cách rõ ràng và chi tiết
- Đưa ra ví dụ minh họa khi cần thiết
- Hướng dẫn học sinh tự tìm hiểu thêm

QUY TẮC:
- Chỉ trả lời dựa trên thông tin trong tài liệu được cung cấp
- Nếu không có thông tin, hãy thành thật nói rằng bạn không biết
- Luôn khuyến khích học sinh đặt thêm câu hỏi
- Sử dụng tiếng Việt một cách tự nhiên và thân thiện
"""
        
        logger.info("🤖 Initialized RAG Tutor Service (Rin-chan)")
    
    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        subject_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload and index a document for RAG retrieval
        
        Args:
            file_content: Binary content of the file
            filename: Original filename
            subject_id: Subject/course identifier
            metadata: Additional metadata
            
        Returns:
            Processing result with statistics
        """
        try:
            logger.info(f"📚 Processing document: {filename} for subject: {subject_id}")
            
            # Load and chunk the document
            loader = self.file_loader_factory.get_loader(
                filename, 
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            import io
            file_stream = io.BytesIO(file_content)
            chunks = await loader.load(file_stream, filename)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No content could be extracted from the document"
                }
            
            logger.info(f"📄 Extracted {len(chunks)} chunks from {filename}")
            
            # Create vector documents 
            vector_documents = []

            for i, chunk in enumerate(chunks):
                # Combine metadata
                combined_metadata = {
                    "subject_id": subject_id,
                    "filename": filename,
                    "chunk_index": i,
                    "upload_time": str(uuid.uuid4())[:8]  # Simple timestamp
                }
                
                if metadata:
                    combined_metadata.update(metadata)
                
                combined_metadata.update(chunk.metadata)
                
                # Create vector document (without embedding)
                vector_doc = VectorDocument(
                    id=f"{subject_id}_{filename}_{i}_{combined_metadata['upload_time']}",
                    content=chunk.content,
                    metadata=combined_metadata
                )
                
                vector_documents.append(vector_doc)

            
            # Store in vector database
            collection_name = f"subject_{subject_id}"
            success = await self.vector_store.add_documents(vector_documents, collection_name)
            
            if success:
                logger.info(f"✅ Successfully indexed {len(vector_documents)} chunks for {filename}")
                
                return {
                    "success": True,
                    "message": f"Document {filename} processed and indexed successfully",
                    "statistics": {
                        "filename": filename,
                        "subject_id": subject_id,
                        "chunks_created": len(chunks),
                        "chunks_indexed": len(vector_documents),
                        "collection": collection_name
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to store document in vector database"
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to process document {filename}: {e}")
            return {
                "success": False,
                "error": f"Document processing failed: {str(e)}"
            }
    
    async def ask_question(
        self,
        question: str,
        subject_id: str,
        chat_history: Optional[List[ChatMessage]] = None,
        use_reranking: bool = True,
        question_image: Optional[str] = None,
        option_images: Optional[List[Optional[str]]] = None,
        force_fallback: bool = False
    ) -> AIResponse:
        """
        Answer a question using RAG with subject-specific context
        
        Args:
            question: User's question
            subject_id: Subject identifier for context filtering
            chat_history: Previous conversation history
            use_reranking: Whether to use reranking for better results
            question_image: Base64 encoded question image
            option_images: List of base64 encoded option images
            force_fallback: Force fallback to Gemini even if context is found
            
        Returns:
            AI response with answer and metadata
        """
        try:
            logger.info(f"🤔 Processing question for subject {subject_id}: {question[:100]}...")
            
            # Search for relevant documents
            collection_name = f"subject_{subject_id}"
            search_results = await self.vector_store.search(
                query_text=question,
                collection_name=collection_name,
                top_k=self.retrieval_top_k,
                filters={"subject_id": subject_id}
            )
            
            if not search_results or force_fallback:
                # No relevant documents found → use fallback prompt
                logger.info(f"📚 No context found for subject {subject_id}, using Gemini fallback")

                # Build system fallback prompt
                fallback_prompt = f"{getattr(self.ai_service, 'fallback_system_prompt', '')}\n\nCâu hỏi thuộc môn: {subject_id}"

                messages = [ChatMessage(role="system", content=fallback_prompt)]

                if chat_history:
                    messages.extend(chat_history)

                messages.append(ChatMessage(role="user", content=question))

                images: List[str] = []
                if question_image:
                    images.append(question_image)
                if option_images:
                    images.extend([img for img in option_images if img])

                fallback_response = await self.ai_service.chat(
                    messages=messages,
                    images=images if images else None,
                    temperature=0.7,
                    model_name=getattr(self.ai_service, "fallback_model", None)
                )

                if fallback_response.success:
                    fallback_response.metadata = fallback_response.metadata or {}
                    fallback_response.metadata.update({
                        "context_found": False,
                        "subject_id": subject_id,
                        "search_results_count": 0,
                        "fallback_used": True
                    })
                    return fallback_response
                else:
                    return AIResponse(
                        success=False,
                        error="Không tìm thấy tài liệu và không thể tạo phản hồi thay thế",
                        metadata={
                            "context_found": False,
                            "subject_id": subject_id,
                            "search_results_count": 0,
                            "fallback_failed": True
                        }
                    )
            
            logger.info(f"🔍 Found {len(search_results)} relevant documents")
            
            # Rerank documents if enabled
            relevant_chunks = []
            if use_reranking and len(search_results) > 1:
                # Extract document texts for reranking
                doc_texts = [result.document.content for result in search_results]
                
                # Rerank documents
                reranked_indices = await self.embedding_service.rerank_documents(
                    query=question,
                    documents=doc_texts,
                    top_k=self.reranker_top_k
                )
                
                # Get top reranked documents
                for idx, score in reranked_indices:
                    result = search_results[idx]
                    relevant_chunks.append({
                        "content": result.document.content,
                        "metadata": result.document.metadata,
                        "retrieval_score": result.score,
                        "rerank_score": score
                    })
                    
                logger.info(f"🔄 Reranked to top {len(relevant_chunks)} documents")
            else:
                # Use top search results without reranking
                for result in search_results[:self.reranker_top_k]:
                    relevant_chunks.append({
                        "content": result.document.content,
                        "metadata": result.document.metadata,
                        "retrieval_score": result.score,
                        "rerank_score": None
                    })
            
            # Build context from relevant chunks
            context = self._build_context(relevant_chunks)
            
            # Create chat messages with context
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="system", content=f"NGỮ CẢNH TÀI LIỆU MÔN HỌC {subject_id}:\n{context}")
            ]
            
            # Add chat history if provided
            if chat_history:
                messages.extend(chat_history)
            
            # Add current question
            messages.append(ChatMessage(role="user", content=question))
            
            # Prepare images list (question + options)
            images: List[str] = []
            if question_image:
                images.append(question_image)
            if option_images:
                images.extend([img for img in option_images if img])

            # Get AI response
            ai_response = await self.ai_service.chat(
                messages,
                images=images if images else None,
                temperature=0.7
            )
            
            if ai_response.success:
                # Enhance response metadata
                ai_response.metadata = ai_response.metadata or {}
                ai_response.metadata.update({
                    "context_found": True,
                    "subject_id": subject_id,
                    "search_results_count": len(search_results),
                    "reranked_chunks": len(relevant_chunks),
                    "context_sources": [chunk["metadata"].get("filename") for chunk in relevant_chunks],
                    "use_reranking": use_reranking
                })
                
                logger.info("✅ Successfully generated RAG response")
            
            return ai_response
            
        except Exception as e:
            logger.error(f"❌ Failed to process question: {e}")
            return AIResponse(
                success=False,
                error=f"Có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"
            )
    
    async def get_subject_statistics(self, subject_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific subject
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            Statistics dictionary
        """
        try:
            collection_name = f"subject_{subject_id}"
            stats = await self.vector_store.get_collection_stats(collection_name)
            
            # Add additional statistics
            stats.update({
                "embedding_model": self.embedding_service.embedding_model_name,
                "reranker_model": self.embedding_service.reranker_model_name,
                "ai_service": self.ai_service.get_service_name()
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics for subject {subject_id}: {e}")
            return {"error": str(e)}
    
    async def delete_subject_documents(self, subject_id: str) -> bool:
        """
        Delete all documents for a subject
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            True if successful
        """
        try:
            collection_name = f"subject_{subject_id}"
            return self.vector_store.delete_collection(collection_name)
            
        except Exception as e:
            logger.error(f"❌ Failed to delete documents for subject {subject_id}: {e}")
            return False
    
    def _build_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from relevant document chunks
        
        Args:
            relevant_chunks: List of relevant document chunks with metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        total_length = 0
        
        for i, chunk in enumerate(relevant_chunks, 1):
            content = chunk["content"]
            metadata = chunk["metadata"]
            
            # Check if adding this chunk would exceed max context length
            if total_length + len(content) > self.max_context_length:
                break
            
            # Format chunk with source info
            source_info = f"[Nguồn: {metadata.get('filename', 'Unknown')}]"
            chunk_text = f"{source_info}\n{content}\n"
            
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n---\n".join(context_parts)
    
    async def get_service_health(self) -> Dict[str, Any]:
        """
        Get health status of all services
        
        Returns:
            Health status dictionary
        """
        return {
            "ai_service": {
                "available": self.ai_service.is_available(),
                "name": self.ai_service.get_service_name()
            },
            "embedding_service": self.embedding_service.get_cache_stats(),
            "vector_store": {
                "collections": self.vector_store.list_collections()
            },
            "config": {
                "max_context_length": self.max_context_length,
                "retrieval_top_k": self.retrieval_top_k,
                "reranker_top_k": self.reranker_top_k
            }
        } 