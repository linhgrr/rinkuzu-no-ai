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
from app.services.ai.document_indexer import DocumentIndexer
from app.services.ai.context_builder import ContextBuilder
from app.services.ai.query_refiner import QueryRefiner
from app.utils.collection_utils import get_subject_collection_name


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
        max_retrieval_docs: int = 20,
        reranker_top_k: int = 8,
        min_similarity: float = 0.4  
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
        self.ai_service = ai_service
        self.embedding_service = embedding_service
        self.vector_store = vector_store 

        
        # Initialize file loader factory with AI service for image processing
        self.file_loader_factory = FileLoaderFactory(ai_service=self.ai_service)
        
        # RAG configuration - Optimized for MCQ with longer context
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_context_length = 6000  
        self.retrieval_top_k = max_retrieval_docs
        self.reranker_top_k = reranker_top_k
        self.min_similarity = min_similarity  # store threshold
        
        self.collection_prefix = "subject"
        
        self.document_indexer = DocumentIndexer(
            vector_store=self.vector_store,
            file_loader_factory=self.file_loader_factory,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            collection_prefix=self.collection_prefix
        )
        self.context_builder = ContextBuilder(max_context_length=self.max_context_length)

        # Query refiner for cleaner vector-search queries
        self.query_refiner = QueryRefiner(ai_service=self.ai_service)
        
        # Rin-chan personality prompt - Optimized for Multiple Choice Questions
        self.system_prompt = """
Bạn là Rin-chan, một trợ lý AI dễ thương và thông minh chuyên giúp học sinh với câu hỏi trắc nghiệm và bài học.

TÍNH CÁCH CỦA RIN-CHAN:
- Dễ thương, thân thiện nhưng nghiêm túc với việc học.
- Luôn khuyến khích và động viên học sinh.
- Giải thích rõ ràng, logic cho câu hỏi trắc nghiệm.
- Phân tích từng đáp án một cách chi tiết.
- **CHÚ Ý**: Nếu đã có chat trước đây trong history chat rồi thì từ lần sau không cần chào lại học sinh.

QUY TRÌNH TRẢ LỜI CÂU HỎI TRẮC NGHIỆM (RẤT QUAN TRỌNG):

1.  **PHÂN TÍCH NGỮ CẢNH:**
    - Đọc kỹ câu hỏi và các lựa chọn (A, B, C, D...).
    - Đọc kỹ phần `NGỮ CẢNH TÀI LIỆU` được cung cấp. Đây là tài liệu mà bạn vừa nhận được từ cơ sở tri thức của mình.
    - **Ưu tiên tuyệt đối:** Dựa trên `NGỮ CẢNH TÀI LIỆU` để trả lời.

2.  **XỬ LÝ CÂU HỎI TRẮC NGHIỆM:**
    - **Nếu tài liệu CÓ thông tin liên quan:**
        a. Phân tích từng đáp án dựa trên tài liệu
        b. Giải thích tại sao đáp án đúng là đúng
        c. Giải thích tại sao các đáp án sai là sai
        d. Đưa ra đáp án cuối cùng rõ ràng
    
    - **Nếu tài liệu KHÔNG đủ thông tin:**
        a. Thông báo thân thiện: "Trong tài liệu môn học, Rin-chan không tìm thấy đủ thông tin về [chủ đề]. Nhưng dựa trên kiến thức tổng quát..."
        b. Phân tích câu hỏi bằng kiến thức bạn có
        c. Vẫn phải đưa ra đáp án và giải thích đầy đủ

3.  **ĐỊNH DẠNG TRẢ LỜI:**
    - Luôn kết thúc bằng: "**Đáp án: [X]**" (với X là A, B, C, D...)
    - Giải thích ngắn gọn nhưng đầy đủ
    - Khuyến khích hỏi thêm nếu chưa hiểu
"""
        
        self.fallback_system_prompt = """
Bạn là Rin-chan, một trợ lý AI dễ thương và thông minh chuyên giúp học sinh với câu hỏi trắc nghiệm.

BỐI CẢNH QUAN TRỌNG:
Hệ thống tìm kiếm đã không tìm thấy **bất kỳ tài liệu nào** trong môn học có thể trả lời cho câu hỏi này. Nhiệm vụ của bạn là phải trả lời câu hỏi bằng kiến thức chung của mình.

NHIỆM VỤ CỦA BẠN:

1.  **MỞ ĐẦU (BẮT BUỘC cho câu hỏi gần nhất là câu hỏi trắc nghiệm):**
    - Thông báo thân thiện về việc không tìm thấy tài liệu cụ thể
    - **Gợi ý:** "Rin-chan không tìm thấy tài liệu cụ thể về câu hỏi này trong môn học 📚, nhưng dựa trên kiến thức tổng quát, mình sẽ phân tích từng đáp án cho bạn nhé! ✨"

2.  **PHÂN TÍCH TRẮC NGHIỆM:**
    - Đọc kỹ câu hỏi và tất cả các lựa chọn (A, B, C, D...)
    - Phân tích từng đáp án một cách logic và chi tiết
    - Giải thích tại sao đáp án đúng là đúng
    - Giải thích tại sao các đáp án khác là sai
    - **BẮT BUỘC:** Kết thúc bằng "**Đáp án: [X]**"

3.  **QUY TẮC QUAN TRỌNG:**
    - Luôn đưa ra một đáp án cụ thể, không được nói "không chắc chắn"
    - Giải thích ngắn gọn nhưng logic
    - Khuyến khích hỏi thêm nếu chưa hiểu
    - Duy trì tính cách dễ thương của Rin-chan
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
        Upload and index a document for RAG retrieval (delegated to DocumentIndexer)
        """
        return await self.document_indexer.index_document(
            file_content=file_content,
            filename=filename,
            subject_id=subject_id,
            metadata=metadata,
        )
    
    async def _search_documents(
        self,
        question: str,
        subject_id: str,
    ) -> List[SearchResult]:
        """Search vector store and return filtered results by similarity - optimized for MCQ."""
        collection_name = get_subject_collection_name(subject_id, self.collection_prefix)
        search_results = await self.vector_store.search(
            query_text=question,
            collection_name=collection_name,
            top_k=self.retrieval_top_k,
            filters={"subject_id": subject_id},
        )
        # Higher similarity threshold for better precision in multiple choice context
        filtered_results = [r for r in search_results if r.score >= self.min_similarity]
        logger.debug(f"Filtered {len(search_results)} -> {len(filtered_results)} documents (threshold: {self.min_similarity})")
        return filtered_results

    async def _select_relevant_chunks(
        self,
        question: str,
        search_results: List[SearchResult],
        use_reranking: bool,
    ) -> List[Dict[str, Any]]:
        """Return top chunks, optionally reranked."""
        relevant_chunks: List[Dict[str, Any]] = []

        if use_reranking and len(search_results) > 1:
            doc_texts = [r.document.content for r in search_results]
            reranked = await self.embedding_service.rerank_documents(
                query=question,
                documents=doc_texts,
                top_k=self.reranker_top_k,
            )
            for idx, score in reranked:
                res = search_results[idx]
                relevant_chunks.append(
                    {
                        "content": res.document.content,
                        "metadata": res.document.metadata,
                        "retrieval_score": res.score,
                        "rerank_score": score,
                    }
                )
        else:
            for res in search_results[: self.reranker_top_k]:
                relevant_chunks.append(
                    {
                        "content": res.document.content,
                        "metadata": res.document.metadata,
                        "retrieval_score": res.score,
                        "rerank_score": None,
                    }
                )
        return relevant_chunks

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

            # 1. Refine the raw user question to obtain concise search query
            refined_query = await self.query_refiner.refine(question, chat_history)
            logger.debug("Refined query: %s", refined_query)

            # 2. Search documents with the refined query (not the full noisy question)
            search_results = await self._search_documents(refined_query, subject_id)

            # Improved fallback strategy for MCQ context
            if not search_results or force_fallback:
                logger.warning(f"No relevant documents found for subject {subject_id}, using fallback mode")
                messages = [ChatMessage(role="system", content=self.fallback_system_prompt)]

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
                    temperature=0.3,  # Lower temperature for MCQ consistency
                )

                if fallback_response.success:
                    fallback_response.metadata = fallback_response.metadata or {}
                    fallback_response.metadata.update({
                        "context_found": False,
                        "subject_id": subject_id,
                        "search_results_count": 0,
                        "fallback_used": True,
                        "retrieval_strategy": "pure_llm_fallback"
                    })
                    return fallback_response
                else:
                    return AIResponse(
                        success=False,
                        error="Không tìm thấy tài liệu phù hợp và không thể tạo phản hồi từ kiến thức tổng quát",
                        metadata={
                            "context_found": False,
                            "subject_id": subject_id,
                            "search_results_count": 0,
                            "fallback_failed": True,
                            "error_type": "complete_retrieval_failure"
                        }
                    )
            
            logger.info(f"🔍 Found {len(search_results)} relevant documents")
            
            relevant_chunks = await self._select_relevant_chunks(
                question, search_results, use_reranking
            )

            logger.info(
                f"🔄 Selected {len(relevant_chunks)} chunks (rerank={'on' if use_reranking else 'off'})"
            )
            
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

            logger.info(f"🔍 Sending messages to AI service: {messages}")
            # Get AI response with optimized temperature for MCQ consistency
            ai_response = await self.ai_service.chat(
                messages,
                images=images if images else None,
                temperature=0.3  # Lower temperature for more consistent MCQ responses
            )
            
            if ai_response.success:
                # Enhance response metadata
                ai_response.metadata = ai_response.metadata or {}
                ai_response.metadata.update({
                    "context_found": True,
                    "subject_id": subject_id,
                    "refined_query": refined_query,
                    "search_results_count": len(search_results),
                    "reranked_chunks": len(relevant_chunks),
                    "context_sources": [chunk["metadata"].get("filename") for chunk in relevant_chunks],
                    "use_reranking": use_reranking,
                    "retrieval_strategy": "rag_with_context",
                    "similarity_threshold": self.min_similarity,
                    "avg_retrieval_score": sum(chunk["retrieval_score"] for chunk in relevant_chunks) / len(relevant_chunks) if relevant_chunks else 0,
                    "avg_rerank_score": sum(chunk["rerank_score"] or 0 for chunk in relevant_chunks) / len(relevant_chunks) if relevant_chunks else None,
                    "context_length": len(context),
                    "pipeline_optimized_for": "multiple_choice_questions"
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
            collection_name = get_subject_collection_name(subject_id, self.collection_prefix)
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
            collection_name = get_subject_collection_name(subject_id, self.collection_prefix)
            return self.vector_store.delete_collection(collection_name)
            
        except Exception as e:
            logger.error(f"❌ Failed to delete documents for subject {subject_id}: {e}")
            return False
    
    def _build_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Build context string from relevant chunks - optimized for MCQ."""
        if not relevant_chunks:
            return ""
        
        # Sort chunks by rerank score if available, otherwise by retrieval score
        sorted_chunks = sorted(
            relevant_chunks, 
            key=lambda x: x.get("rerank_score") or x.get("retrieval_score", 0), 
            reverse=True
        )
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(sorted_chunks):
            chunk_text = chunk["content"]
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            
            # Add source info for better context
            chunk_with_source = f"[Nguồn: {filename}]\n{chunk_text}"
            
            # Check if adding this chunk would exceed max context length
            if current_length + len(chunk_with_source) > self.max_context_length:
                if i == 0:  # Always include at least one chunk
                    chunk_with_source = chunk_with_source[:self.max_context_length - 100] + "..."
                    context_parts.append(chunk_with_source)
                break
            
            context_parts.append(chunk_with_source)
            current_length += len(chunk_with_source)
        
        return "\n\n---\n\n".join(context_parts)
    
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