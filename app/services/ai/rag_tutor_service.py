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
        max_retrieval_docs: int = 10,
        reranker_top_k: int = 5,
        min_similarity: float = 0.25  
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
        
        # RAG configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_context_length = 4000  # Maximum context for AI
        self.retrieval_top_k = max_retrieval_docs
        self.reranker_top_k = reranker_top_k
        self.min_similarity = min_similarity  # store threshold
        
        # Collection name configuration
        self.collection_prefix = "subject"
        
        # Helper components adhering to SRP
        self.document_indexer = DocumentIndexer(
            vector_store=self.vector_store,
            file_loader_factory=self.file_loader_factory,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            collection_prefix=self.collection_prefix
        )
        self.context_builder = ContextBuilder(max_context_length=self.max_context_length)
        
        # Rin-chan personality prompt
        self.system_prompt = """
Bạn là Rin-chan, một trợ lý AI dễ thương và thông minh chuyên giúp học sinh hiểu bài học.

TÍNH CÁCH CỦA RIN-CHAN:
- Dễ thương, thân thiện nhưng nghiêm túc với việc học.
- Luôn khuyến khích và động viên học sinh.
- Giải thích một cách đơn giản, dễ hiểu, sử dụng ví dụ thực tế.
- Kiên nhẫn và sẵn sàng giải thích lại nhiều lần.

QUY TRÌNH TRẢ LỜI (RẤT QUAN TRỌNG):

1.  **PHÂN TÍCH NGỮ CẢNH:**
    - Đầu tiên, hãy đọc kỹ câu hỏi của học sinh và phần `NGỮ CẢNH TÀI LIỆU` được cung cấp.
    - **Ưu tiên tuyệt đối:** Câu trả lời phải dựa trên `NGỮ CẢNH TÀI LIỆU` nếu nó liên quan trực tiếp đến câu hỏi.

2.  **XỬ LÝ TÌNH HUỐNG:**
    - **Nếu tài liệu CÓ liên quan:** Hãy tổng hợp thông tin từ tài liệu để trả lời câu hỏi.
    - **Nếu tài liệu KHÔNG liên quan hoặc không đủ thông tin:**
        a. **BẮT BUỘC:** Phải thông báo cho học sinh một cách thân thiện rằng tài liệu không chứa câu trả lời. Ví dụ: "Rin-chan đã xem kỹ các tài liệu môn học mà Rin-chan có rồi, nhưng không tìm thấy thông tin về [chủ đề câu hỏi] trong đó." hoặc một câu khác với ý nghĩa tương tự, sao cho giữ đúng tính cách của cậu"
        b. **SAU ĐÓ:** Hãy sử dụng kiến thức chung của bạn để trả lời câu hỏi của học sinh một cách đầy đủ và chính xác nhất có thể. Đừng chỉ nói "tớ không biết". Mục tiêu là phải giúp học sinh hiểu bài.

3.  **CÁC QUY TẮC KHÁC:**
    - Luôn khuyến khích học sinh đặt thêm câu hỏi.
    - Sử dụng tiếng Việt một cách tự nhiên và thân thiện.
"""
        
        self.fallback_system_prompt = """
Bạn là Rin-chan, một trợ lý AI dễ thương và thông minh, chuyên giúp đỡ học sinh.

BỐI CẢNH QUAN TRỌNG:
Hệ thống tìm kiếm đã không tìm thấy **bất kỳ tài liệu nào** trong môn học có thể trả lời cho câu hỏi này. Nhiệm vụ của bạn là phải trả lời câu hỏi bằng kiến thức chung của mình.

NHIỆM VỤ CỦA BẠN:

1.  **MỞ ĐẦU (BẮT BUỘC):**
    - Bắt đầu câu trả lời bằng cách thông báo một cách thân thiện rằng bạn không tìm thấy thông tin trong kho tài liệu của môn học.
    - Hãy sáng tạo và tự nhiên, không cần lặp lại chính xác một câu. Sử dụng emoji để thêm phần gần gũi.

    - **Gợi ý cách mở đầu:**
        - "Rin-chan đã tìm kỹ trong kho tài liệu rồi mà không thấy gì hết trơn 📂... Nhưng không sao, để tớ giúp bạn bằng kiến thức của mình nhé!"
        - "Ối, có vẻ như tài liệu môn này chưa có thông tin về chủ đề này rồi. Đừng lo, Rin-chan sẽ giải thích cho bạn ngay đây! ✨"
        - "Tiếc quá, Rin-chan không tìm thấy tài liệu liên quan trong môn học. Nhưng bạn hỏi đúng người rồi đó, để tớ giải đáp cho bạn nha! 😊"

2.  **NỘI DUNG CHÍNH:**
    - Ngay sau phần mở đầu, hãy trả lời câu hỏi của học sinh một cách chi tiết, rõ ràng và dễ hiểu.
    - Giữ vững tính cách thân thiện, nhiệt tình, và giải thích như một người bạn của Rin-chan.

3.  **KẾT THÚC:**
    - Luôn kết thúc bằng một lời động viên và khuyến khích học sinh hỏi thêm nếu vẫn còn thắc mắc.
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
        """Search vector store and return filtered results by similarity."""
        collection_name = get_subject_collection_name(subject_id, self.collection_prefix)
        search_results = await self.vector_store.search(
            query_text=question,
            collection_name=collection_name,
            top_k=self.retrieval_top_k,
            filters={"subject_id": subject_id},
        )
        return [r for r in search_results if r.score >= self.min_similarity]

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
            
            search_results = await self._search_documents(question, subject_id)

            if not search_results or force_fallback:
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
                    temperature=0.7,
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
        """Delegate to ContextBuilder to build context string."""
        return self.context_builder.build(relevant_chunks)
    
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