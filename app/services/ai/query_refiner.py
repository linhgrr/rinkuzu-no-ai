from __future__ import annotations

"""
QueryRefiner – helper that removes noise (answer options, instructions, emojis, etc.)
from a raw user question so that the Vector Store receives a concise, highly-relevant
query for embedding search.

Example
-------
Input:
    Giải thuật nào sau đây xem xét đến ước lượng tới nút đích?
    Options:
    A. Depth-first search
    B. Best-first search
    C. A* search
    D. Greedy best-first search
Output:
    Thuật toán nào xem xét ước lượng (heuristic) tới nút đích?

If the input is already short and clean, the output should be identical.
"""

from typing import Dict
from loguru import logger

from app.core.interfaces.ai_service import AIService, ChatMessage


class QueryRefiner:
    """Use LLM to produce a cleaner search query from a potentially noisy prompt."""

    SYSTEM_PROMPT: str = (
    "Bạn là một chuyên gia trong việc tinh chỉnh câu hỏi của người dùng cho hệ thống tìm kiếm ngữ nghĩa (semantic search). "
    "Nhiệm vụ của bạn là diễn giải lại câu hỏi thô của người dùng thành một câu truy vấn ngắn gọn, rõ ràng và trực tiếp, phù hợp để tìm kiếm trong cơ sở dữ liệu vector.\n\n"
    "QUY TẮC:\n"
    "1. Giữ lại ý định cốt lõi của câu hỏi.\n"
    "2. Loại bỏ tất cả thông tin nhiễu: các tùy chọn trả lời (ví dụ: A, B, C, D), hướng dẫn (ví dụ: 'Hãy chọn câu trả lời đúng'), các câu chào hỏi, emoji, v.v.\n"
    "3. KHÔNG được trả lời câu hỏi. Chỉ tinh chỉnh lại câu hỏi.\n"
    "4. KHÔNG thêm bất kỳ lời giải thích hay đoạn văn giới thiệu nào. Chỉ trả về duy nhất câu truy vấn đã được tinh chỉnh.\n"
    "5. Nếu câu hỏi đã đủ rõ ràng và ngắn gọn, hãy trả về y nguyên.\n"
    "6. Giữ nguyên ngôn ngữ của câu hỏi gốc (ví dụ: nếu hỏi bằng tiếng Việt, câu truy vấn cũng phải bằng tiếng Việt).\n\n"
    "VÍ DỤ:\n\n"
    "Ví dụ 1:\n"
    "Input: Giải thuật nào sau đây xem xét đến ước lượng tới nút đích? Options: A. Depth-first search B. Best-first search C. A* search D. Greedy best-first search\n"
    "Output: Thuật toán nào xem xét ước lượng heuristic tới nút đích?\n\n"
    "Ví dụ 2:\n"
    "Input: RAG là gì?\n"
    "Output: RAG là gì?\n\n"
    "Ví dụ 3:\n"
    "Input: ad ơi cho mình hỏi làm thế nào để tối ưu hóa mô hình ngôn ngữ lớn ạ? cảm ơn ad nhiều 😘\n"
    "Output: cách tối ưu hóa mô hình ngôn ngữ lớn"
)

    def __init__(self, ai_service: AIService, cache_size: int = 256) -> None:  # noqa: D401
        """Create a new QueryRefiner.

        Args:
            ai_service: Instance of AIService (GeminiService) used to perform the refinement.
            cache_size: Maximum number of refined queries to keep in memory.
        """
        self.ai_service = ai_service
        self._cache_size = cache_size
        self._cache: Dict[int, str] = {}

    async def refine(self, raw_question: str) -> str:  # noqa: D401
        """Return a concise query string extracted from *raw_question*.

        If the LLM call fails for any reason, the original question is returned to guarantee
        that the pipeline continues to work.
        """
        cache_key = hash(raw_question)
        if cache_key in self._cache:
            return self._cache[cache_key]

        messages = [
            ChatMessage(role="system", content=self.SYSTEM_PROMPT),
            ChatMessage(role="user", content=raw_question),
        ]

        try:
            response = await self.ai_service.chat(messages, temperature=0.0)
            if response.success and response.content.strip():
                refined = response.content.strip()
                if len(self._cache) < self._cache_size:
                    self._cache[cache_key] = refined
                logger.info(f"🔍 Query refined from '{raw_question}' to '{refined}'")
                return refined
        except Exception as exc:  
            logger.warning("QueryRefiner failed; falling back to raw question: %s", exc)

        # Fallback to original question
        return raw_question 