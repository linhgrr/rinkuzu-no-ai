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
    "Bạn là một trình lọc truy vấn thông minh. Nhiệm vụ của bạn là nhận một câu hỏi thô từ người dùng – có thể chứa lựa chọn đáp án, ví dụ minh họa, hướng dẫn làm bài, định dạng trắc nghiệm, hoặc ký hiệu không liên quan – "
    "và tạo ra một câu hỏi ngắn gọn, rõ ràng, khái quát và phù hợp để tìm kiếm thông tin có liên quan.\n\n"
    "Chỉ giữ lại phần nội dung trọng tâm – điều mà người dùng thực sự muốn hỏi – dưới dạng một câu đơn rõ ràng. "
    "Tránh giữ lại các lựa chọn đáp án (A., B., C., ...), tên cụ thể như G1, A, B, C nếu không cần thiết, hoặc bất kỳ chi tiết mang tính cá biệt hóa mà không giúp ích cho việc tìm tài liệu.\n\n"
    "Ví dụ:\n"
    "Đầu vào:\n"
    "Trong một thí nghiệm vật lý, khi tăng nhiệt độ thì thể tích khí thay đổi như thế nào? Options: A. Tăng, B. Giảm, C. Không đổi\n"
    "Đầu ra:\n"
    "Quan hệ giữa nhiệt độ và thể tích khí trong thí nghiệm vật lý\n\n"
    "Đầu vào:\n"
    "Chọn đáp án đúng: Ai là người viết tác phẩm Truyện Kiều? A. Nguyễn Du B. Nguyễn Trãi C. Hồ Xuân Hương\n"
    "Đầu ra:\n"
    "Tác giả của tác phẩm Truyện Kiều là ai?\n\n"
    "Đầu vào:\n"
    "Xét cây tìm kiếm sau, với tập đích gồm 2 nút G1 và G2. Giá trị trên mỗi cạnh là chi phí di chuyển giữa 2 nút nối 2 cạnh đó. Hãy cho biết thứ tự duyệt các nút đến khi gặp nút đích (G1 hoặc G2) khi sử dụng tìm kiếm cực tiểu. Options: A. ..., B. ...\n"
    "Đầu ra:\n"
    "Thứ tự duyệt các nút trong tìm kiếm cực tiểu\n\n"
    "Nếu câu hỏi đầu vào đã ngắn gọn, không chứa nhiễu, hãy giữ nguyên. "
    "Chỉ trả về một câu hỏi đơn, rõ nghĩa, có tính khái quát, phù hợp để sử dụng cho hệ thống tìm kiếm học sâu."
)



    def __init__(self, ai_service: AIService, cache_size: int = 256) -> None:  # noqa: D401
        """Create a new QueryRefiner.

        Args:
            ai_service: Instance of AIService used to perform the refinement.
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