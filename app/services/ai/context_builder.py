"""
ContextBuilder – responsible for assembling a context string from
retrieved document chunks, separated from business logic to satisfy SRP.
"""
from __future__ import annotations

from typing import List, Dict, Any


class ContextBuilder:
    """Compose the text context fed into the language model."""

    def __init__(self, max_context_length: int = 4000) -> None:
        self.max_context_length = max_context_length

    def build(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Combine chunks into a single context string within length limit, optimized for MCQ."""
        if not relevant_chunks:
            return ""

        # Sort chunks by rerank_score (nếu có), nếu không thì theo retrieval_score
        sorted_chunks = sorted(
            relevant_chunks,
            key=lambda x: x.get("rerank_score") if x.get("rerank_score") is not None else x.get("retrieval_score", 0),
            reverse=True
        )

        context_parts: List[str] = []
        current_length = 0

        for i, chunk in enumerate(sorted_chunks):
            content = chunk["content"]
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            
            # Add source info for better context
            chunk_with_source = f"[Nguồn: {filename}]\n{content}"
            
            # Check if adding this chunk would exceed max context length
            if current_length + len(chunk_with_source) > self.max_context_length:
                if i == 0:  # Always include at least one chunk
                    chunk_with_source = chunk_with_source[:self.max_context_length - 100] + "..."
                    context_parts.append(chunk_with_source)
                break
            
            context_parts.append(chunk_with_source)
            current_length += len(chunk_with_source)
        
        return "\n\n---\n\n".join(context_parts) 