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
        """Combine chunks into a single context string within length limit."""
        context_parts: List[str] = []
        total_length = 0

        for chunk in relevant_chunks:
            content = chunk["content"]
            metadata = chunk["metadata"]

            # Stop if we would exceed the maximum context length
            if total_length + len(content) > self.max_context_length:
                break

            source_info = f"[Nguồn: {metadata.get('filename', 'Unknown')}]"
            chunk_text = f"{source_info}\n{content}\n"

            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        return "\n---\n".join(context_parts) 