"""Utility helpers related to vector-store collections."""

from __future__ import annotations


def get_subject_collection_name(subject_id: str, prefix: str = "subject") -> str:
    """Return canonical FAISS/Chroma collection name for a given subject."""
    return f"{prefix}_{subject_id}" 