"""검색 요청/응답 Pydantic 모델."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResultMetadata(BaseModel):
    """검색 결과 메타데이터 모델"""
    chunk_index: Optional[int] = None
    doc_id: str
    doc_name: Optional[str] = None
    original_collection: str
    source: Optional[str] = None


class SearchResult(BaseModel):
    """검색 결과 단일 아이템 모델"""
    page_content: str
    metadata: SearchResultMetadata
    score: float


class SearchResponse(BaseModel):
    """검색 응답 모델"""
    results: List[SearchResult]