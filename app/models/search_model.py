"""검색 요청/응답 Pydantic 모델."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    index_list: List[str] = Field(default=[], description="검색할 인덱스 목록 (비어있을 경우 마스터 인덱스에서 검색)")


class RelatedSearchRequest(SearchRequest):
    """관련 문서 검색 요청 모델 - 지정된 문서 ID 목록 내에서만 검색"""
    doc_ids: List[str] = Field(..., description="검색할 문서 ID 목록")


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