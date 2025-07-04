"""벡터 검색 & 하이브리드 검색 라우터."""
from fastapi import APIRouter, HTTPException
from app.services.search_service import SearchService
from app.models.search_model import SearchRequest, SearchResponse, RelatedSearchRequest

router = APIRouter()
search_service = SearchService()


@router.post("/vector", response_model=SearchResponse)
async def vector_search(request: SearchRequest):
    """
임베딩 벡터를 사용한 의미 기반 검색을 수행합니다.
    
    이 엔드포인트는 텍스트 쿼리를 임베딩 벡터로 변환한 후, 코사인 유사도를 기준으로
    의미적으로 가장 관련성이 높은 문서를 검색합니다. 검색 결과는 유사도 점수로 정렬됩니다.
    """
    try:
        return await search_service.vector_search(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/keyword", response_model=SearchResponse)
async def keyword_search(request: SearchRequest):
    """
순수한 BM25 기반 키워드 검색을 수행합니다.
    
    이 엔드포인트는 벡터 검색 없이 순수한 텍스트 기반 검색을 수행합니다.
    점수는 0~1 범위로 정규화됩니다.
    """
    try:
        return await search_service.keyword_search(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search_native(request: SearchRequest):
    """
OpenSearch 내장 하이브리드 검색을 수행합니다.
    
    이 엔드포인트는 OpenSearch의 내장 하이브리드 검색 기능을 사용합니다.
    검색 파이프라인을 통해 점수 정규화와 알고리즘을 통합하여 더 정확한 결과를 제공합니다.
    """
    try:
        return await search_service.hybrid_search_native(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid-reranked", response_model=SearchResponse)
async def hybrid_search_reranked(request: SearchRequest):
    """
크로스 인코더 재정렬을 적용한 하이브리드 검색을 수행합니다.
    
    이 엔드포인트는 벡터 검색과 BM25 검색 결과를 결합한 후 크로스 인코더 모델(BAAI/bge-reranker-v2-m3)을 사용하여 재정렬합니다.
    크로스 인코더는 쿼리와 응답 간의 의미적 관계를 더 정확하게 평가하여 더 관련성 높은 결과를 제공합니다.
    """
    try:
        return await search_service.hybrid_search_reranked(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-related", response_model=SearchResponse)
async def search_related_documents(request: RelatedSearchRequest):
    """
특정 문서 ID 목록 내에서만 검색을 수행합니다.
    
    이 엔드포인트는 지정된 문서 ID 목록에 속하는 청크들 중에서만 검색을 수행합니다.
    크로스 인코더 재정렬 하이브리드 검색 방식을 사용하여 정확한 결과를 제공합니다.
    """
    try:
        return await search_service.search_related(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
