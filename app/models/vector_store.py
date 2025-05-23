"""OpenSearch VectorStore 초기화 및 헬퍼 모듈 (정리 버전)"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import sys
import uuid
import logging
logger = logging.getLogger(__name__)

from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from langchain_core.documents import Document
from app.models.search_model import SearchResult, SearchResultMetadata

from app.config.settings import get_settings, Settings
from app.config.opensearch_config import index_mappings, MASTER_INDEX
from app.models.embedding_model import embedding_model

# 전역 싱글톤 핸들러
_vector_store: Optional[OpenSearchVectorSearch] = None
_os_client: Optional[OpenSearch] = None


# ──────────────────────────────────────────────────────────────
# 초기화
# ──────────────────────────────────────────────────────────────

def init_vector_store(settings: Settings):
    """OpenSearch 클라이언트 & VectorStore 전역 초기화."""
    global _vector_store, _os_client

    scheme = settings.opensearch_scheme
    base_url = f"{scheme}://{settings.opensearch_username}:{settings.opensearch_password}@{settings.opensearch_host}:{settings.opensearch_port}"

    _os_client = OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_auth=(settings.opensearch_username, settings.opensearch_password),
        use_ssl=scheme == "https",
        verify_certs=settings.opensearch_verify_certs,
        timeout=30,
    )

    # 인덱스 존재 확인 → 없으면 생성
    for name, mapping in index_mappings.items():
        if not _os_client.indices.exists(name):
            _os_client.indices.create(index=name, body=mapping)

    # 마스터 인덱스 VectorStore (embedding_function 지정)
    _vector_store = OpenSearchVectorSearch(
        opensearch_url=base_url,
        index_name=settings.master_index,
        embedding_function=embedding_model,
        http_auth=(settings.opensearch_username, settings.opensearch_password),
        use_ssl=scheme == "https",
        verify_certs=settings.opensearch_verify_certs,
        timeout=30,
        text_field="content",  # OpenSearch 매핑의 텍스트 필드명
        vector_field="embedding"  # OpenSearch 매핑의 벡터 필드명
    )


# ──────────────────────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────────────────────

def _get_store_for_index(index_name: str) -> OpenSearchVectorSearch:
    """마스터 이외 인덱스는 임시 VectorStore 인스턴스 생성."""
    settings = get_settings()
    if index_name == settings.master_index:
        return _vector_store  # type: ignore

    base_url = f"{settings.opensearch_scheme}://{settings.opensearch_username}:{settings.opensearch_password}@{settings.opensearch_host}:{settings.opensearch_port}"
    return OpenSearchVectorSearch(
        opensearch_url=base_url,
        index_name=index_name,
        embedding_function=embedding_model,
        http_auth=(settings.opensearch_username, settings.opensearch_password),
        use_ssl=settings.opensearch_scheme == "https",
        verify_certs=settings.opensearch_verify_certs,
        timeout=30,
        text_field="content",  # OpenSearch 매핑의 텍스트 필드명
        vector_field="embedding"  # 명시적으로 "embedding" 사용
    )


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

async def add_documents(index_name: str, docs: List[Document]) -> List[str]:
    """텍스트+메타데이터를 인덱스에 비동기적으로 추가 (직접 임베딩 계산)."""
    if _os_client is None:
        raise RuntimeError("OpenSearch client not initialized")
    
    # 임베딩 모델 가져오기 - 이미 임포트된 embedding_model 사용
    from app.models.embedding_model import embedding_model
    
    logger.info(f"Adding {len(docs)} documents to index '{index_name}' using direct OpenSearch API")
    
    added_ids = []
    for doc in docs:
        # 임베딩 계산
        embeddings = embedding_model.embed_documents([doc.page_content])
        
        # embed_documents는 중첩 리스트를 반환하므로 첫 번째 항목 추출
        embedding_vector = embeddings[0] if isinstance(embeddings, list) and len(embeddings) > 0 else embeddings
        
        # OpenSearch 문서 준비
        opensearch_doc = {
            "content": doc.page_content,  # 텍스트 내용은 content 필드에
            "embedding": embedding_vector,  # 임베딩 벡터는 embedding 필드에
            **doc.metadata                # 메타데이터는 최상위 레벨에 전개
        }
        
        # chunk_id를 문서 ID로 사용, 없으면 랜덤 생성
        doc_id = doc.metadata.get("chunk_id")
        if not doc_id:
            doc_id = str(uuid.uuid4())
            logger.warning(f"No chunk_id found in metadata, generated: {doc_id}")
        
        try:
            # OpenSearch에 문서 저장
            response = _os_client.index(
                index=index_name,
                body=opensearch_doc,
                id=doc_id,
                refresh=True  # 즉시 검색 가능하도록
            )
            
            added_ids.append(response["_id"])
            logger.debug(f"Document with ID {doc_id} added to index {index_name}")
        except Exception as e:
            logger.error(f"Error adding document with ID {doc_id} to index {index_name}: {e}", exc_info=True)
            raise
    
    logger.info(f"Successfully added {len(added_ids)} documents to index '{index_name}'")
    return added_ids


def similarity_search(query_vector, index_name: str, k: int = 10):
    """임베딩 벡터를 사용하여 유사 문서 검색 (OpenSearch kNN 쿼리 직접 사용)"""
    if _os_client is None:
        raise RuntimeError("OpenSearch client not initialized")
    
    # kNN 쿼리 구성 - embedding 필드 사용
    knn_query = {
        "size": k,
        "_source": True,  # 전체 문서 반환
        "query": {
            "knn": {
                "embedding": {  # OpenSearch에서 embedding 필드 사용
                    "vector": query_vector,  # 검색할 벡터
                    "k": k  # 반환할 결과 수
                }
            }
        }
    }
    
    try:
        # OpenSearch에 쿼리 요청
        response = _os_client.search(index=index_name, body=knn_query)
        
        # 결과를 SearchResult 객체 리스트로 변환
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            # 메타데이터 구성
            metadata = SearchResultMetadata(
                chunk_index=source.get("chunk_index"),
                doc_id=source.get("doc_id", ""),
                doc_name=source.get("doc_name", source.get("original_file_name", "")),
                original_collection=source.get("original_collection", ""),
                source=source.get("source", "")
            )
            
            # SearchResult 객체 생성
            result = SearchResult(
                page_content=source.get("content", ""),
                metadata=metadata,
                score=hit["_score"]
            )
            results.append(result)
        
        logger.debug(f"Found {len(results)} similar documents in index '{index_name}'")
        return results
    except Exception as e:
        logger.error(f"Error performing similarity search in index '{index_name}': {e}", exc_info=True)
        raise


def bm25_search(query: str, index_name: str, k: int = 10):
    """문서 키워드 검색 (BM25 알고리즘 사용)"""
    if _os_client is None:
        raise RuntimeError("Vector store not initialized")
    
    # BM25 쿼리 구성
    body = {"size": k, "query": {"match": {"content": query}}}
    
    try:
        # OpenSearch에 쿼리 요청
        res = _os_client.search(index=index_name, body=body)
        
        # 결과를 SearchResult 객체 리스트로 변환 (similarity_search와 동일한 형식)
        results = []
        for hit in res["hits"]["hits"]:
            source = hit["_source"]
            # 메타데이터 구성
            metadata = SearchResultMetadata(
                chunk_index=source.get("chunk_index"),
                doc_id=source.get("doc_id", ""),
                doc_name=source.get("doc_name", source.get("original_file_name", "")),
                original_collection=source.get("original_collection", ""),
                source=source.get("source", "")
            )
            
            # SearchResult 객체 생성
            result = SearchResult(
                page_content=source.get("content", ""),
                metadata=metadata,
                score=hit["_score"]
            )
            results.append(result)
        
        logger.debug(f"Found {len(results)} BM25 matches in index '{index_name}'")
        return results
    except Exception as e:
        logger.error(f"Error performing BM25 search in index '{index_name}': {e}", exc_info=True)
        raise


def hybrid_search_with_pipeline(query_text: str, query_vector: List[float], index_name: str, pipeline_name: str, k: int = 10):
    """OpenSearch 내장 하이브리드 검색 수행
    
    OpenSearch의 내장 하이브리드 검색 기능을 사용하여 벡터 검색과 키워드 검색을 동시에 수행합니다.
    검색 파이프라인을 통해 두 결과를 정규화하고 가중치를 적용하여 통합된 결과를 반환합니다.
    
    Args:
        query_text: 검색할 텍스트 쿼리
        query_vector: 쿼리의 벡터 임베딩
        index_name: 검색할 인덱스 이름
        pipeline_name: 사용할 검색 파이프라인 이름
        k: 반환할 결과 수
        
    Returns:
        SearchResult 객체 리스트
    """
    if _os_client is None:
        raise RuntimeError("OpenSearch client not initialized")
    
    # 하이브리드 쿼리 구성
    hybrid_query = {
        "size": k,
        "query": {
            "hybrid": {
                "queries": [
                    # BM25 키워드 검색 부분
                    {
                        "match": {
                            "content": {
                                "query": query_text
                            }
                        }
                    },
                    # 벡터 검색 부분
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": k
                            }
                        }
                    }
                ]
            }
        }
    }
    
    try:
        # 검색 파이프라인을 사용하여 검색 수행
        response = _os_client.search(
            body=hybrid_query,
            index=index_name,
            params={"search_pipeline": pipeline_name}
        )
        
        # 결과를 SearchResult 객체 리스트로 변환
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            score = hit["_score"]
            
            # 메타데이터 구성
            metadata = SearchResultMetadata(
                chunk_index=source.get("chunk_index"),
                doc_id=source.get("doc_id", ""),
                doc_name=source.get("doc_name", source.get("original_file_name", "")),
                original_collection=source.get("original_collection", ""),
                source=source.get("source", "")
            )
            
            # SearchResult 객체 생성
            result = SearchResult(
                page_content=source.get("content", ""),
                metadata=metadata,
                score=score
            )
            
            results.append(result)
        
        logger.debug(f"Found {len(results)} hybrid matches in index '{index_name}'")
        return results
        
    except Exception as e:
        logger.error(f"하이브리드 검색 오류: {str(e)}", exc_info=True)
        # 오류 발생 시 빈 결과 반환
        return []


def fetch_documents(index_name: str):
    if _os_client is None:
        raise RuntimeError("Vector store not initialized")
    return _os_client.search(index=index_name, body={"query": {"match_all": {}}})

# ──────────────────────────────────────────────────────────────
# Alias for backward‑compat import
# ──────────────────────────────────────────────────────────────
# 다른 모듈에서 `from app.models.vector_store import vector_store` 로 불러왔던 기존 코드 호환용.
vector_store = sys.modules[__name__]