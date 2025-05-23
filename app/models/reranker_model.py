"""하이브리드 검색 결과 재정렬을 위한 Cross-Encoder 모델"""
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing import List, Dict, Tuple, Any
import logging
from app.models.search_model import SearchResult
from app.config.settings import get_settings


logger = logging.getLogger(__name__)

# 크로스 인코더 모델 및 리랭커 전역 인스턴스
_cross_encoder = None
_reranker = None

def init_reranker():
    settings = get_settings()
    """크로스 인코더 모델 초기화"""
    global _cross_encoder, _reranker
    try:
        logger.info("크로스 인코더 모델 초기화 시작...")
        
        # HuggingFaceCrossEncoder 모델 생성
        _cross_encoder = HuggingFaceCrossEncoder(
            model_name=settings.reranker_model
        )
        
        # CrossEncoderReranker 생성 (원할한 수의 문서 반환을 위해 큰 값 설정)
        _reranker = CrossEncoderReranker(model=_cross_encoder, top_n=100)
        logger.info("크로스 인코더 모델 초기화 완료")
        
    except Exception as e:
        logger.error(f"크로스 인코더 모델 초기화 실패: {str(e)}", exc_info=True)
        # 오류가 발생해도 서비스는 계속 실행되도록 함
        _cross_encoder = None
        _reranker = None

def rerank_results(query: str, docs: List[SearchResult], top_k: int = None) -> List[SearchResult]:
    """
    검색 결과를 크로스 인코더로 재정렬
    
    Args:
        query: 사용자 검색 쿼리
        docs: 재정렬할 SearchResult 객체 목록
        top_k: 반환할 최대 결과 수 (None이면 모든 결과 반환)
        
    Returns:
        점수 순으로 재정렬된 SearchResult 목록
    """
    # 완전히 새로운 방식으로 구현
    # 모델이 없으면 원래 점수로 정렬해서 반환
    if _reranker is None or _cross_encoder is None:
        logger.warning("크로스 인코더 모델이 초기화되지 않았습니다. 점수 기준으로만 정렬합니다.")
        # 그냥 원래 점수로 정렬해서 반환
        sorted_docs = sorted(docs, key=lambda doc: doc.score, reverse=True)
        if top_k:
            return sorted_docs[:top_k]
        return sorted_docs
    
    try:
        # 1. 직접 크로스 인코더 사용하여 점수 계산
        logger.info(f"크로스 인코더 재정렬 시작: {len(docs)} 개 문서")
        
        # 새 배열에 문서를 저장하여 인덱스 추적
        scored_docs = []
        
        # 문서마다 크로스 인코더로 점수 계산
        if _cross_encoder is not None:
            for i, doc in enumerate(docs):
                try:
                    # 쿼리와 문서 내용으로 점수 계산
                    pair = [[query, doc.page_content]]
                    score = _cross_encoder.score(pair)[0]  # score 메서드 사용
                    
                    # 새로운 문서 객체 생성
                    scored_doc = SearchResult(
                        page_content=doc.page_content,
                        metadata=doc.metadata,
                        score=score  # 새 점수로 바꾸기
                    )
                    scored_docs.append(scored_doc)
                except Exception as e:
                    logger.warning(f"문서 {i} 점수 계산 중 오류: {str(e)}")
                    # 오류 발생 시 원본 문서 사용
                    scored_docs.append(doc)
            
            # 점수 기준으로 정렬
            scored_docs.sort(key=lambda d: d.score, reverse=True)
            logger.info(f"크로스 인코더 재정렬 완료: {len(scored_docs)} 개 결과")
        else:
            # 크로스 인코더가 없으면 원본 독서 사용
            scored_docs = sorted(docs, key=lambda d: d.score, reverse=True)
            logger.info("크로스 인코더 없음 - 원본 점수로 정렬")
        
        # 3. top_k가 지정되었다면 결과 개수 제한
        if top_k and len(scored_docs) > top_k:
            return scored_docs[:top_k]
        
        return scored_docs
        
    except Exception as e:
        logger.error(f"재정렬 중 오류 발생: {str(e)}", exc_info=True)
        # 오류 발생 시 원래 순서와 점수 유지
        if top_k:
            return sorted(docs, key=lambda doc: doc.score, reverse=True)[:top_k]
        return sorted(docs, key=lambda doc: doc.score, reverse=True)

# 모듈 로드 시 모델 초기화
init_reranker()
