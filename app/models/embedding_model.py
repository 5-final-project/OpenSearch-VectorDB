"""jina-embeddings-v3 임베딩 모델 래퍼."""
import torch
import numpy as np
import logging
from functools import lru_cache
from typing import List, Optional, Union
from transformers import AutoModel
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _load_device():
    """GPU 사용 가능 여부에 따라 기기 선택"""
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"CUDA 가용 - GPU 사용: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
        logger.info("MPS 가용 - Apple Silicon GPU 사용")
    else:   
        device = 'cpu'
        logger.info("GPU 없음 - CPU 사용")
    return device


class JinaEmbeddings:
    """jinaai/jina-embeddings-v3 모델을 사용하는 임베딩 클래스"""
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3", task: str = "retrieval.passage"):
        """Jina Embeddings 초기화
        
        Args:
            model_name: 사용할 모델 이름
            task: 임베딩 태스크 ('retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching')
        """
        self.model_name = model_name
        self.device = _load_device()
        self.task = task
        logger.info(f"Jina Embeddings 초기화 시작: {model_name}, 태스크: {task}")
        
        try:
            # Transformers를 사용하여 모델 로드
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.model.to(self.device)
            logger.info(f"Jina Embeddings 모델 로드 완료: {model_name}")
        except Exception as e:
            logger.error(f"Jina Embeddings 모델 로드 오류: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """1개 이상의 문서를 임베딩
        
        Args:
            texts: 임베딩할 문서 목록
            
        Returns:
            임베딩 벡터 목록
        """
        if not texts:
            return []
        
        # Passage 태스크를 사용하여 문서 임베딩 (검색 대상)
        try:
            embeddings = self.model.encode(texts, task=self.task)
            return embeddings.tolist()  # numpy 배열을 파이썬 리스트로 변환
        except Exception as e:
            logger.error(f"문서 임베딩 오류: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """1개의 문서를 임베딩
        
        Args:
            text: 임베딩할 쿼리 문자열
            
        Returns:
            임베딩 벡터
        """
        # 쿼리 태스크를 사용하여 쿼리 임베딩 (검색 쿼리)
        query_task = "retrieval.query"  # 쿼리 태스크는 항상 retrieval.query 사용
        try:
            embedding = self.model.encode([text], task=query_task)[0]
            return embedding.tolist()  # numpy 배열을 파이썬 리스트로 변환
        except Exception as e:
            logger.error(f"쿼리 임베딩 오류: {str(e)}")
            raise


def _load_embeddings():
    """Jina Embeddings 모델 로드"""
    settings = get_settings()
    
    # 설정에서 모델 이름을 가져오지만 기본값으로 jinaai/jina-embeddings-v3 사용
    model_name = settings.embedding_model
    
    # 임베딩 태스크 설정 (기본값: passage 임베딩)
    embedding_task = "retrieval.passage"
    
    return JinaEmbeddings(
        model_name=model_name,
        task=embedding_task
    )

# 싱글턴 임베딩 모델 인스턴스
embedding_model = _load_embeddings()