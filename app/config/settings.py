"""애플리케이션 전역 설정 (Pydantic Settings)."""
from functools import lru_cache
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # OpenSearch 접속 정보
    opensearch_host: str = os.environ.get("OPENSEARCH_HOST", "opensearch")
    opensearch_port: int = os.environ.get("OPENSEARCH_PORT", 9200)
    opensearch_scheme: str = os.environ.get("OPENSEARCH_SCHEME", "http")
    opensearch_username: str = os.environ.get("OPENSEARCH_USERNAME", "admin")
    opensearch_password: str = os.environ.get("OPENSEARCH_PASSWORD", "FisaTeam!5")
    opensearch_verify_certs: bool = os.environ.get("OPENSEARCH_VERIFY_CERTS", False)
    
    # 인덱스 설정
    master_index: str = os.environ.get("MASTER_INDEX", "master_documents")
    
    # 임베딩 모델
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")
    
    # 리랭킹 모델
    reranker_model: str = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()