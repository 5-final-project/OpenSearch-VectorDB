"""애플리케이션 전역 설정 (Pydantic Settings)."""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenSearch 접속 정보
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_scheme: str = "http"           # http | https
    opensearch_username: str = "admin"
    opensearch_password: str = "FisaTeam!5"
    opensearch_verify_certs: bool = False       # 개발용 false

    # 인덱스 설정
    master_index: str = "master_documents"

    # 임베딩 모델
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"

    # 리랭킹 모델
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # 청크 파라미터
    chunk_size: int = Field(default=1000, description="텍스트 분할 시 목표 청크 크기 (문자 수 기준)")
    chunk_overlap: int = Field(default=200, description="텍스트 분할 시 청크 간 중첩 크기 (문자 수 기준)")

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()