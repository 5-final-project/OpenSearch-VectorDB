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
    
    # MySQL 데이터베이스 설정
    mysql_host: str = os.environ.get("MYSQL_HOST", "118.67.131.22")
    mysql_port: int = int(os.environ.get("MYSQL_PORT", 3306))
    mysql_user: str = os.environ.get("MYSQL_USER", "fisaai")
    mysql_password: str = os.environ.get("MYSQL_PASSWORD", "Woorifisa!4")
    mysql_db: str = os.environ.get("MYSQL_DB", "ai_team_5")
    
    # LLM API 설정
    llm_api_url: str = os.environ.get("LLM_API_URL", "https://qwen3.ap.loclx.io/api/generate")
    llm_max_tokens: int = int(os.environ.get("LLM_MAX_TOKENS", 32768))
    
    # AWS S3 설정
    aws_access_key: str = os.environ.get("AWS_ACCESS_KEY")
    aws_secret_key: str = os.environ.get("AWS_SECRET_KEY")
    bucket_name: str = os.environ.get("BUCKET_NAME")
    aws_default_region: str = os.environ.get("AWS_DEFAULT_REGION")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()