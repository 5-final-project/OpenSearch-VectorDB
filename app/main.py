"""FastAPI 진입점 및 전역 초기화 모듈."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from app.config.settings import get_settings
from app.routers.document_router import router as document_router
from app.routers.search_router import router as search_router
from app.models.vector_store import init_vector_store

# 로깅 설정
def setup_logging():
    # 로그 포맷 설정
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 루트 로거 구성
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # 기본 레벨을 INFO로 설정
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(console_handler)
    
    # hierarchical_summarizer 모듈은 DEBUG 레벨로 설정
    summarizer_logger = logging.getLogger("app.utils.hierarchical_summarizer")
    summarizer_logger.setLevel(logging.DEBUG)
    
    # 기타 로거 설정
    logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx 로그 줄이기


def create_app() -> FastAPI:
    # 로깅 설정 초기화
    setup_logging()
    
    settings = get_settings()  # .env 로부터 환경 변수 로드

    app = FastAPI(title="Document Management Server", version="0.3.0")

    # CORS 설정 (필요 시)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OpenSearch VectorStore 초기화
    init_vector_store(settings=settings)

    # 라우터 등록
    app.include_router(document_router, prefix="/documents", tags=["Documents"])
    app.include_router(search_router, prefix="/search", tags=["Search"])

    return app


app = create_app()
