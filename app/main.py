"""FastAPI 진입점 및 전역 초기화 모듈."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import get_settings
from app.routers.document_router import router as document_router
from app.routers.search_router import router as search_router
from app.models.vector_store import init_vector_store


def create_app() -> FastAPI:
    settings = get_settings()  # .env 로부터 환경 변수 로드

    app = FastAPI(title="Vector DB management Server", version="0.3.0")

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
