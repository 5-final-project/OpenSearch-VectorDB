"""문서 관련 Pydantic 모델."""
from pydantic import BaseModel
from typing import Any


class UploadResponse(BaseModel):
    index: str
    chunks: int


class VisualizationResponse(BaseModel):
    index: str
    documents: Any