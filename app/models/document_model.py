"""문서 관련 Pydantic 모델."""
from pydantic import BaseModel
from typing import Any, List, Dict


class UploadResponse(BaseModel):
    index: str
    chunks: int
    
    
class FileUploadResult(BaseModel):
    filename: str
    chunks: int
    success: bool
    error: str = None


class MultiUploadResponse(BaseModel):
    index: str
    total_chunks: int
    files: List[FileUploadResult]


class VisualizationResponse(BaseModel):
    index: str
    documents: Any