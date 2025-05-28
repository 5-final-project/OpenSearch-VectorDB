"""문서 관련 Pydantic 모델."""
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional


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


class STTUploadRequest(BaseModel):
    """STT 텍스트 업로드 요청 모델"""
    text: str = Field(..., description="업로드할 STT 텍스트")
    index_name: str = Field(default="stt_texts", description="저장할 인덱스 이름")
    title: Optional[str] = Field(None, description="문서 제목")
    meeting_attendees: Optional[List[str]] = Field(default=[], description="회의 참석자 목록")
    writer: Optional[str] = Field(None, description="작성자")

class STTUploadResponse(BaseModel):
    """STT 텍스트 업로드 응답 모델"""
    index: str
    chunks: int
    doc_id: str = Field(..., description="문서 ID")

class UploadWithoutS3Response(BaseModel):
    """S3 저장 없이 PDF 업로드 응답 모델"""
    index: str
    chunks: int
    doc_id: str = Field(..., description="문서 ID")