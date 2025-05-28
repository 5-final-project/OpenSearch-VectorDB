from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
from app.services.document_service import DocumentService
from app.models.document_model import UploadResponse, MultiUploadResponse, STTUploadRequest, STTUploadResponse, UploadWithoutS3Response

print("--- Loading document_router.py ---", flush=True)

router = APIRouter()
doc_service = DocumentService()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    index_name: str = Form(..., description="저장할 인덱스 이름"),
    file: UploadFile = File(..., description="업로드할 PDF 파일")
):
    """
PDF 파일을 업로드하고 지정한 인덱스에 저장합니다.
    
    [요청 인수]
    - index_name (str): 문서를 저장할 인덱스 이름
    - file (UploadFile): 업로드할 PDF 파일
    
    [응답]
    - index (str): 저장된 인덱스 이름
    - chunks (int): 생성된 청크 수

    [인덱스]
    - strategy_documents: 경영/사업 전략 관련
    - compliance_documents: 준법/법규 대응
    - operation_documents: 실무 절차
    - it_security_documents: IT 보안/사고 대응
    - organization_documents: 조직/인사 변경
    - stt_texts: STT 텍스트
    
PDF 파일은 텍스트 추출 후 의미 단위로 분할되어 벡터 저장소에 저장됩니다.
각 청크는 임베딩 모델을 통해 벡터화되어 검색 가능한 형태로 저장됩니다.
    """
    print(f"--- Received request for /upload. index_name: {index_name}, filename: {file.filename} ---", flush=True)
    try:
        result = await doc_service.process_upload(file, index_name)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-multiple", response_model=MultiUploadResponse)
async def upload_multiple_documents(
    index_name: str = Form(..., description="저장할 인덱스 이름"),
    files: List[UploadFile] = File(..., description="업로드할 여러 PDF 파일")
):
    """
여러 PDF 파일을 한 번에 업로드하고 지정한 인덱스에 저장합니다.
    
    [요청 인수]
    - index_name (str): 문서를 저장할 인덱스 이름
    - files (List[UploadFile]): 업로드할 여러 PDF 파일 목록
    
    [응답]
    - index (str): 저장된 인덱스 이름
    - total_chunks (int): 전체 생성된 청크 수
    - files (List[FileUploadResult]): 각 파일별 업로드 결과
      - filename (str): 파일 이름
      - chunks (int): 해당 파일에서 생성된 청크 수
      - success (bool): 업로드 성공 여부
      - error (str, 선택): 실패한 경우 오류 메시지
    
    [인덱스]
    - strategy_documents: 경영/사업 전략 관련
    - compliance_documents: 준법/법규 대응
    - operation_documents: 실무 절차
    - it_security_documents: IT 보안/사고 대응
    - organization_documents: 조직/인사 변경
    - stt_texts: STT 텍스트
    
각 PDF 파일은 병렬로 처리되어 텍스트 추출 및 청킹 과정을 거쳐 벡터 저장소에 저장됩니다.
일부 파일이 실패하더라도 나머지 파일은 계속 처리됩니다.
    """
    print(f"--- Received request for /upload-multiple. index_name: {index_name}, file count: {len(files)} ---", flush=True)
    try:
        result = await doc_service.process_multiple_uploads(files, index_name)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-without-s3", response_model=UploadWithoutS3Response)
async def upload_document_without_s3(
    index_name: str = Form(default="reports", description="저장할 인덱스 이름"),
    file: UploadFile = File(..., description="업로드할 PDF 파일")
):
    """
PDF 파일을 업로드하고 S3 저장 및 요약 없이 벡터 DB에만 저장합니다.
    
    [요청 인수]
    - index_name (str): 문서를 저장할 인덱스 이름
    - file (UploadFile): 업로드할 PDF 파일
    
    [응답]
    - index (str): 저장된 인덱스 이름
    - chunks (int): 생성된 청크 수
    - doc_id (str): 생성된 문서 ID (UUID)

    [인덱스]
    - strategy_documents: 경영/사업 전략 관련
    - compliance_documents: 준법/법규 대응
    - operation_documents: 실무 절차
    - it_security_documents: IT 보안/사고 대응
    - organization_documents: 조직/인사 변경
    - stt_texts: STT 텍스트
    
PDF 파일은 텍스트 추출 후 의미 단위로 분할되어 벡터 저장소에 저장됩니다.
각 청크는 임베딩 모델을 통해 벡터화되어 검색 가능한 형태로 저장됩니다.
기존 업로드와 달리 S3에 원본 파일을 저장하지 않으며, 문서 요약 기능도 수행하지 않습니다.
"""
    print(f"--- Received request for /upload-without-s3. index_name: {index_name}, filename: {file.filename} ---", flush=True)
    try:
        result = await doc_service.process_upload_without_s3(file, index_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-stt-text", response_model=STTUploadResponse)
async def upload_stt_text(request: STTUploadRequest):
    """
STT(Speech-to-Text) 텍스트를 업로드하고 지정한 인덱스에 저장합니다.
    
    [요청 인수]
    - text (str): 업로드할 STT 텍스트 내용 
    - index_name (str): 저장할 인덱스 이름 (기본값: "stt_texts")
    - title (str, 선택): 문서 제목
    - meeting_attendees (List[str], 선택): 회의 참석자 목록
    - writer (str, 선택): 작성자 이름
    
    [응답]
    - index (str): 저장된 인덱스 이름
    - chunks (int): 생성된 청크 수
    - doc_id (str): 생성된 문서 ID (UUID)
    
SemanticChunker를 사용하여 텍스트를 의미 단위로 분할하고, 유사한 부분들을 하나의 청크로 그룹화합니다.
문서 ID는 서버에서 자동으로 생성되며, 메타데이터(참석자, 작성자 등)와 함께 S3와 벡터 저장소에 저장됩니다.
기존 PDF 업로드와 달리 텍스트를 직접 처리하므로 Docling 파서를 사용하지 않습니다.
    """
    print(f"--- Received request for /upload-stt-text. index_name: {request.index_name}, text length: {len(request.text)} ---", flush=True)
    try:
        result = await doc_service.process_stt_text(request)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
