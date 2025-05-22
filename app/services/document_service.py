""" PDF 업로드 → 청크 분할 → VectorStore 저장. """
# import uuid # 이미 불필요
# from typing import List # 이미 불필요

from fastapi import UploadFile # HTTPException은 사용자가 이전에 제거함
import logging
import uuid
import tempfile
import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from langchain_docling import DoclingLoader
from kiwipiepy import Kiwi # Kiwi import 확인
from langchain_core.documents import Document # LangChain Document 임포트
from app.config.opensearch_config import INDEX_MAP # MASTER_INDEX는 settings에서 가져옴
from app.models.document_model import UploadResponse, MultiUploadResponse, FileUploadResult
from app.models import vector_store as vector_store_module # VectorStore 클래스 대신 vector_store 모듈을 임포트
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
vector_store = vector_store_module # DocumentService에서 사용할 vector_store가 임포트한 모듈을 가리키도록 함

class DocumentService:
    async def process_upload(self, file: UploadFile, index_name: str) -> UploadResponse:
        """단일 파일 업로드 처리"""
        return await self._process_single_file(file, index_name)
        
    async def process_multiple_uploads(self, files: List[UploadFile], index_name: str) -> MultiUploadResponse:
        """여러 파일 업로드 처리"""
        logger.info(f"Processing {len(files)} files for index {index_name}")
        
        if index_name not in INDEX_MAP:
            logger.error(f"Invalid index_name: {index_name}. Must be one of {list(INDEX_MAP.keys())}")
            raise ValueError(f"잘못된 인덱스 이름입니다: {index_name}")
        
        results = []
        total_chunks = 0
        
        for file in files:
            try:
                logger.info(f"Processing file: {file.filename}")
                result = await self._process_single_file(file, index_name)
                results.append(FileUploadResult(
                    filename=file.filename,
                    chunks=result.chunks,
                    success=True
                ))
                total_chunks += result.chunks
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
                results.append(FileUploadResult(
                    filename=file.filename,
                    chunks=0,
                    success=False,
                    error=str(e)
                ))
        
        return MultiUploadResponse(
            index=index_name,
            total_chunks=total_chunks,
            files=results
        )
    
    async def _process_single_file(self, file: UploadFile, index_name: str) -> UploadResponse:
        logger.info(f"Starting Stage 2: Processing file {file.filename} for index {index_name}")

        if index_name not in INDEX_MAP:
            logger.error(f"Invalid index_name: {index_name}. Must be one of {list(INDEX_MAP.keys())}")
            raise ValueError(f"잘못된 인덱스 이름입니다: {index_name}")
        
        main_doc_id = str(uuid.uuid4()) # 파일 전체에 대한 고유 ID
        documents_to_store: List[Document] = []

        temp_file_path = None
        try:
            # 임시 파일 생성 및 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_file_path = tmp_file.name
            logger.info(f"Uploaded file '{file.filename}' saved temporarily to '{temp_file_path}'.")

            # DoclingLoader를 사용하여 문서 로드
            loader = DoclingLoader(temp_file_path)
            loaded_documents_from_docling = loader.load() # List[langchain_core.documents.Document]
            
            if not loaded_documents_from_docling:
                logger.warning(f"No documents were loaded from '{file.filename}' by DoclingLoader.")
                return UploadResponse(index=index_name, chunks=0)

            logger.info(f"DoclingLoader loaded {len(loaded_documents_from_docling)} sections from '{file.filename}'.")

            # Kiwi 초기화 (kiwi.space 전처리용)
            try:
                kiwi = Kiwi()
                logger.info("Kiwi initialized for kiwi.space() preprocessing.")
            except Exception as e:
                logger.error(f"Failed to initialize Kiwi: {e}", exc_info=True)
                # Kiwi 초기화 실패 시 오류 발생 또는 다른 처리 필요
                raise ValueError(f"Kiwi 초기화 중 오류 발생: {str(e)}")

            # 각 page_content를 하나의 청크로 처리하고 kiwi.space() 적용
            for i, doc_from_docling in enumerate(loaded_documents_from_docling):
                original_page_content = doc_from_docling.page_content

                if not original_page_content or not original_page_content.strip():
                    logger.warning(f"Skipping empty page_content from DoclingLoader (pre-chunk index {i}) for file '{file.filename}'.")
                    continue
                
                # kiwi.space()로 전처리
                try:
                    processed_chunk_text = kiwi.space(original_page_content)
                except Exception as e:
                    logger.error(f"Error during kiwi.space() for pre-chunk {i} from '{file.filename}': {e}", exc_info=True)
                    processed_chunk_text = original_page_content # 오류 시 원본 사용 (선택적)
                    logger.warning(f"Using original content for pre-chunk {i} due to kiwi.space() error.")

                if not processed_chunk_text.strip():
                    logger.warning(f"Skipping pre-chunk {i} for '{file.filename}' as it became empty after kiwi.space().")
                    continue

                # 메타데이터 준비 (기존 DoclingLoader 메타데이터 복사)
                final_metadata = doc_from_docling.metadata.copy() if doc_from_docling.metadata else {}

                # dl_meta 필드 제거 (사용자 요청)
                final_metadata.pop("dl_meta", None)

                # 현재 시간 가져오기 (ISO 형식)
                current_time = datetime.now(timezone.utc).isoformat()

                # 추가 메타데이터 설정
                final_metadata["doc_id"] = main_doc_id
                final_metadata["chunk_id"] = f"{main_doc_id}_{i}"  # chunk_id 추가
                final_metadata["chunk_index"] = i  # 청크 인덱스 추가
                final_metadata["original_file_name"] = file.filename
                final_metadata["doc_name"] = file.filename  # 문서 이름으로 파일 이름 사용
                final_metadata["original_collection"] = index_name  # 원본 커렉션 이름 저장
                final_metadata["upload_timestamp"] = current_time  # 업로드 시간 추가
                final_metadata["source"] = file.filename  # source에도 파일명 사용

                document_for_opensearch = Document(
                    page_content=processed_chunk_text,
                    metadata=final_metadata
                )
                documents_to_store.append(document_for_opensearch)
            
            if not documents_to_store:
                logger.warning(f"No processable chunks found for '{file.filename}' after DoclingLoader and kiwi.space() processing.")
                return UploadResponse(index=index_name, chunks=0)

        except ValueError as ve:
            logger.error(f"Validation error during processing {file.filename}: {ve}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
            raise ValueError(f"파일 처리 중 오류 발생: {str(e)}")
        finally:
            # 임시 파일 삭제
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Temporary file '{temp_file_path}' deleted.")
                except OSError as e:
                    logger.error(f"Error deleting temporary file '{temp_file_path}': {e}", exc_info=True)

        # OpenSearch에 문서 추가 (이 부분은 기존 로직과 유사하게 작동)
        logger.info(f"Adding {len(documents_to_store)} processed chunks from {file.filename} to OpenSearch index '{index_name}'.")
        
        try:
            # 1. 지정된 인덱스에 문서 추가
            added_ids_specific_index = await vector_store.add_documents(index_name=index_name, docs=documents_to_store)
            logger.info(f"Successfully added {len(added_ids_specific_index)} chunks to specific index '{index_name}' for file: {file.filename}. Document ID: {main_doc_id}")

            # 2. 지정된 인덱스가 마스터 인덱스와 다르고, documents_to_store가 비어있지 않은 경우 마스터 인덱스에도 추가
            if index_name != settings.master_index and documents_to_store:
                logger.info(f"Also adding {len(documents_to_store)} chunks to master index '{settings.master_index}' for file: {file.filename}.")
                await vector_store.add_documents(index_name=settings.master_index, docs=documents_to_store)
                logger.info(f"Successfully added {len(documents_to_store)} chunks to master index '{settings.master_index}' for file: {file.filename}.")
            
            total_added_count = len(added_ids_specific_index)

            return UploadResponse(
                index=index_name, 
                chunks=total_added_count
            )
        except Exception as e:
            logger.error(f"Error adding documents to OpenSearch for file {file.filename}: {e}", exc_info=True)
            # 이미 ValueError로 변환되었을 수 있으므로, 여기서는 ValueError를 다시 발생시키지 않도록 주의
            # 혹은 특정 예외 유형에 따라 다르게 처리
            if not isinstance(e, ValueError):
                 raise ValueError(f"OpenSearch에 문서 추가 중 오류 발생: {str(e)}")
            else:
                raise e
