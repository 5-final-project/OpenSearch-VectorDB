""" PDF 업로드 → 청크 분할 → 요약 → VectorStore 저장. """
# import uuid # 이미 불필요
# from typing import List # 이미 불필요

from fastapi import UploadFile # HTTPException은 사용자가 이전에 제거함
import logging
import uuid
import tempfile
import os
import json
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Callable, Tuple

from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from kiwipiepy import Kiwi # Kiwi import 확인
from langchain_core.documents import Document # LangChain Document 임포트
from langchain_experimental.text_splitter import SemanticChunker
from app.config.opensearch_config import INDEX_MAP # MASTER_INDEX는 settings에서 가져옴
from app.models.document_model import UploadResponse, MultiUploadResponse, FileUploadResult, STTUploadRequest, STTUploadResponse, UploadWithoutS3Response
from app.models import vector_store as vector_store_module # VectorStore 클래스 대신 vector_store 모듈을 임포트
from app.config.settings import get_settings
from app.utils.hierarchical_summarizer import enhanced_summarize_document, call_llm_api
from app.services.summary_service import SummaryService
from app.models.embedding_model import embedding_model
from app.utils.s3_client import S3Client

logger = logging.getLogger(__name__)
settings = get_settings()
vector_store = vector_store_module # DocumentService에서 사용할 vector_store가 임포트한 모듈을 가리키도록 함

class DocumentService:
    def __init__(self):
        self.summary_service = SummaryService()
        # S3 클라이언트 초기화
        self.s3_client = S3Client()
        
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
        
        # 임베딩을 추출하여 저장할 리스트
        chunks_for_summary: List[str] = []
        embeddings_for_summary: List[np.ndarray] = []

        temp_file_path = None
        try:
            # 임시 파일 생성 및 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_file_path = tmp_file.name
            logger.info(f"Uploaded file '{file.filename}' saved temporarily to '{temp_file_path}'.")

            # S3에 PDF 파일 업로드
            s3_upload_success, s3_message = self.s3_client.upload_file(
                file_data=content,
                index_name=index_name,
                file_name=file.filename
            )
            
            if s3_upload_success:
                logger.info(f"S3 업로드 성공: {s3_message}")
            else:
                logger.warning(f"S3 업로드 건너뜀: {s3_message}")
                
            # 파일 처리를 위해 파일 포인터를 처음으로 되돌림
            await file.seek(0)
            
            # PDF 파일 전처리 - 유효하지 않은 유니코드 문자 처리
            try:
                # 임시 클린 PDF 경로 생성
                cleaned_temp_path = temp_file_path + ".cleaned.pdf"
                
                # 오류가 발생할 가능성이 있는 PDF 파일 복사
                import shutil
                shutil.copy2(temp_file_path, cleaned_temp_path)
                logger.info(f"Created a copy of the PDF for processing: {cleaned_temp_path}")
                
                # PDF 타입 확인 및 유효성 검사
                import subprocess
                try:
                    # pdfinfo를 사용하여 PDF 정보 확인 (선택적)
                    result = subprocess.run(['pdfinfo', cleaned_temp_path], capture_output=True, text=True)
                    logger.info(f"PDF validation result: {result.stdout[:200]}...")
                except Exception as pdf_check_e:
                    logger.warning(f"PDF info check failed: {str(pdf_check_e)}. Continuing anyway.")
                
                # DoclingLoader를 사용하여 문서 로드
                logger.info(f"Loading PDF with DoclingLoader: {file.filename}")
                # 가장 기본적인 DoclingLoader 사용
                loader = DoclingLoader(file_path=cleaned_temp_path)
                
                try:
                    loaded_documents_from_docling = loader.load()
                except RuntimeError as re:
                    if "Invalid code point" in str(re):
                        logger.warning("Invalid code point detected, attempting to fix and reload...")
                        # PDF 코드포인트 오류 처리
                        try:
                            # qpdf를 사용하여 PDF 수정 (선택적)
                            repair_result = subprocess.run(['qpdf', '--replace-input', cleaned_temp_path], 
                                                      capture_output=True, text=True)
                            logger.info(f"PDF repair attempt result: {repair_result.stdout}")
                            
                            # 수정된 PDF 다시 로드 - 기본 매개변수만 사용
                            loader = DoclingLoader(file_path=cleaned_temp_path)
                            loaded_documents_from_docling = loader.load()
                        except Exception as repair_e:
                            logger.error(f"PDF repair failed: {str(repair_e)}")
                            raise
                    else:
                        raise
                
                # 임시 클린 PDF 파일 삭제
                try:
                    os.remove(cleaned_temp_path)
                    logger.info(f"Removed temporary cleaned PDF: {cleaned_temp_path}")
                except Exception as clean_e:
                    logger.warning(f"Failed to remove temporary cleaned PDF: {str(clean_e)}")
                
                if not loaded_documents_from_docling:
                    logger.warning(f"No documents were loaded from '{file.filename}' by DoclingLoader.")
                    return UploadResponse(index=index_name, chunks=0, error="PDF에서 문서 내용을 추출할 수 없습니다.")
                
                logger.info(f"DoclingLoader loaded {len(loaded_documents_from_docling)} sections from '{file.filename}'.")
            except Exception as e:
                logger.error(f"Error loading PDF: {str(e)}", exc_info=True)
                return UploadResponse(
                    index=index_name, 
                    chunks=0, 
                    error=f"PDF 파일 처리 중 오류가 발생했습니다: {str(e)[:100]}"
                )


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

                # 요약을 위한 텍스트 저장
                chunks_for_summary.append(processed_chunk_text)
                
                # OpenSearch Document 생성
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
            # 1. 지정된 인덱스에 문서 추가 (임베딩도 함께 반환받음)
            added_ids_specific_index, embeddings_for_summary = await vector_store.add_documents(index_name=index_name, docs=documents_to_store)
            logger.info(f"Successfully added {len(added_ids_specific_index)} chunks to specific index '{index_name}' for file: {file.filename}. Document ID: {main_doc_id}")

            # 2. 지정된 인덱스가 마스터 인덱스와 다르고, documents_to_store가 비어있지 않은 경우 마스터 인덱스에도 추가
            if index_name != settings.master_index and documents_to_store:
                logger.info(f"Also adding {len(documents_to_store)} chunks to master index '{settings.master_index}' for file: {file.filename}.")
                await vector_store.add_documents(index_name=settings.master_index, docs=documents_to_store)
                logger.info(f"Successfully added {len(documents_to_store)} chunks to master index '{settings.master_index}' for file: {file.filename}.")
            
            total_added_count = len(added_ids_specific_index)
            
            # 3. 문서 요약 수행
            if chunks_for_summary:
                try:
                    logger.info(f"Starting document summarization for file: {file.filename}")
                    
                    # add_documents에서 반환받은 임베딩 사용 (임베딩 재계산 필요 없음)
                    logger.info(f"Using {len(embeddings_for_summary)} pre-calculated embeddings for summarization")
                    
                    # numpy 배열로 변환
                    embeddings_for_summary = [np.array(emb) for emb in embeddings_for_summary]
                    
                    if len(chunks_for_summary) == len(embeddings_for_summary):
                        # 비동기 요약 함수 생성
                        async def summarize_fn(text: str) -> str:
                            return await call_llm_api(text)
                        
                        # 계층적 요약 수행 (비동기 호출)
                        logger.info(f"Running hierarchical summarization for {len(chunks_for_summary)} chunks")
                        cluster_summaries, final_summary = await enhanced_summarize_document(
                            chunks=chunks_for_summary,
                            embeddings=embeddings_for_summary,
                            summarize_fn=summarize_fn
                        )
                        
                        # MySQL DB에 요약 정보 저장
                        save_result = await self.summary_service.save_document_summary(
                            doc_id=main_doc_id,
                            filename=file.filename,
                            collection=index_name,
                            summary=final_summary,
                            cluster_summaries=cluster_summaries,
                            chunk_count=len(chunks_for_summary)
                        )
                        
                        if save_result:
                            logger.info(f"Successfully saved summary for file: {file.filename}, doc_id: {main_doc_id}")
                        else:
                            logger.warning(f"Failed to save summary for file: {file.filename}, doc_id: {main_doc_id}")
                    else:
                        logger.warning(
                            f"Mismatch between chunks_for_summary ({len(chunks_for_summary)}) and "
                            f"embeddings_for_summary ({len(embeddings_for_summary)}). Skipping summarization."
                        )
                except Exception as e:
                    logger.error(f"Error during document summarization: {str(e)}", exc_info=True)
                    # 요약 실패는 전체 업로드 실패로 간주하지 않음
            else:
                logger.warning(f"No chunks available for summarization for file: {file.filename}")

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

    async def process_upload_without_s3(self, file: UploadFile, index_name: str) -> UploadWithoutS3Response:
        """S3 저장 및 문서 요약 없이 PDF 파일을 업로드하고 벡터 저장소에만 저장"""
        logger.info(f"Starting upload without S3 for file {file.filename} to index {index_name}")

        if index_name not in INDEX_MAP:
            logger.error(f"Invalid index_name: {index_name}. Must be one of {list(INDEX_MAP.keys())}")
            raise ValueError(f"잘못된 인덱스 이름입니다: {index_name}")
        
        main_doc_id = str(uuid.uuid4())  # 파일 전체에 대한 고유 ID
        documents_to_store: List[Document] = []

        temp_file_path = None
        try:
            # 임시 파일 생성 및 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_file_path = tmp_file.name
            logger.info(f"Uploaded file '{file.filename}' saved temporarily to '{temp_file_path}'.")
            
            # 파일 처리를 위해 파일 포인터를 처음으로 되돌림
            await file.seek(0)
            
            # PDF 파일 전처리 - 유효하지 않은 유니코드 문자 처리
            try:
                # 임시 클린 PDF 경로 생성
                cleaned_temp_path = temp_file_path + ".cleaned.pdf"
                
                # 오류가 발생할 가능성이 있는 PDF 파일 복사
                import shutil
                shutil.copy2(temp_file_path, cleaned_temp_path)
                logger.info(f"Created a copy of the PDF for processing: {cleaned_temp_path}")
                
                # PDF 타입 확인 및 유효성 검사
                import subprocess
                try:
                    # pdfinfo를 사용하여 PDF 정보 확인 (선택적)
                    result = subprocess.run(['pdfinfo', cleaned_temp_path], capture_output=True, text=True)
                    logger.info(f"PDF validation result: {result.stdout[:200]}...")
                except Exception as pdf_check_e:
                    logger.warning(f"PDF info check failed: {str(pdf_check_e)}. Continuing anyway.")
                
                # DoclingLoader를 사용하여 문서 로드
                logger.info(f"Loading PDF with DoclingLoader: {file.filename}")
                # 가장 기본적인 DoclingLoader 사용
                loader = DoclingLoader(file_path=cleaned_temp_path)
                
                try:
                    loaded_documents_from_docling = loader.load()
                except RuntimeError as re:
                    if "Invalid code point" in str(re):
                        logger.warning("Invalid code point detected, attempting to fix and reload...")
                        # PDF 코드포인트 오류 처리
                        try:
                            # qpdf를 사용하여 PDF 수정 (선택적)
                            repair_result = subprocess.run(['qpdf', '--replace-input', cleaned_temp_path], 
                                                      capture_output=True, text=True)
                            logger.info(f"PDF repair attempt result: {repair_result.stdout}")
                            
                            # 수정된 PDF 다시 로드 - 기본 매개변수만 사용
                            loader = DoclingLoader(file_path=cleaned_temp_path)
                            loaded_documents_from_docling = loader.load()
                        except Exception as repair_e:
                            logger.error(f"PDF repair failed: {str(repair_e)}")
                            raise
                    else:
                        raise
                
                # 임시 클린 PDF 파일 삭제
                try:
                    os.remove(cleaned_temp_path)
                    logger.info(f"Removed temporary cleaned PDF: {cleaned_temp_path}")
                except Exception as clean_e:
                    logger.warning(f"Failed to remove temporary cleaned PDF: {str(clean_e)}")
                
                if not loaded_documents_from_docling:
                    logger.warning(f"No documents were loaded from '{file.filename}' by DoclingLoader.")
                    return UploadWithoutS3Response(index=index_name, chunks=0, doc_id=main_doc_id)
                
                logger.info(f"DoclingLoader loaded {len(loaded_documents_from_docling)} sections from '{file.filename}'.")
            except Exception as e:
                logger.error(f"Error loading PDF: {str(e)}", exc_info=True)
                return UploadWithoutS3Response(
                    index=index_name, 
                    chunks=0, 
                    doc_id=main_doc_id
                )

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
                    processed_chunk_text = original_page_content  # 오류 시 원본 사용 (선택적)
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

                # OpenSearch Document 생성
                document_for_opensearch = Document(
                    page_content=processed_chunk_text,
                    metadata=final_metadata
                )
                documents_to_store.append(document_for_opensearch)
            
            if not documents_to_store:
                logger.warning(f"No processable chunks found for '{file.filename}' after DoclingLoader and kiwi.space() processing.")
                return UploadWithoutS3Response(index=index_name, chunks=0, doc_id=main_doc_id)

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

        # OpenSearch에 문서 추가
        logger.info(f"Adding {len(documents_to_store)} processed chunks from {file.filename} to OpenSearch index '{index_name}'.")
        
        try:
            # 1. 지정된 인덱스에 문서 추가
            added_ids_specific_index, _ = await vector_store.add_documents(index_name=index_name, docs=documents_to_store)
            logger.info(f"Successfully added {len(added_ids_specific_index)} chunks to specific index '{index_name}' for file: {file.filename}. Document ID: {main_doc_id}")

            # 2. 지정된 인덱스가 마스터 인덱스와 다르고, documents_to_store가 비어있지 않은 경우 마스터 인덱스에도 추가
            if index_name != settings.master_index and documents_to_store:
                logger.info(f"Also adding {len(documents_to_store)} chunks to master index '{settings.master_index}' for file: {file.filename}.")
                await vector_store.add_documents(index_name=settings.master_index, docs=documents_to_store)
                logger.info(f"Successfully added {len(documents_to_store)} chunks to master index '{settings.master_index}' for file: {file.filename}.")
            
            total_added_count = len(added_ids_specific_index)
            
            return UploadWithoutS3Response(
                index=index_name, 
                chunks=total_added_count,
                doc_id=main_doc_id
            )
        except Exception as e:
            logger.error(f"Error adding documents to OpenSearch for file {file.filename}: {e}", exc_info=True)
            # 이미 ValueError로 변환되었을 수 있으므로, 여기서는 ValueError를 다시 발생시키지 않도록 주의
            if not isinstance(e, ValueError):
                raise ValueError(f"OpenSearch에 문서 추가 중 오류 발생: {str(e)}")
            else:
                raise e

    async def process_stt_text(self, request: STTUploadRequest) -> STTUploadResponse:
        """STT 텍스트를 처리하여 청크로 분할하고 벡터 저장소에 저장"""
        logger.info(f"Processing STT text for index {request.index_name}")
        
        if request.index_name not in INDEX_MAP:
            logger.error(f"Invalid index_name: {request.index_name}. Must be one of {list(INDEX_MAP.keys())}")
            raise ValueError(f"잘못된 인덱스 이름입니다: {request.index_name}")
        
        # 문서 ID 생성
        doc_id = str(uuid.uuid4())
        logger.info(f"Using document ID: {doc_id}")
        
        # 현재 시간 가져오기 (ISO 형식)
        current_time = datetime.now(timezone.utc).isoformat()
        
        # 문서 제목 설정
        doc_title = request.title if request.title else f"STT Text {current_time}"
        
        # S3 메타데이터 준비
        s3_metadata = {
            "doc_id": doc_id,
            "title": doc_title,
            "upload_timestamp": current_time,
            "source": "stt_text",
            "index_name": request.index_name
        }
        
        # 회의 참석자 추가 (있는 경우)
        if request.meeting_attendees and len(request.meeting_attendees) > 0:
            s3_metadata["meeting_attendees"] = request.meeting_attendees
        
        # 작성자 추가 (있는 경우)
        if request.writer:
            s3_metadata["writer"] = request.writer
        
        # S3에 텍스트 파일 업로드 (메타데이터 포함)
        s3_upload_success, s3_message = self.s3_client.upload_text_as_file(
            text=request.text,
            index_name=request.index_name,
            file_name=doc_title,
            metadata=s3_metadata
        )
        
        if s3_upload_success:
            logger.info(f"S3 텍스트 파일 업로드 성공: {s3_message}")
        else:
            logger.warning(f"S3 텍스트 파일 업로드 건너뜀: {s3_message}")
        
        # SemanticChunker 초기화 (설정된 임베딩 모델 사용)
        try:
            # SemanticChunker는 embedding_model을 직접 받을 수 있으므로, 그대로 전달
            text_splitter = SemanticChunker(embedding_model)
            logger.info(f"Initialized SemanticChunker with embedding model: {settings.embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing SemanticChunker: {e}", exc_info=True)
            raise ValueError(f"SemanticChunker 초기화 오류: {str(e)}")
        
        # 텍스트 처리
        try:
            # 텍스트가 비어있는지 확인
            if not request.text or not request.text.strip():
                logger.error("Empty text provided")
                raise ValueError("빈 텍스트가 제공되었습니다")
            
            # Kiwi 초기화 (한국어 전처리용)
            kiwi = Kiwi()
            
            # kiwi.space()로 텍스트 전처리
            processed_text = kiwi.space(request.text)
            
            # SemanticChunker로 텍스트 분할
            logger.info(f"Splitting text with SemanticChunker, text length: {len(processed_text)}")
            chunks = text_splitter.split_text(processed_text)
            logger.info(f"Text split into {len(chunks)} chunks")
            
            if not chunks:
                logger.warning("No chunks were created from the text")
                return STTUploadResponse(index=request.index_name, chunks=0, doc_id=doc_id)
            
            # Document 객체 생성
            documents_to_store = []
            for i, chunk_text in enumerate(chunks):
                # 메타데이터 준비
                metadata = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{i}",
                    "chunk_index": i,
                    "doc_name": doc_title,
                    "original_file_name": doc_title,
                    "original_collection": request.index_name,
                    "upload_timestamp": current_time,
                    "source": "stt_text",
                    "meeting_attendees": request.meeting_attendees,
                    "writer": request.writer
                }
        
                # Document 객체 생성
                document = Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
                documents_to_store.append(document)
            
            # OpenSearch에 문서 추가
            logger.info(f"Adding {len(documents_to_store)} chunks to OpenSearch index '{request.index_name}'")
            
            # 1. 지정된 인덱스에 문서 추가
            added_ids, _ = await vector_store.add_documents(index_name=request.index_name, docs=documents_to_store)
            logger.info(f"Successfully added {len(added_ids)} chunks to index '{request.index_name}' for doc_id: {doc_id}")
            
            # 2. 마스터 인덱스에도 추가 (지정된 인덱스와 다른 경우)
            if request.index_name != settings.master_index:
                logger.info(f"Also adding chunks to master index '{settings.master_index}'")
                await vector_store.add_documents(index_name=settings.master_index, docs=documents_to_store)
            
            # 요약 기능 제거됨 - STT 텍스트에 대해서는 요약을 수행하지 않음
            
            return STTUploadResponse(
                index=request.index_name,
                chunks=len(added_ids),
                doc_id=doc_id
            )
            
        except ValueError as ve:
            logger.error(f"Validation error during processing STT text: {ve}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error processing STT text: {e}", exc_info=True)
            raise ValueError(f"STT 텍스트 처리 중 오류 발생: {str(e)}")
