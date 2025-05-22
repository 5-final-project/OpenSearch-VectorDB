""" PDF 파싱 유틸리티 (Langchain DoclingLoader 기반). """
import tempfile
import os
from io import BytesIO # BytesIO는 이제 직접 사용하지 않을 수 있음
from typing import List

from langchain_docling import DoclingLoader # 새로운 로더
from langchain_core.documents import Document # Document 타입 힌트용

# 기존 DoclingPdfParser 관련 코드는 제거
# from docling_parse.pdf_parser import DoclingPdfParser
# from docling_core.types.doc.page import TextCellUnit
# _pdf_parser = DoclingPdfParser()


def parse_pdf(content: bytes) -> str:
    """PDF 바이너리(content)를 Langchain DoclingLoader를 사용하여 파싱하고, 
    페이지별 텍스트를 병합하여 단일 문자열로 반환합니다.

    1. 바이트 데이터를 임시 파일에 저장합니다.
    2. `DoclingLoader`로 임시 PDF 파일을 로드합니다.
    3. 로드된 `Document` 객체들의 `page_content`를 합칩니다.
    4. 페이지 경계마다 줄바꿈(`\n`)을 추가합니다.
    5. 임시 파일을 삭제합니다.
    """
    temp_file_path = None
    try:
        # 임시 파일 생성 (NamedTemporaryFile은 자동 삭제되므로, 수동 삭제를 위해 delete=False)
        # 'with' 문을 사용하여 파일 핸들러가 자동으로 닫히도록 합니다.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        if temp_file_path:
            # DoclingLoader 인스턴스 생성 및 문서 로드
            loader = DoclingLoader(temp_file_path)
            documents: List[Document] = loader.load()

            page_texts: List[str] = []
            for doc in documents:
                page_texts.append(doc.page_content)
            
            return "\n".join(page_texts)
        else:
            # 이 경우는 temp_file_path가 None일 때 발생 (이론적으로는 with 문 안에서 할당되므로 거의 발생 안함)
            raise Exception("Temporary file creation failed.")

    except Exception as e:
        # 여기서 에러 로깅을 하거나, 호출 측으로 예외를 다시 던질 수 있습니다.
        # print(f"Error parsing PDF with DoclingLoader: {e}") # 디버깅용
        raise e # 에러를 다시 발생시켜 document_service에서 처리하도록 함
    finally:
        # 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
