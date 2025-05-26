"""문서 요약 데이터 관리 서비스"""
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from mysql.connector import Error

from app.config.mysql_config import get_db_manager
from app.models.summary_model import initialize_tables

logger = logging.getLogger(__name__)

class SummaryService:
    """문서 요약 정보를 저장하고 조회하는 서비스"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        # 초기화 시 테이블 생성 확인
        initialize_tables(self.db_manager)
    
    async def save_document_summary(self, 
                                   doc_id: str, 
                                   filename: str, 
                                   collection: str, 
                                   summary: str, 
                                   cluster_summaries: List[str],
                                   chunk_count: int) -> bool:
        """
        문서 요약 정보를 데이터베이스에 저장합니다.
        
        Parameters
        ----------
        doc_id : 문서 UUID
        filename : 원본 파일명
        collection : 저장된 인덱스/컬렉션명
        summary : 문서 최종 요약
        cluster_summaries : 클러스터별 요약 리스트
        chunk_count : 문서 청크 수
        
        Returns
        -------
        bool : 저장 성공 여부
        """
        try:
            # JSON으로 직렬화
            cluster_summaries_json = json.dumps(cluster_summaries, ensure_ascii=False)
            
            # 커서 가져오기
            cursor = self.db_manager._get_cursor()
            
            # 기존 요약이 있는지 확인
            cursor.execute(
                "SELECT doc_id FROM document_summaries WHERE doc_id = %s",
                (doc_id,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # 기존 요약 업데이트
                query = """
                UPDATE document_summaries 
                SET filename = %s, collection = %s, summary = %s, cluster_summaries = %s, chunk_count = %s
                WHERE doc_id = %s
                """
                cursor.execute(query, (
                    filename,
                    collection,
                    summary,
                    cluster_summaries_json,
                    chunk_count,
                    doc_id
                ))
            else:
                # 새 요약 생성
                query = """
                INSERT INTO document_summaries 
                (doc_id, filename, collection, summary, cluster_summaries, chunk_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (
                    doc_id,
                    filename,
                    collection,
                    summary,
                    cluster_summaries_json,
                    chunk_count
                ))
            
            # 변경 사항 커밋
            self.db_manager.connection.commit()
            cursor.close()
            logger.info(f"문서 요약 저장 성공: {doc_id}, {filename}")
            return True
            
        except Error as e:
            logger.error(f"문서 요약 저장 실패 (DB 오류): {str(e)}")
            if hasattr(self.db_manager, 'connection'):
                self.db_manager.connection.rollback()
            return False
        except Exception as e:
            logger.error(f"문서 요약 저장 중 예외 발생: {str(e)}")
            if hasattr(self.db_manager, 'connection'):
                self.db_manager.connection.rollback()
            return False
        finally:
            if 'cursor' in locals() and cursor is not None:
                cursor.close()
            
    async def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        문서 요약 정보를 조회합니다.
        
        Parameters
        ----------
        doc_id : 문서 UUID
        
        Returns
        -------
        Optional[Dict[str, Any]] : 문서 요약 정보 (없으면 None)
        """
        cursor = None
        try:
            # 커서 가져오기
            cursor = self.db_manager._get_cursor(dictionary=True)
            
            # 요약 조회
            cursor.execute(
                "SELECT * FROM document_summaries WHERE doc_id = %s",
                (doc_id,)
            )
            db_summary = cursor.fetchone()
            
            if not db_summary:
                return None
                
            # 클러스터 요약 JSON 파싱
            cluster_summaries = json.loads(db_summary["cluster_summaries"]) if db_summary["cluster_summaries"] else []
                
            # 응답 데이터 구성
            return {
                "doc_id": db_summary["doc_id"],
                "filename": db_summary["filename"],
                "collection": db_summary["collection"],
                "summary": db_summary["summary"],
                "cluster_summaries": cluster_summaries,
                "chunk_count": db_summary["chunk_count"],
                "created_at": db_summary["created_at"],
                "updated_at": db_summary["updated_at"]
            }
            
        except Exception as e:
            logger.error(f"문서 요약 조회 중 오류 발생: {str(e)}")
            return None
        finally:
            if cursor is not None:
                cursor.close()
