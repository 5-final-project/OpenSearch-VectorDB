"""MySQL 데이터베이스 연결 설정"""
import mysql.connector
from mysql.connector import Error
import logging
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                database=settings.mysql_db
            )
            logger.info("MySQL 데이터베이스 연결 성공")
        except Error as e:
            logger.error(f"MySQL 연결 오류: {e}")

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL 연결 종료")

    def _ensure_connection(self):
        if self.connection is None or not self.connection.is_connected():
            self.connect()

    def _get_cursor(self, dictionary: bool = False):
        self._ensure_connection()
        return self.connection.cursor(dictionary=dictionary, buffered=True)

# 데이터베이스 관리자 인스턴스 생성
db_manager = DatabaseManager()

# 데이터베이스 관리자 가져오기
def get_db_manager():
    return db_manager
