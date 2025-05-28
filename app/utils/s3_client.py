import boto3
import io
import json
import logging
import sys
import base64
from botocore.exceptions import ClientError
from app.config import settings

# 로거 설정 개선
logger = logging.getLogger("app.utils.s3_client")
logger.setLevel(logging.INFO)

# 콘솔 출력을 위한 핸들러 추가
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class S3Client:
    def __init__(self):
        """AWS S3 클라이언트 초기화"""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.get_settings().aws_access_key,
            aws_secret_access_key=settings.get_settings().aws_secret_key,
            region_name=settings.get_settings().aws_default_region
        )
        self.bucket_name = settings.get_settings().bucket_name
        logger.info(f"S3 클라이언트 초기화 완료: 버킷={self.bucket_name}, 리전={settings.get_settings().aws_default_region}")
        # 터미널에도 출력
        print(f"[S3Client] 초기화 완료: 버킷={self.bucket_name}", flush=True)

    def _encode_metadata_value(self, value):
        """
        메타데이터 값을 ASCII로 인코딩합니다.
        한글이나 다른 유니코드 문자를 Base64로 인코딩하여 ASCII 문자열로 변환합니다.
        
        Args:
            value: 인코딩할 값
        
        Returns:
            인코딩된 ASCII 문자열
        """
        if isinstance(value, str):
            # 값이 ASCII만 포함하는지 확인
            try:
                value.encode('ascii')
                return value  # 이미 ASCII면 그대로 반환
            except UnicodeEncodeError:
                # ASCII가 아닌 문자가 포함된 경우 Base64로 인코딩
                encoded = base64.b64encode(value.encode('utf-8')).decode('ascii')
                return f"base64:{encoded}"
        elif value is None:
            return ""
        else:
            # 문자열이 아닌 경우 JSON으로 직렬화 후 인코딩
            json_str = json.dumps(value)
            try:
                json_str.encode('ascii')
                return f"json:{json_str}"
            except UnicodeEncodeError:
                encoded = base64.b64encode(json_str.encode('utf-8')).decode('ascii')
                return f"base64json:{encoded}"

    def upload_file(self, file_data, index_name, file_name, metadata=None):
        """
        파일 데이터를 S3에 업로드합니다.
        
        Args:
            file_data: 업로드할 파일 데이터
            index_name: 인덱스 이름 (S3 폴더 경로)
            file_name: 파일 이름
            metadata: 추가할 메타데이터 딕셔너리 (선택 사항)
        
        Returns:
            (성공 여부, 메시지)
        """
        try:
            # 파일 경로 생성 (index_name/file_name)
            s3_path = f"{index_name}/{file_name}"
            
            # 작업 시작 로그
            logger.info(f"S3 업로드 시작: 경로={s3_path}, 버킷={self.bucket_name}")
            print(f"[S3Client] 파일 업로드 시작: {s3_path}", flush=True)
            
            # 파일이 이미 존재하는지 확인
            if self.check_file_exists(index_name, file_name):
                logger.info(f"파일이 이미 존재합니다: {s3_path}")
                print(f"[S3Client] 파일이 이미 존재합니다: {s3_path}", flush=True)
                return False, f"파일이 이미 존재합니다: {s3_path}"
            
            # 업로드 파라미터 설정
            put_params = {
                'Bucket': self.bucket_name,
                'Key': s3_path,
                'Body': file_data
            }
            
            # 메타데이터가 있으면 추가
            if metadata:
                # S3는 메타데이터 값으로 ASCII 문자만 허용하므로 변환
                s3_metadata = {}
                for key, value in metadata.items():
                    # 모든 값은 인코딩 함수를 통해 ASCII로 변환
                    s3_metadata[key] = self._encode_metadata_value(value)
                
                put_params['Metadata'] = s3_metadata
                logger.info(f"메타데이터 추가(인코딩 후): {s3_metadata}")
                print(f"[S3Client] 메타데이터 추가: {s3_metadata}", flush=True)
            
            # 파일 업로드
            self.s3_client.put_object(**put_params)
            logger.info(f"파일 업로드 성공: {s3_path}")
            print(f"[S3Client] 파일 업로드 성공: {s3_path}", flush=True)
            return True, s3_path
        except ClientError as e:
            error_msg = f"S3 업로드 오류: {str(e)}"
            logger.error(error_msg)
            print(f"[S3Client] 오류: {error_msg}", flush=True)
            return False, str(e)
    
    def upload_text_as_file(self, text, index_name, file_name, metadata=None):
        """
        텍스트를 파일로 변환하여 S3에 업로드합니다.
        
        Args:
            text: 업로드할 텍스트
            index_name: 인덱스 이름 (S3 폴더 경로)
            file_name: 파일 이름
            metadata: 추가할 메타데이터 딕셔너리 (선택 사항)
        
        Returns:
            (성공 여부, 메시지)
        """
        try:
            # 확장자 확인 및 추가
            if not file_name.endswith('.txt'):
                file_name = f"{file_name}.txt"
            
            logger.info(f"텍스트를 파일로 변환하여 업로드 시작: {file_name}")
            print(f"[S3Client] 텍스트 파일 업로드 시작: {index_name}/{file_name}", flush=True)
            
            # 텍스트를 파일 스트림으로 변환
            text_data = io.BytesIO(text.encode('utf-8'))
            
            # 업로드 수행 (메타데이터 전달)
            return self.upload_file(text_data, index_name, file_name, metadata)
        except Exception as e:
            error_msg = f"텍스트 파일 업로드 오류: {str(e)}"
            logger.error(error_msg)
            print(f"[S3Client] 오류: {error_msg}", flush=True)
            return False, str(e)
    
    def check_file_exists(self, index_name, file_name):
        """파일이 S3에 이미 존재하는지 확인합니다."""
        try:
            s3_path = f"{index_name}/{file_name}"
            logger.info(f"파일 존재 여부 확인: {s3_path}")
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_path)
            logger.info(f"파일이 존재합니다: {s3_path}")
            return True
        except ClientError as e:
            # 파일이 존재하지 않는 경우 404 에러 발생
            if e.response['Error']['Code'] == '404':
                logger.info(f"파일이 존재하지 않습니다: {s3_path}")
                return False
            # 다른 오류인 경우
            logger.error(f"S3 파일 확인 오류: {str(e)}")
            raise e