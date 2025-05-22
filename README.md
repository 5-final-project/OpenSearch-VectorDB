# OpenSearch 관리 서버 (OpenSearch Management Server)

## 프로젝트 소개

이 프로젝트는 OpenSearch를 기반으로 한 벡터 검색 관리 서버입니다. PDF 문서를 업로드하고 벡터 저장소에 저장한 후, 다양한 검색 방법(벡터 검색, 키워드 검색, 하이브리드 검색)을 제공합니다.

## 주요 기능

- **PDF 문서 업로드**: PDF 파일을 업로드하고 지정한 인덱스에 저장
- **벡터 검색**: 의미적 유사성 기반 검색
- **키워드 검색**: BM25 알고리즘 기반 검색
- **하이브리드 검색**:
  - Native: OpenSearch 내장 하이브리드 검색 기능 사용
  - Reranked: 크로스 인코더 재정렬을 적용한 하이브리드 검색

## 기술 스택

- **Backend**: FastAPI
- **Vector Database**: OpenSearch
- **PDF 처리**: DoclingLoader
- **한국어 처리**: Kiwipiepy
- **임베딩 모델**: LangChain과 Sentence Transformers 사용
- **배포**: Docker, Docker Compose

## 시스템 아키텍처

```
app/
├── config/        # 환경 설정 관련 파일
├── models/        # 데이터 모델 정의
├── routers/       # API 엔드포인트 라우터
├── services/      # 비즈니스 로직 서비스
├── utils/         # 유틸리티 함수
└── main.py        # FastAPI 진입점
```

## API 엔드포인트

### 문서 관리
- `POST /documents/upload`: PDF 파일을 업로드하고 인덱싱

### 검색
- `POST /search/vector`: 벡터 검색 수행
- `POST /search/keyword`: 키워드 검색 수행
- `POST /search/hybrid-native`: OpenSearch 내장 하이브리드 검색
- `POST /search/hybrid-reranked`: 크로스 인코더 재정렬이 적용된 하이브리드 검색

## 설치 및 실행

### 필수 조건
- Docker 및 Docker Compose 설치
- 환경 변수 설정 (.env 파일)

### Docker Compose로 실행
```bash
docker-compose up -d
```

## 환경 변수 설정

.env 파일에 다음 설정이 필요합니다:
- OpenSearch 연결 정보 (호스트, 포트, 사용자명, 비밀번호)
- 인덱스 설정
- 임베딩 모델 설정

## 개발 환경 설정

```bash
# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload
```

## 문서 처리 과정

1. PDF 업로드 → DoclingLoader로 문서 로드
2. Kiwipiepy로 한국어 텍스트 전처리
3. 청크로 분할 및 메타데이터 추가
4. OpenSearch 벡터 저장소에 저장

## 검색 프로세스

- **벡터 검색**: 쿼리 텍스트를 임베딩하여 벡터 공간에서 유사성 검색
- **키워드 검색**: BM25 알고리즘을 사용한 텍스트 기반 검색
- **하이브리드 검색**: 벡터 검색과 키워드 검색을 결합하여 정확도 향상
- **재정렬 검색**: 크로스 인코더 모델을 사용하여 검색 결과 재정렬
