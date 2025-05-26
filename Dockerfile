# Stage 1: Build stage - 필요한 빌드 도구 설치 및 Python 패키지 빌드
FROM python:3.11-slim AS builder

WORKDIR /opt/build

# kiwipiepy 빌드에 필요한 시스템 패키지 설치 (build-essential, cmake, python-dev)
# curl은 기존에 있었으므로 유지, 필요 없다면 제거 가능
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3.11-dev \
    curl && \
    rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 Python 패키지 설치 (venv 사용)
COPY requirements.txt ./
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Python 패키지 빌드/설치 도구 업그레이드
RUN pip install --upgrade pip setuptools wheel

# PEP 517 빌드 격리 문제 우회를 위해 numpy를 먼저 명시적으로 설치
RUN pip install numpy>=1.20.0

# 그 다음 나머지 requirements.txt 설치 (numpy는 이미 설치되었으므로 건너뜀)
# --no-build-isolation 플래그 추가하여 빌드 격리 비활성화
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt


# Stage 2: Final stage - 빌드된 패키지와 애플리케이션 코드만 포함
FROM python:3.11-slim AS final

WORKDIR /code

# PDF 처리 도구 및 필수 유틸리티 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    qpdf \
    && rm -rf /var/lib/apt/lists/*

# builder 스테이지에서 생성된 가상 환경 복사
COPY --from=builder /opt/venv /opt/venv

# 애플리케이션 소스 복사
COPY . .

# 가상 환경 활성화
ENV PATH="/opt/venv/bin:$PATH"

# uvicorn 실행 시 --reload 옵션은 개발 환경에서는 유용하지만,
# 프로덕션 이미지에서는 제거하는 것이 일반적입니다. 필요에 따라 유지 또는 제거하세요.
# 여기서는 기존 CMD에 --reload가 없었으므로 그대로 유지합니다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
