"""OpenSearch 인덱스 이름·매핑 정의.

- **INDEX_MAP** : 업로드 API에서 사용하는 컬렉션 키 ↔ 실제 인덱스명
- **MASTER_INDEX** : 모든 문서가 중복 저장되는 통합 인덱스
- **index_mappings** : 인덱스 생성에 사용되는 매핑(dict) 집합

"""
from typing import Dict, Any

# ──────────────────────────────────────────────────────────────
# 인덱스 이름 정의
# ──────────────────────────────────────────────────────────────

INDEX_MAP: Dict[str, str] = {
    "strategy_documents": "strategy_documents",
    "compliance_documents": "compliance_documents",
    "operation_documents": "operation_documents",
    "it_security_documents": "it_security_documents",
    "organization_documents": "organization_documents",
    "stt_texts": "stt_texts"
}

MASTER_INDEX: str = "master_documents"

# ──────────────────────────────────────────────────────────────
# 공통 매핑 프로퍼티 (field definitions)
# ──────────────────────────────────────────────────────────────

BASE_PROPERTIES: Dict[str, Any] = {
    "content": {"type": "text", "analyzer": "korean_nori"},
    "embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
            "name": "hnsw",
            "space_type": "cosinesimil",
            "parameters": {"ef_construction": 128, "m": 24},
        },
    },
    "source": {"type": "keyword"},
    "doc_id": {"type": "keyword"},                 # 문서 고유 ID
    "chunk_index": {"type": "integer"},            # 청크의 인덱스
    "original_file_name": {"type": "keyword"},    # 원본 파일명
    "doc_name": {"type": "keyword"},              # 문서 이름 (PDF 파일명 등)
    "original_collection": {"type": "keyword"},   # 원본 컬렉션(인덱스) 이름
    "upload_timestamp": {"type": "date"}           # 업로드 시간
}


def _build_mapping() -> Dict[str, Any]:
    """단일 인덱스 매핑(settings + mappings) 반환."""
    return {
        "settings": {
            "index": {"knn": True},
            "analysis": {
                "analyzer": {
                    "korean_nori": {
                        "type": "nori",
                        "decompound_mode": "mixed",
                    }
                }
            },
        },
        "mappings": {"properties": BASE_PROPERTIES},
    }


# 모든 인덱스에 동일 매핑 적용
index_mappings: Dict[str, Dict[str, Any]] = {
    **{name: _build_mapping() for name in INDEX_MAP.values()},
    MASTER_INDEX: _build_mapping(),
}
