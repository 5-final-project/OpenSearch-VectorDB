"""
계층적 요약 파이프라인 모듈

입력  : chunks[List[str]], embeddings[List[np.ndarray]]
출력  : (cluster_summaries[List[str]], final_summary[str])
"""

from __future__ import annotations
from typing import List, Tuple, Callable, Dict, Any
import numpy as np
from sklearn.mixture import GaussianMixture
import logging
import httpx
import json
import asyncio
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# ────────────────────────────────
# GMM+BIC로 군집 수 결정
# ────────────────────────────────
def _select_k(vecs: np.ndarray,
              k_min: int = 1,
              k_max: int = 8) -> int:
    n = vecs.shape[0]
    if n <= k_min:
        logger.info(f"청크 수({n})가 k_min({k_min})보다 작거나 같아 k={n}로 설정")
        return n
    k_max = min(k_max, n)
    logger.info(f"GMM+BIC로 최적의 k 찾는 중... k 범위: {k_min}~{k_max}, 청크 수: {n}")
    
    bics = {}
    for k in range(k_min, k_max + 1):
        gm = GaussianMixture(n_components=k, random_state=42).fit(vecs)
        bics[k] = gm.bic(vecs)
        logger.info(f"  k={k}일 때 BIC 점수: {bics[k]:.2f}")
    
    best_k = min(bics, key=bics.get)
    logger.info(f"최적의 군집 수로 k={best_k} 선택됨 (BIC={bics[best_k]:.2f})")
    return best_k


# ────────────────────────────────
# 재귀 요약 (비동기 처리 추가)
# ────────────────────────────────
async def _recursive_summary(texts: List[str],
                       vecs: np.ndarray,
                       summarize_fn: Callable[[str], str],
                       depth: int,
                       max_depth: int) -> str:
    logger.info(f"\n===== 재귀 요약 단계 (깊이: {depth}/{max_depth}) =====")
    logger.info(f"현재 레벨의 텍스트 수: {len(texts)}")
    
    if len(texts) == 1 or depth >= max_depth:
        logger.info(f"텍스트가 하나만 남았거나({len(texts)}), 최대 깊이에 도달({depth}/{max_depth}) -> 재귀 종료")
        return await summarize_fn("\n".join(texts))

    k = _select_k(vecs)
    if k == 1:
        logger.info(f"GMM이 군집을 1개로 나눠 -> 재귀 종료")
        return await summarize_fn("\n".join(texts))

    logger.info(f"{k}개 군집으로 GMM 학습 및 예측 시작...")
    gmm = GaussianMixture(n_components=k, random_state=42).fit(vecs)
    labels = gmm.predict(vecs)
    
    # 군집 분포 확인
    cluster_sizes = []
    for cid in range(k):
        size = np.sum(labels == cid)
        cluster_sizes.append(size)
        logger.info(f"  군집 {cid}: {size}개 텍스트 포함")

    next_texts, next_vecs = [], []
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        cluster_text = "\n".join(texts[i] for i in idx)
        
        logger.info(f"\n----- 군집 {cid} 요약 중 ({len(idx)}개 텍스트) -----")
        summary_preview = cluster_text[:100] + "..." if len(cluster_text) > 100 else cluster_text
        logger.info(f"  요약할 텍스트(일부): {summary_preview}")
        
        summary = await summarize_fn(cluster_text)
        summary_preview = summary[:100] + "..." if len(summary) > 100 else summary
        logger.info(f"  요약 결과: {summary_preview}")
        
        next_texts.append(summary)
        next_vecs.append(vecs[idx].mean(axis=0))  # 평균 임베딩

    logger.info(f"\n----- 다음 단계로 진행: 깊이 {depth} -> {depth+1} -----")
    return await _recursive_summary(next_texts,
                              np.vstack(next_vecs),
                              summarize_fn,
                              depth + 1,
                              max_depth)


# ────────────────────────────────
# 공개 API (비동기 처리 추가)
# ────────────────────────────────
async def summarize_document(
    chunks: List[str],
    embeddings: List[np.ndarray],
    summarize_fn: Callable[[str], str],
    *,
    k_min: int = 2,
    k_max: int = 8,
    max_depth: int = 4
) -> Tuple[List[str], str]:
    """
    계층적 요약 파이프라인.

    Parameters
    ----------
    chunks       : 분할된 텍스트 조각 리스트
    embeddings   : 각 조각의 임베딩 (chunks와 길이 같아야 함)
    summarize_fn : 텍스트를 요약해 주는 LLM 함수 (비동기 함수)
    k_min, k_max : GMM 군집 수 탐색 범위
    max_depth    : 재귀 요약 최대 깊이

    Returns
    -------
    cluster_summaries : 1차 군집별 요약 리스트
    final_summary     : 통합 최종 요약
    """
    logger.info("\n\n===== 계층적 요약 시작 =====")
    logger.info(f"총 {len(chunks)}개 청크, 임베딩 차원: {embeddings[0].shape if embeddings else 'N/A'}")
    
    # 청크 정보 샘플링 출력
    if chunks:
        sample_size = min(3, len(chunks))
        for i in range(sample_size):
            chunk_preview = chunks[i][:100] + "..." if len(chunks[i]) > 100 else chunks[i]
            logger.info(f"\n청크 #{i} 샘플: {chunk_preview}")
    
    if len(chunks) != len(embeddings):
        logger.error(f"chunks({len(chunks)})와 embeddings({len(embeddings)}) 길이가 다름")
        raise ValueError("chunks와 embeddings 길이가 다릅니다.")
    if not chunks:
        logger.error("빈 문서 입력됨")
        raise ValueError("빈 문서입니다.")

    logger.info("임베딩 행렬 생성 중...")
    vecs = np.vstack([np.asarray(v) for v in embeddings])
    logger.info(f"임베딩 행렬 생성 완료. 형태: {vecs.shape}")

    # 1차 군집화
    logger.info("\n===== 1차 군집화 시작 =====")
    k = _select_k(vecs, k_min, k_max)
    if k == 1:
        logger.info("군집이 1개로 나와서 직접 요약 수행")
        all_text = "\n".join(chunks)
        text_preview = all_text[:150] + "..." if len(all_text) > 150 else all_text
        logger.info(f"요약할 텍스트(일부): {text_preview}")
        
        cluster_summaries = [await summarize_fn(all_text)]
        final_summary = cluster_summaries[0]
        
        logger.info(f"최종 요약 결과: {final_summary}")
        logger.info("===== 계층적 요약 완료 =====\n")
        return cluster_summaries, final_summary

    logger.info(f"{k}개 군집으로 GMM 학습 및 예측 시작...")
    gmm = GaussianMixture(n_components=k, random_state=42).fit(vecs)
    labels = gmm.predict(vecs)
    
    # 군집 분포 확인
    cluster_distribution = {}
    for cid in range(k):
        cluster_size = np.sum(labels == cid)
        cluster_distribution[cid] = cluster_size
        logger.info(f"  군집 {cid}: {cluster_size}개 청크 포함 ({cluster_size/len(chunks)*100:.1f}%)")

    # 각 청크가 속한 군집 샘플 출력
    sample_chunks = min(5, len(chunks))
    logger.info("\n청크별 속한 군집 샘플:")
    for i in range(sample_chunks):
        chunk_preview = chunks[i][:50] + "..." if len(chunks[i]) > 50 else chunks[i]
        logger.info(f"  청크 #{i} -> 군집 {labels[i]}: {chunk_preview}")

    logger.info("\n===== 1차 군집별 요약 시작 =====")
    cluster_summaries, cluster_vecs = [], []
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        cluster_text = "\n".join(chunks[i] for i in idx)
        
        logger.info(f"\n----- 군집 {cid} 요약 중 ({len(idx)}개 청크) -----")
        text_preview = cluster_text[:150] + "..." if len(cluster_text) > 150 else cluster_text
        logger.info(f"  요약할 텍스트(일부): {text_preview}")
        
        summary = await summarize_fn(cluster_text)
        logger.info(f"  요약 결과: {summary}")
        
        cluster_summaries.append(summary)
        cluster_vecs.append(vecs[idx].mean(axis=0))  # 평균 임베딩

    logger.info("\n===== 2차 재귀 요약 시작 =====")
    final_summary = await _recursive_summary(cluster_summaries,
                                       np.vstack(cluster_vecs),
                                       summarize_fn,
                                       depth=1,
                                       max_depth=max_depth)
    
    logger.info(f"\n===== 최종 요약 결과 =====")
    logger.info(f"1차 군집 요약 {len(cluster_summaries)}개와 최종 요약 생성 완료")
    logger.info(f"최종 요약: {final_summary}")
    logger.info("===== 계층적 요약 완료 =====\n")
    
    return cluster_summaries, final_summary


# ────────────────────────────────
# LLM API 호출 유틸리티
# ────────────────────────────────
async def call_llm_api(text: str, max_tokens: int = None) -> str:
    """
    LLM API를 호출하여 텍스트 요약을 생성합니다.
    
    Parameters
    ----------
    text : 요약할 텍스트
    max_tokens : 생성할 최대 토큰 수 (None인 경우 settings에서 가져옴)
    
    Returns
    -------
    summary : 생성된 요약 텍스트
    """
    try:
        # 요약할 텍스트 일부 로깅
        text_preview = text[:150] + "..." if len(text) > 150 else text
        logger.info(f"\n===== LLM API 호출 시작 =====")
        logger.info(f"LLM 요약 요청할 텍스트 길이: {len(text)}문자")
        logger.info(f"\n입력 텍스트 일부: {text_preview}")
        
        # settings에서 API URL과 max_tokens 가져오기
        settings = get_settings()
        api_url = settings.llm_api_url
        if max_tokens is None:
            max_tokens = settings.llm_max_tokens
            
        logger.info(f"API 설정 - URL: {api_url}, max_tokens: {max_tokens}")
        
        system_prompt = "입력된 텍스트를 깔끔하게 요약하세요. 단, 정보의 손실이 없도록 요약하세요."
        user_prompt = f"다음 텍스트를 요약하세요:\n\n{text}"
        
        logger.info(f"System 프롬프트: {system_prompt}")
        logger.info(f"User 프롬프트: 다음 텍스트를 요약하세요: [Text 생략]")
        
        payload = {
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "max_tokens": min(max_tokens, 32768),
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        logger.info(f"API 호출 시작 - 파라미터: temperature={payload['temperature']}, top_p={payload['top_p']}")
        start_time = asyncio.get_event_loop().time()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            elapsed_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"API 응답 받음 - 상태 코드: {response.status_code}, 소요 시간: {elapsed_time:.2f}초")
            
            if response.status_code != 200:
                logger.error(f"LLM API 호출 실패: {response.status_code} - {response.text}")
                return f"요약 생성 실패: API 오류 ({response.status_code})"
                
            result = response.json()
            logger.info(f"API 응답 구조: {list(result.keys())}")
            
            # 응답 형식에 따라 적절히 파싱
            if "choices" in result and len(result["choices"]) > 0:
                logger.info("OpenAI 형식 응답 처리 중...")
                summary = result["choices"][0]["message"]["content"]
                '''
                # <think> 태그 제거 처리
                if "</think>" in summary:
                    logger.warning("</think> 태그 발견, 제거 중...")
                    original_length = len(summary)
                    think_content = summary.split("</think>")[0]
                    logger.info(f"\n제거된 생각 과정: {think_content[:150]}...")
                    summary = summary.split("</think>")[1].strip()
                    
                    new_length = len(summary)
                    logger.info(f"<think> 태그 제거 완료: {original_length} -> {new_length} 문자")
                
                summary_preview = summary[:150] + "..." if len(summary) > 150 else summary
                logger.info(f"\n최종 요약 결과({len(summary)}문자): {summary_preview}")
                return summary
                '''
            elif "response" in result:
                logger.info("Qwen 형식 응답 처리 중...")
                content = result["response"]
                processing_time = result.get("processing_time", "N/A")
                logger.info(f"API 처리 시간: {processing_time}")
                '''
                # <think> 태그 제거 처리
                if "</think>" in content:
                    logger.warning("<think> 태그 발견, 제거 중...")
                    original_length = len(content)
                    # <think>...</think> 사이의 내용 제거
                    think_content = content.split("</think>")[0]
                    logger.info(f"\n제거된 생각 과정: {think_content[:150]}...")
                    content = content.split("</think>")[1].strip()
                    new_length = len(content)
                    logger.info(f"<think> 태그 제거 완료: {original_length} -> {new_length} 문자")
                '''
                content_preview = content[:150] + "..." if len(content) > 150 else content
                logger.info(f"\n최종 요약 결과({len(content)}문자): {content_preview}")
                return content
                
            else:
                logger.error(f"LLM API 응답 형식 오류: {result}")
                return "요약 생성 실패: 응답 형식 오류"
                
    except Exception as e:
        logger.exception(f"LLM API 호출 중 오류 발생: {str(e)}")
        return f"요약 생성 실패: {str(e)}"


def create_summarize_fn() -> Callable[[str], str]:
    """
    동기 환경에서 사용할 수 있는 요약 함수를 생성합니다.
    """
    def summarize_text(text: str) -> str:
        return asyncio.run(call_llm_api(text))
    
    return summarize_text


async def create_async_summarize_fn() -> Callable[[str], str]:
    """
    비동기 환경에서 사용할 수 있는 요약 함수를 생성합니다.
    """
    async def summarize_text(text: str) -> str:
        return await call_llm_api(text)
    
    return summarize_text
