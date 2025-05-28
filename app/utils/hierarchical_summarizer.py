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
from google import genai
from google.genai.types import GenerateContentConfig

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
    k_max: int = 20,
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
    k = int(np.sqrt(len(chunks)))  # 입력 Chunks 길이의 제곱근을 k로 설정
    # k 는 k_min 이상 k_max이하로 제한
    k = max(k, k_min)
    k = min(k, k_max)
    logger.info(f"초기 k 값: {k} (청크 수: {len(chunks)})")
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

async def enhanced_summarize_document(
    chunks: List[str],
    embeddings: List[np.ndarray],
    summarize_fn: Callable[[str], str],
    **kwargs
) -> Tuple[List[str], str]:
    """
    계층적 요약 파이프라인을 실행하고 결과를 가독성 높은 형식으로 변환합니다.
    
    Parameters
    ----------
    chunks       : 분할된 텍스트 조각 리스트
    embeddings   : 각 조각의 임베딩 (chunks와 길이 같아야 함)
    summarize_fn : 텍스트를 요약해 주는 LLM 함수 (비동기 함수)
    **kwargs     : summarize_document에 전달할 추가 인자
    
    Returns
    -------
    formatted_cluster_summaries : 형식화된 1차 군집별 요약 리스트
    formatted_final_summary     : 형식화된 통합 최종 요약
    """
    # 기존 summarize_document 함수 호출
    cluster_summaries, final_summary = await summarize_document(
        chunks, embeddings, summarize_fn, **kwargs
    )
    
    logger.info("\n===== 최종 형식화 처리 시작 =====")
    
    # 모든 클러스터 요약을 하나의 통합 문서로 처리
    logger.info(f"모든 클러스터 요약({len(cluster_summaries)}개)을 통합 문서로 처리 중...")
    
    # 클러스터 요약들을 하나의 문자열로 결합
    all_cluster_summaries = "\n\n".join([
        f"[클러스터 {i+1}]\n{summary}" for i, summary in enumerate(cluster_summaries)
    ])
    
    # 통합 정리본 생성 프롬프트
    format_clusters_prompt = f"""
다음은 문서의 요약들입니다. 이 요약들을 하나의 통합된 문서로 재구성해주세요:

1. 문서 목적, 주요 내용 요약, 핵심 아이디어 및 중요한 사실, 결론을 소제목으로 적용하세요.
2. 주요 내용 요약 (각 부분의 핵심을 포함)
3. 핵심 아이디어 및 중요한 사실 (중요한 포인트들을 글머리 기호(•)로 나열)
4. 결론 및 시사점
5. 결과는 한국어로 작성해주세요.
6. 접두어를 붙이지 않고 순수한 브리핑 텍스트만 작성해주세요.

각 섹션은 명확하게 구분되어야 하며, 정보는 간결하고 명확하게 제시해주세요.
적절한 소제목과 문단 나누기를 적용하여 읽기 쉽게 만들어주세요.

===== 요약 텍스트 =====
{all_cluster_summaries}
"""
    
    try:
        # 모든 클러스터 요약을 통합하여 단일 정리본 생성
        logger.info("모든 클러스터 요약을 통합하여 정리본 생성 중...")
        integrated_summary = await call_llm_api(format_clusters_prompt, max_tokens=16384, prompt_type="raw")
        
        # 최종 요약 형식화
        logger.info("최종 요약 형식화 중...")
        format_final_prompt = f"""
다음은 문서의 요약입니다. 이 요약을 읽기 쉽고 이해하기 쉬운 형태로 재구성해주세요:

1. 적절한 문단 나누기를 적용하세요.
2. 필요한 경우 글머리 기호를 사용하여 주요 포인트를 강조하세요.
3. 논리적 흐름을 유지하면서 문장을 자연스럽게 연결하세요.
4. 결과는 한국어로 작성해주세요.
5. 접두어를 붙이지 않고 순수한 문장만 작성해주세요.

원본 요약:
{final_summary}
"""
        formatted_final_summary = await call_llm_api(format_final_prompt, max_tokens=16384, prompt_type="raw")
        
        logger.info("===== 최종 형식화 처리 완료 =====\n")
        
        # 클러스터 요약을 단일 항목(통합 정리본)으로 반환
        return [integrated_summary], formatted_final_summary
        
    except Exception as e:
        logger.error(f"요약 형식화 중 오류 발생: {str(e)}")
        # 오류 발생 시 원본 요약 반환
        logger.info("오류 발생으로 원본 형태로 반환합니다.")
        return [f"# 통합 요약\n\n{all_cluster_summaries}"], final_summary

# ────────────────────────────────
# LLM API 호출 유틸리티
# ────────────────────────────────
async def call_llm_api(text: str, max_tokens: int = None, prompt_type: str = "summarize") -> str:
    """
    LLM API를 호출하여 텍스트 요약을 생성합니다.
    
    Parameters
    ----------
    text : 요약할 텍스트
    max_tokens : 생성할 최대 토큰 수 (None인 경우 settings에서 가져옴)
    prompt_type : 프롬프트 타입 ("summarize" 또는 "raw")
                  "summarize": 기본 요약 프롬프트 적용
                  "raw": 입력 텍스트를 그대로 사용
    
    Returns
    -------
    summary : 생성된 요약 텍스트
    """
    try:
        settings = get_settings()
        
        # Gemini API 또는 기존 API 선택
        use_gemini = True  # Gemini API 사용 여부 (True/False)
        
        if use_gemini:
            # Gemini API 사용
            api_key = settings.gemini_api_key
            if not api_key:
                logger.error("Gemini API 키가 설정되지 않았습니다.")
                return "요약 생성 실패: API 키가 설정되지 않았습니다."
            
            try:
                # Gemini 클라이언트 초기화 (새로운 방식)
                client = genai.Client(api_key=api_key)
                
                # 프롬프트 타입에 따라 내용 결정
                if prompt_type == "summarize":
                    # 요약 프롬프트 적용
                    system_prompt = """다음 텍스트를 요약해주세요. 
요약할 때는 다음 규칙을 따라주세요:
1. 중요한 내용을 누락하지 않고 간결하게 요약하세요.
2. 원문의 주요 내용과 핵심 정보를 포함하세요.
3. 불필요한 세부 사항은 생략하되, 중요한 세부 정보는 유지하세요.
4. 요약은 원문의 흐름과 구조를 유지해야 합니다.
5. 결과는 한국어로 작성해주세요.
6. "요약:" 또는 "**요약:**"과 같은 접두어를 붙이지 말고 순수한 요약 문장만 작성해주세요.
"""
                    final_prompt = f"{system_prompt}\n\n요약할 텍스트:\n\n{text}"
                else:
                    # raw 모드: 입력 텍스트를 그대로 사용
                    final_prompt = text
                
                # 비동기 호출을 위한 wrapper 함수
                async def run_gemini_call():
                    # 새로운 방식으로 생성 요청
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model='gemini-1.5-flash',
                        contents=final_prompt,
                        config=GenerateContentConfig(
                            max_output_tokens=max_tokens or settings.llm_max_tokens
                        )
                    )
                    return response.text
                
                logger.info("Gemini API 호출 시작")
                start_time = asyncio.get_event_loop().time()
                
                content = await run_gemini_call()
                
                end_time = asyncio.get_event_loop().time()
                logger.info(f"Gemini API 응답 받음 - 소요 시간: {end_time - start_time:.2f}초")
                
                # 내용 확인 및 로그
                content_preview = content[:150] + "..." if len(content) > 150 else content
                logger.info(f"\n최종 요약 결과({len(content)}문자): {content_preview}")
                return content
                
            except Exception as e:
                logger.exception(f"Gemini API 호출 중 오류 발생: {str(e)}")
                return f"요약 생성 실패: {str(e)}"
        else:
            # 기존 API 사용 코드 유지
            api_url = settings.llm_api_url
            logger.info(f"LLM API URL: {api_url}")
            
            if not max_tokens:
                max_tokens = settings.llm_max_tokens
            
            # 문서가 너무 길면 내부 청크로 분할하여 재귀적 요약
            if len(text) > 100000:  # 예: 10만자 이상
                logger.warning(f"텍스트가 너무 깁니다 ({len(text)}자). 내부 청크로 분할합니다.")
                
                # 단락 기준으로 분할
                paragraphs = text.split('\n\n')
                if len(paragraphs) < 3:  # 단락이 너무 적으면 다른 방식으로 분할
                    paragraphs = text.split('\n')
                
                logger.info(f"총 {len(paragraphs)}개 단락으로 분할됨")
                
                # 임베딩 생성 (샘플링으로 처리)
                paragraph_embeddings = []
                for _ in range(len(paragraphs)):
                    # 임의의 임베딩 (실제로는 임베딩 모델 사용 필요)
                    paragraph_embeddings.append(np.random.random(1024))
                
                # 내부 클러스터링 및 요약 수행
                logger.info(f"내부 클러스터링 시작: {len(paragraphs)}개 단락 처리 중...")
                summarize_fn = await create_async_summarize_fn()
                _, internal_summary = await enhanced_summarize_document(
                    paragraphs, 
                    paragraph_embeddings,
                    summarize_fn,
                    k_min=2,
                    k_max=5,
                    max_depth=2  # 깊이를 줄여 빠르게 처리
                )
                
                logger.info(f"내부 클러스터링 요약 완료. 요약 길이: {len(internal_summary)}")
                return internal_summary
                
            # 요약 프롬프트 생성
            system_prompt = """다음 텍스트를 요약해주세요. 
요약할 때는 다음 규칙을 따라주세요:
1. 중요한 내용을 누락하지 않고 간결하게 요약하세요.
2. 원문의 주요 내용과 핵심 정보를 포함하세요.
3. 불필요한 세부 사항은 생략하되, 중요한 세부 정보는 유지하세요.
4. 요약은 원문의 흐름과 구조를 유지해야 합니다.
5. 결과는 한국어로 작성해주세요.
6. "요약:" 또는 "**요약:**"과 같은 접두어를 붙이지 말고 순수한 요약 문장만 작성해주세요.
"""

            user_prompt = f"요약할 텍스트:\n\n{text}"
            
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
            
            # API 호출
            async with httpx.AsyncClient(timeout=600) as client:  # 10분 타임아웃
                try:
                    response = await client.post(
                        api_url, 
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()  # 오류 응답 검사
                    
                    end_time = asyncio.get_event_loop().time()
                    logger.info(f"API 응답 받음 - 상태 코드: {response.status_code}, 소요 시간: {end_time - start_time:.2f}초")
                    
                    # 응답 파싱
                    result = response.json()
                    logger.info(f"API 응답 구조: {list(result.keys())}")
                    
                    # 응답 형식에 따라 적절히 파싱
                    if "choices" in result and len(result["choices"]) > 0:
                        logger.info("OpenAI 형식 응답 처리 중...")
                        summary = result["choices"][0]["message"]["content"]
                        
                        # <think> 태그 제거 처리
                        if "</think>" in summary:
                            logger.warning("<think> 태그 발견, 제거 중...")
                            original_length = len(summary)
                            # <think>...</think> 사이의 내용 제거
                            think_content = summary.split("</think>")[0]
                            logger.info(f"\n제거된 생각 과정: {think_content[:150]}...")
                            summary = summary.split("</think>")[1].strip()
                            new_length = len(summary)
                            logger.info(f"<think> 태그 제거 완료: {original_length} -> {new_length} 문자")
                        
                        content_preview = summary[:150] + "..." if len(summary) > 150 else summary
                        logger.info(f"\n최종 요약 결과({len(summary)}문자): {content_preview}")
                        return summary
                    
                    elif "response" in result:
                        logger.info("Qwen 형식 응답 처리 중...")
                        content = result["response"]
                        logger.info(f"API 처리 시간: {result.get('processing_time', 'N/A')}")
                        
                        # <think> 태그 제거 처리
                        if "<think>" in content and "</think>" in content:
                            logger.warning("<think> 태그 발견, 제거 중...")
                            original_length = len(content)
                            # <think>...</think> 사이의 내용 제거
                            think_content = content.split("</think>")[0]
                            logger.info(f"\n제거된 생각 과정: {think_content[:150]}...")
                            content = content.split("</think>")[1].strip()
                            new_length = len(content)
                            logger.info(f"<think> 태그 제거 완료: {original_length} -> {new_length} 문자")
                        
                        content_preview = content[:150] + "..." if len(content) > 150 else content
                        logger.info(f"\n최종 요약 결과({len(content)}문자): {content_preview}")
                        return content
                        
                    else:
                        logger.error(f"LLM API 응답 형식 오류: {result}")
                        return "요약 생성 실패: 응답 형식 오류"
                        
                except Exception as e:
                    logger.exception(f"LLM API 호출 중 오류 발생: {str(e)}")
                    return f"요약 생성 실패: {str(e)}"
    
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
