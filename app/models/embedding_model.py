"""HuggingFace 임베딩 래퍼."""
import torch
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.settings import get_settings


@lru_cache(maxsize=1)
def _load_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:   
        device = 'cpu'
    return device


def _load_embeddings():
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={
            "device": _load_device(),            # GPU 없을 경우 CPU
        },
    )

embedding_model = _load_embeddings()