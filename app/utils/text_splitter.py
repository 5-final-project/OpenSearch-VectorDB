# coding: utf-8
from typing import List, Any
import logging

from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document
from kiwipiepy import Kiwi

from app.config.settings import Settings

logger = logging.getLogger(__name__)
settings_obj = Settings() # 전역 설정 객체

# --- Kiwi 초기화 --- 
# Kiwi 객체는 한 번만 초기화하여 재사용
print("--- Initializing Kiwi object for KiwiSentenceSplitter... ---", flush=True)
_kiwi_splitter_instance = Kiwi()
print("--- Kiwi object for KiwiSentenceSplitter initialized and prepared. ---", flush=True)

class KiwiSentenceSplitter(TextSplitter):
    """
    Splits text into sentences using Kiwi (kiwipiepy) and then groups them into chunks
    respecting chunk size and overlap, inheriting from TextSplitter.
    This is an adaptation of KoreanSentenceSplitter using Kiwi instead of KSS.
    """

    def __init__(
        self,
        chunk_size: int = settings_obj.chunk_size, # 문자 수 기준
        chunk_overlap: int = settings_obj.chunk_overlap, # 문자 수 기준
        length_function = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
        **kwargs: Any,
    ):
        """Initializes the KiwiSentenceSplitter.

        Args:
            chunk_size: Max size of chunks (in characters).
            chunk_overlap: Max overlap between chunks (in characters).
            length_function: Function to measure text length.
            keep_separator: Passed to TextSplitter.
            add_start_index: Passed to TextSplitter.
            strip_whitespace: Passed to TextSplitter.
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            keep_separator=keep_separator,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
            **kwargs
        )
        # Store custom arguments if any, or rely on parent's storage
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._strip_whitespace = strip_whitespace # Ensure this is stored for _merge_sentences

    def _split_text_with_kiwi(self, text: str) -> List[str]:
        """Splits text into sentences using Kiwi."""
        try:
            sentences_obj = _kiwi_splitter_instance.split_into_sents(text)
            sentences = [s.text.strip() for s in sentences_obj if s.text.strip()] # Kiwi의 Sentence 객체에서 텍스트 추출 및 정리
            logger.debug(f"Kiwi split into {len(sentences)} sentences.")
            return sentences
        except Exception as e:
            logger.error(f"Error splitting text with Kiwi: {e}", exc_info=True)
            logger.warning("Falling back to newline splitting due to Kiwi error.")
            sentences = text.split('\n')
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences

    def split_text(self, text: str) -> List[str]:
        """Splits text by sentences and merges them into chunks."""
        sentences = self._split_text_with_kiwi(text)
        if not sentences:
            return []
        separator = " " # 문장들을 합칠 때 사용할 구분자
        return self._merge_sentences(sentences, separator)

    def _merge_sentences(self, sentences: List[str], separator: str) -> List[str]:
        """Merges sentences into chunks respecting chunk_size and chunk_overlap.
        This logic is adapted from the KoreanSentenceSplitter in VectorDB project.
        """
        chunks = []
        current_chunk_sentences: List[str] = []
        current_length = 0
        separator_len = self._length_function(separator)

        for i, sentence in enumerate(sentences):
            sentence_len = self._length_function(sentence)

            if sentence_len > self._chunk_size:
                logger.warning(
                    f"Sentence starting with '{sentence[:80]}...' (len: {sentence_len}) is longer "
                    f"than chunk_size {self._chunk_size}. It will be a separate chunk."
                )
                if current_chunk_sentences: # Add previous chunk first
                    chunks.append(separator.join(current_chunk_sentences))
                chunks.append(sentence)
                current_chunk_sentences = []
                current_length = 0
                continue

            potential_length = current_length + sentence_len + (separator_len if current_chunk_sentences else 0)

            if potential_length <= self._chunk_size:
                current_chunk_sentences.append(sentence)
                current_length = potential_length
            else:
                if current_chunk_sentences:
                    chunks.append(separator.join(current_chunk_sentences))

                overlap_sentences: List[str] = []
                overlap_length = 0
                for j in range(len(current_chunk_sentences) - 1, -1, -1):
                    prev_sentence = current_chunk_sentences[j]
                    prev_sentence_len = self._length_function(prev_sentence)
                    potential_overlap_len = overlap_length + prev_sentence_len + (separator_len if overlap_sentences else 0)

                    if potential_overlap_len <= self._chunk_overlap:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_length = potential_overlap_len
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences
                current_length = overlap_length

                potential_length_with_overlap = current_length + sentence_len + (separator_len if current_chunk_sentences else 0)
                if potential_length_with_overlap <= self._chunk_size:
                     current_chunk_sentences.append(sentence)
                     current_length = potential_length_with_overlap
                else:
                     if current_chunk_sentences: # Finalize overlap chunk if it exists
                         chunks.append(separator.join(current_chunk_sentences))
                     current_chunk_sentences = [sentence] # Start new chunk with current sentence
                     current_length = sentence_len
        
        if current_chunk_sentences:
            chunks.append(separator.join(current_chunk_sentences))

        if self._strip_whitespace:
             chunks = [chunk.strip() for chunk in chunks if chunk.strip()] # 추가: 비어있는 청크 제거

        logger.info(f"Split text into {len(chunks)} chunks using KiwiSentenceSplitter.")
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits Documents by calling split_text."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)