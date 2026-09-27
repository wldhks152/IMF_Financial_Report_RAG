"""
vectorstore.py — 청크를 벡터로 바꿔 Chroma DB에 저장하고, 다시 불러오는 파일

RAG 파이프라인의 세 번째 단계.
[Document(청크)] ──임베딩 모델──▶ [벡터(숫자 1536개)] ──▶ Chroma DB (chroma_db 폴더)
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def get_embeddings() -> OpenAIEmbeddings:
    """텍스트 → 벡터 변환기. 저장할 때와 검색할 때 반드시 같은 것을 써야 한다."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def make_chunk_id(chunk: Document) -> str:
    """청크마다 고유 ID를 만든다. 예: 'text.pdf:p4:1600', 표 행이면 'text2.pdf:p13:row5'

    같은 청크는 항상 같은 ID가 나오므로, 인덱스를 다시 만들어도 중복 저장되지 않고 덮어쓰기(upsert)가 된다.
    """
    m = chunk.metadata
    if m.get("type") == "table":
        return f"{m['file_name']}:p{m['page'] + 1}:row{m['row']}"
    return f"{m['file_name']}:p{m['page'] + 1}:{m['start_index']}"


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """청크 전부를 임베딩해서 Chroma DB에 저장한다. (OpenAI API 호출 발생)"""
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),           # 각 청크의 page_content를 이 모델로 벡터화
        ids=[make_chunk_id(c) for c in chunks],
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),    # 이 폴더에 자동 저장 → 프로그램을 꺼도 남아 있다
        collection_metadata={"hnsw:space": "cosine"},  # 벡터 사이 거리를 '코사인 거리'로 잰다
    )


def load_vectorstore() -> Chroma:
    """이미 저장된 Chroma DB를 연다. (임베딩을 다시 계산하지 않으므로 API 비용 없음)"""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),  # 질문을 벡터로 바꿀 때 필요
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    # 여기서는 DB를 만들지 않고, '임베딩이 무엇인지'만 작게 실험해본다.
    # (DB 전체 생성은 다음 단계 build_index.py에서)
    import numpy as np

    embeddings = get_embeddings()

    # 1) 문장 하나를 벡터로 바꿔보기
    vec = embeddings.embed_query("Global growth is projected at 3.3 percent in 2026.")
    print(f"벡터 길이: {len(vec)}개의 숫자")
    print(f"앞 5개 값: {[round(v, 4) for v in vec[:5]]}\n")

    # 2) 의미가 비슷한 문장끼리 벡터도 가까운지 확인
    question = "올해 세계 경제 성장률 전망은?"
    sentences = [
        "Global growth is projected at 3.3 percent in 2026.",               # 의미가 가장 비슷 (영어!)
        "Global headline inflation is expected to decline to 3.8 percent.",  # 경제 얘기지만 주제가 다름
        "The cat is sleeping on the sofa.",                                   # 전혀 무관
    ]

    q_vec = np.array(embeddings.embed_query(question))
    s_vecs = np.array(embeddings.embed_documents(sentences))

    # 코사인 유사도 = 두 벡터가 가리키는 방향이 얼마나 같은가 (1에 가까울수록 비슷)
    sims = s_vecs @ q_vec / (np.linalg.norm(s_vecs, axis=1) * np.linalg.norm(q_vec))

    print(f"질문: {question}")
    for s, sim in sorted(zip(sentences, sims), key=lambda x: -x[1]):
        print(f"  {sim:.3f}  {s}")
