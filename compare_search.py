"""
compare_search.py — 한국어 질문 그대로 검색 vs 영어로 재작성해서 검색, 결과 비교 실험

    python compare_search.py

GPT 답변은 만들지 않고 '검색 단계'만 비교한다. (재작성용 GPT 호출 + 임베딩 호출만 발생)
거리(distance) = 1 - 코사인 유사도 → 작을수록 질문과 가깝다.
"""

from rag_chain import build_rewrite_chain, report_name
from vectorstore import load_vectorstore

QUESTIONS = [
    "2026년 세계 경제 성장률 전망은?",
    "한국 경제 전망은 어떤가요?",
    "인공지능 투자 붐이 꺼지면 어떤 일이 생기나요?",
    "중앙은행은 금리를 어떻게 해야 하나요?",
]


def show(db, query: str):
    results = db.similarity_search_with_score(query, k=4)
    for doc, dist in results:
        preview = doc.page_content[:70].replace("\n", " ")
        print(f"    {dist:.3f} | {report_name(doc)[:38]:<38} {doc.metadata['page'] + 1:>3}p | {preview}")
    dists = [d for _, d in results]
    print(f"    → 1등 거리 {dists[0]:.3f}, 1등과 4등 차이 {dists[-1] - dists[0]:.3f}")


if __name__ == "__main__":
    db = load_vectorstore()
    rewrite = build_rewrite_chain()

    for q in QUESTIONS:
        en = rewrite.invoke({"question": q, "chat_history": []})
        print("=" * 110)
        print(f"[한국어 그대로] {q}")
        show(db, q)
        print(f"\n[영어 재작성]   {en}")
        show(db, en)
        print()
