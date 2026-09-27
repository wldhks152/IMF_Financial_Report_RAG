"""
splitter.py — 페이지 단위 Document를 작은 청크(chunk)로 자르는 파일

RAG 파이프라인의 두 번째 단계.
[Document(13페이지)] → [Document(청크1), Document(청크2), ...]
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents(docs: list[Document]) -> list[Document]:
    """Document 리스트를 CHUNK_SIZE 크기의 청크로 잘라서 반환한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 자르는 기준을 '큰 단위 → 작은 단위' 순서로 시도한다.
        # 문단(\n\n)으로 잘라서 크기가 맞으면 거기서 끝, 너무 크면 줄(\n) → 문장(". ") → 단어(" ") → 글자("") 순으로 내려간다.
        separators=["\n\n", "\n", ". ", " ", ""],
        # 각 청크가 원래 페이지의 몇 번째 글자에서 시작했는지 metadata에 기록
        add_start_index=True,
    )
    # split_documents()는 원본 Document의 metadata(source, page 등)를 모든 청크에 그대로 복사한다.
    return splitter.split_documents(docs)


if __name__ == "__main__":
    from loader import load_all_pdfs

    pages = load_all_pdfs()
    chunks = split_documents(pages)

    print(f"페이지 {len(pages)}개 → 청크 {len(chunks)}개\n")

    # 1) 청크 길이 분포
    lengths = [len(c.page_content) for c in chunks]
    print("=== 청크 길이 ===")
    print(f"  최소 {min(lengths)}자 / 평균 {sum(lengths) // len(lengths)}자 / 최대 {max(lengths)}자\n")

    # 2) 페이지별 청크 개수
    print("=== 페이지별 청크 개수 ===")
    for page in pages:
        name, page_no = page.metadata["file_name"], page.metadata["page"]
        n = sum(1 for c in chunks if c.metadata["file_name"] == name and c.metadata["page"] == page_no)
        print(f"  {name} {page_no + 1:>2}페이지: {'■' * n} ({n}개)")

    # 3) 청크 하나 들여다보기 — metadata가 페이지에서 그대로 복사됐는지 확인
    sample = chunks[2]
    print("\n=== chunks[2] ===")
    print("metadata:", {k: sample.metadata[k] for k in ("file_name", "page", "start_index")})
    print(sample.page_content)

    # 4) 겹침(overlap) 확인 — 같은 페이지의 이웃한 두 청크에서 끝부분과 시작 부분 비교
    #    (겹침은 같은 페이지 안에서만 생긴다. 페이지가 다르면 원래 다른 Document였기 때문)
    a, b = chunks[2], chunks[3]
    print("\n=== 겹침 확인: chunks[2]의 끝 250자 ===")
    print("..." + a.page_content[-250:])
    print("\n=== chunks[3]의 앞 250자 ===")
    print(b.page_content[:250] + "...")
