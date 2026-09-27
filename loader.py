"""
loader.py — data 폴더의 PDF를 전부 읽어서 LangChain Document 리스트로 바꾸는 파일

RAG 파이프라인의 첫 단계.
data/*.pdf → [Document(A.pdf 1페이지), Document(A.pdf 2페이지), ..., Document(B.pdf 1페이지), ...]
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import DATA_DIR


def load_pdf(path: Path) -> list[Document]:
    """PDF 하나를 페이지 단위로 읽어 Document 리스트로 반환한다."""
    loader = PyPDFLoader(str(path))

    # load()는 페이지마다 Document 하나를 만든다.
    # Document = page_content(본문 텍스트) + metadata(출처 정보) 두 부분으로 구성된 객체
    docs = loader.load()

    # 여러 PDF를 섞어 쓰니까, 나중에 "어느 파일의 몇 페이지"인지 보여주기 쉽도록 파일 이름을 따로 기록해둔다.
    # (metadata["source"]에는 전체 경로가 들어 있어서 화면에 보여주기엔 길다)
    for doc in docs:
        doc.metadata["file_name"] = path.name
    return docs


def load_all_pdfs(data_dir: Path = DATA_DIR) -> list[Document]:
    """폴더 안의 모든 PDF를 읽어서 하나의 Document 리스트로 합친다."""
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"{data_dir} 폴더에 PDF가 없습니다.")

    all_docs = []
    for path in pdf_paths:
        docs = load_pdf(path)
        print(f"  읽음: {path.name} ({len(docs)}페이지)")
        all_docs.extend(docs)
    return all_docs


if __name__ == "__main__":
    docs = load_all_pdfs()

    print(f"\n총 {len(docs)}개의 Document (= 모든 PDF의 페이지 수 합계)\n")

    # 첫 번째 Document가 어떻게 생겼는지 들여다보기
    first = docs[0]
    print("=== docs[0].metadata ===")
    print(first.metadata)
    print("\n=== docs[0].page_content (앞 500자) ===")
    print(first.page_content[:500])

    # 파일·페이지별 글자 수 → 청크를 몇 개로 나누게 될지 가늠해볼 수 있다
    print("\n=== 페이지별 글자 수 ===")
    for doc in docs:
        name = doc.metadata["file_name"]
        page = doc.metadata["page"]  # 0부터 시작
        print(f"  {name} {page + 1:>2}페이지: {len(doc.page_content):>5}자")
    print(f"  합계: {sum(len(d.page_content) for d in docs)}자")
