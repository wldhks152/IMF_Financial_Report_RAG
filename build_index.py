"""
build_index.py — PDF → 청크 → 벡터 → Chroma DB 를 한 번에 실행하는 스크립트

지금까지 만든 부품(loader, splitter, tables, vectorstore)을 순서대로 조립한다.
PDF를 추가·삭제·교체했을 때만 다시 실행하면 된다.

    python build_index.py
"""

import time

from loader import load_all_pdfs
from splitter import split_documents
from tables import convert_table_pages, is_table_page
from vectorstore import build_vectorstore, load_vectorstore


def main():
    start = time.time()

    # 1) 로드: data 폴더의 PDF → 페이지 단위 Document
    print("[1/5] PDF 읽는 중...")
    pages = load_all_pdfs()
    print(f"      → {len(pages)}페이지\n")

    # 2) 표 페이지와 일반 페이지 나누기
    table_pages = [p for p in pages if is_table_page(p)]
    text_pages = [p for p in pages if not is_table_page(p)]

    # 3) 표 페이지 → 행 단위 Document (GPT 변환, 결과는 table_cache.json에 저장되어 다음부터는 재사용)
    print(f"[2/5] 표 페이지 {len(table_pages)}개 → 행 단위로 변환 중...")
    table_rows, failed_pages = convert_table_pages(table_pages)
    print(f"      → 표 행 {len(table_rows)}개")
    if failed_pages:
        # 변환에 실패한 표 페이지는 버리지 않고 일반 텍스트로 처리 (없는 것보다는 낫다)
        print(f"      → 변환 실패 {len(failed_pages)}페이지는 일반 텍스트로 처리")
        text_pages += failed_pages
    print()

    # 4) 일반 페이지 → 청크
    print(f"[3/5] 일반 페이지 {len(text_pages)}개를 청크로 자르는 중...")
    chunks = split_documents(text_pages)
    print(f"      → {len(chunks)}개 청크\n")

    documents = chunks + table_rows

    # 5) 기존 컬렉션 비우기
    #    ID 덕분에 같은 청크는 덮어쓰기가 되지만, data 폴더에서 '삭제한' PDF의 청크는 DB에 그대로 남는다.
    #    그래서 항상 깨끗하게 비우고 새로 만든다.
    print("[4/5] 기존 DB 비우는 중...")
    load_vectorstore().delete_collection()
    print("      → 완료\n")

    # 6) 임베딩 + 저장: 청크 → 벡터 → chroma_db 폴더
    print(f"[5/5] {len(documents)}개 문서(청크 {len(chunks)} + 표 행 {len(table_rows)}) 임베딩 & 저장 중... (OpenAI API 호출)")
    db = build_vectorstore(documents)
    print(f"      → DB에 저장된 벡터 수: {db._collection.count()}\n")

    print(f"완료! ({time.time() - start:.1f}초)")


if __name__ == "__main__":
    main()
