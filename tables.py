"""
tables.py — 표 페이지를 찾아서 '한 행 = 한 문장' Document로 바꾸는 파일

문제: 표를 1000자 단위로 자르면 "Japan –0.2 1.1 0.6 0.7" 같은 숫자 줄이 열 제목(연도)과 떨어진다.
      → 임베딩해도 의미 없는 벡터가 되고, GPT도 0.6이 몇 년도 값인지 알 수 없다.
해결: 색인할 때 표 페이지를 GPT에게 보내서, 각 행을 제목·열 이름이 붙은 문장으로 바꾼다.

    [표 페이지] ──GPT──▶ "Real GDP — Japan: 2024 -0.2; 2025 1.1; 2026 projection 0.6; ..."  (행마다 Document 1개)
"""

import hashlib
import json
import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import CHAT_MODEL, TABLE_CACHE_PATH


# ---------------------------------------------------------------
# 1. 표 페이지 찾기
# ---------------------------------------------------------------
# 줄 맨 앞에 "Table 1." / "Annex Table 1.1.2." / "Table A4." 같은 표 제목이 있는지
TABLE_TITLE = re.compile(r"^\s*(Annex )?Table [A-Z]?\d+(\.\d+)*\.", re.MULTILINE)
NUMBER_TOKEN = re.compile(r"^[–\-−]?\d+(\.\d+)?$")


def is_table_page(doc: Document) -> bool:
    """표 제목이 있고, 단어 중 숫자 비율이 15% 이상이면 표 페이지로 본다.

    (본문에서 "see Table 1.1" 처럼 표를 언급만 하는 페이지는 숫자 비율이 낮아서 걸러진다)
    """
    tokens = doc.page_content.split()
    if not tokens or not TABLE_TITLE.search(doc.page_content):
        return False
    number_ratio = sum(bool(NUMBER_TOKEN.match(t)) for t in tokens) / len(tokens)
    return number_ratio >= 0.15


# ---------------------------------------------------------------
# 2. GPT에게 표 → 문장 변환 요청 (구조화된 출력)
# ---------------------------------------------------------------
# with_structured_output(): GPT가 자유 문장 대신 아래 형식(JSON)에 맞춰서만 답하도록 강제한다.
class Table(BaseModel):
    title: str = Field(description="표 제목. 예: 'Table 1. Overview of the World Economic Outlook Projections'")
    unit: str = Field(description="단위. 예: 'Percent change'")
    rows: list[str] = Field(description="데이터 행마다 한 줄씩, 스스로 뜻이 통하는 문장")


class TablePage(BaseModel):
    tables: list[Table]


TABLE_PROMPT = """You convert tables extracted from an IMF World Economic Outlook PDF page into self-contained text rows.
The text was extracted without layout, so each data row appears as a label followed by numbers in column order.
Use the header lines (years, 'Projections', 'Difference from ...', 'Q4 over Q4', etc.) to work out what each column means.

Write each data row as ONE line:
  <section, if the table has sections> — <row label>: <column label> <value>; <column label> <value>; ...
- Column labels must be complete, combining multi-level headers. e.g. "2026 projection", "difference from April 2026 WEO, 2027", "Q4 over Q4, 2026".
- Copy every number exactly as printed. Use '-' for negative numbers. Never compute, round or guess.
- If a row's numbers cannot be matched to columns with confidence, skip that row.
- Skip footnotes, source notes and page headers.
- If the page has several tables, return each separately."""

table_prompt = ChatPromptTemplate.from_messages([
    ("system", TABLE_PROMPT),
    ("human", "{page_text}"),
])


def build_table_chain():
    # 열과 숫자를 짝짓는 건 꼼꼼함이 필요한 작업이라 reasoning_effort를 low로 둔다
    llm = ChatOpenAI(model=CHAT_MODEL, reasoning_effort="low")
    return table_prompt | llm.with_structured_output(TablePage)


# ---------------------------------------------------------------
# 3. 검증: GPT가 원문에 없는 숫자를 만들어냈는지 확인
# ---------------------------------------------------------------
def normalize(text: str) -> str:
    text = text.replace("–", "-").replace("−", "-")
    return re.sub(r"(\d) \.(\d)", r"\1.\2", text)  # PDF에서 "0 .6"처럼 끊긴 숫자 복구


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r"-?\d+(?:\.\d+)?", normalize(text)))


def verified_rows(table: Table, page_text: str) -> tuple[list[str], int]:
    """원문 페이지에 없는 숫자가 하나라도 들어간 행은 버린다. (숫자를 '지어내는' 실수 방지)

    한계: 숫자는 원문에 있지만 '다른 열'에 잘못 붙인 실수까지는 잡지 못한다.
    """
    allowed = numbers_in(page_text)
    kept = [row for row in table.rows if numbers_in(row) <= allowed]
    return kept, len(table.rows) - len(kept)


# ---------------------------------------------------------------
# 4. 캐시: 같은 페이지는 GPT를 다시 부르지 않는다
# ---------------------------------------------------------------
def cache_key(doc: Document) -> str:
    # 페이지 내용의 해시를 키에 넣어서, PDF가 바뀌면 자동으로 다시 변환되게 한다
    digest = hashlib.md5(doc.page_content.encode()).hexdigest()[:10]
    return f"{doc.metadata['file_name']}:p{doc.metadata['page'] + 1}:{digest}"


def load_cache() -> dict:
    if TABLE_CACHE_PATH.exists():
        return json.loads(TABLE_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict):
    TABLE_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------
# 5. 전체: 표 페이지 목록 → 행 Document 목록
# ---------------------------------------------------------------
def convert_table_pages(pages: list[Document]) -> tuple[list[Document], list[Document]]:
    """표 페이지들을 행 단위 Document로 바꾼다.

    반환: (행 Document 목록, 변환에 실패한 페이지 목록 → 호출한 쪽에서 일반 텍스트로 처리)
    """
    cache = load_cache()
    todo = [p for p in pages if cache_key(p) not in cache]

    if todo:
        print(f"      GPT로 표 변환: {len(todo)}페이지 (캐시 {len(pages) - len(todo)}페이지)")
        chain = build_table_chain()
        # batch(): 여러 요청을 동시에 보낸다. max_concurrency = 동시에 보낼 최대 개수
        results = chain.batch(
            [{"page_text": p.page_content} for p in todo],
            config={"max_concurrency": 8},
            return_exceptions=True,   # 한 페이지가 실패해도 나머지는 계속
        )
        for page, result in zip(todo, results):
            if isinstance(result, Exception):
                print(f"      ⚠ 실패: {cache_key(page)} ({type(result).__name__})")
                continue
            cache[cache_key(page)] = result.model_dump()
        save_cache(cache)

    row_docs, failed_pages, dropped = [], [], 0
    for page in pages:
        if cache_key(page) not in cache:
            failed_pages.append(page)
            continue
        row_no = 0
        for table in TablePage(**cache[cache_key(page)]).tables:
            rows, n_dropped = verified_rows(table, page.page_content)
            dropped += n_dropped
            for row in rows:
                row_docs.append(Document(
                    # 제목과 단위를 행마다 붙여야 행 하나만 검색돼도 무슨 숫자인지 알 수 있다
                    page_content=f"{table.title} ({table.unit})\n{row}",
                    metadata={**page.metadata, "type": "table", "table_title": table.title, "row": row_no},
                ))
                row_no += 1

    if dropped:
        print(f"      검증에서 버린 행: {dropped}개 (원문에 없는 숫자 포함)")
    return row_docs, failed_pages


if __name__ == "__main__":
    from loader import load_all_pdfs

    pages = load_all_pdfs()
    table_pages = [p for p in pages if is_table_page(p)]

    print(f"\n표 페이지 {len(table_pages)}개:")
    for p in table_pages:
        print(f"  {p.metadata['file_name']} {p.metadata['page'] + 1}p | {TABLE_TITLE.search(p.page_content).group(0).strip()}")

    # 테스트는 7월 업데이트의 Table 1 한 페이지만 (전체 변환은 build_index.py에서)
    sample = [p for p in table_pages if p.metadata["file_name"] == "text2.pdf"][:1]
    rows, failed = convert_table_pages(sample)
    print(f"\n{sample[0].metadata['file_name']} {sample[0].metadata['page'] + 1}p → 행 {len(rows)}개")
    for r in rows:
        if any(k in r.page_content for k in ("World Output", "Japan", "Korea", "United States")):
            print("-", r.page_content.replace("\n", " | "))
