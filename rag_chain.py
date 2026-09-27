"""
rag_chain.py — 검색(Retriever) + 프롬프트 + GPT를 하나의 체인으로 연결하는 파일

RAG의 질의(query) 단계 전체. 대화 기억 포함.
질문 + 대화기록 ──▶ ⓪ 질문 재작성(영어) ──▶ ① 검색 ──▶ ② 프롬프트 조립 ──▶ ③ GPT 답변

    "한국 성장률은?" → (답변) → "그럼 일본은?"
                                   └─ ⓪에서 "What is the IMF's 2026 GDP growth projection for Japan?" 으로 바꾼 뒤 검색
"""

import re
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import CHAT_MODEL, TOP_K
from vectorstore import load_vectorstore


# ---------------------------------------------------------------
# 프롬프트 ⓪: 질문 → '검색용 영어 질문'으로 재작성
# ---------------------------------------------------------------
# 검색은 '질문 문장 하나'만 벡터로 바꿔서 비교한다. 여기서 두 가지 문제를 한 번에 해결한다.
#   1) 대화 맥락: "그럼 일본은?"만으로는 무엇을 묻는지 알 수 없다 → 맥락을 채워 넣은 독립 질문으로
#   2) 언어 차이: 보고서는 영어, 질문은 한국어 → 벡터가 덜 가깝게 나온다 → 영어로 번역
REWRITE_PROMPT = """You rewrite user questions into search queries for a vector database of IMF World Economic Outlook reports (written in English).

Using the chat history and the latest user question, write ONE standalone search query in English.
- Resolve references to earlier turns (e.g. "그럼 일본은?", "the first one") into explicit terms.
- Keep the same scope as the user's question. Do not add topics the user did not ask about.
- Never invent countries, years or numbers that do not appear in the question or chat history.
- Every document in the database is an IMF World Economic Outlook report, so do NOT add words like
  "IMF", "World Economic Outlook", "report" or "projections" — they match every document and dilute the query.
  Focus on what makes the question specific (country, topic, variable, year).
- Keep it short: at most 12 words. Prefer wording an IMF report would use.
- Do not answer the question. Output only the English query, without quotes."""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", REWRITE_PROMPT),
    MessagesPlaceholder("chat_history"),   # 이전 대화 메시지 목록이 이 자리에 들어간다
    ("human", "{question}"),
])


# ---------------------------------------------------------------
# 프롬프트 ②: GPT에게 주는 '업무 지시서'
# ---------------------------------------------------------------
SYSTEM_PROMPT = """너는 IMF 세계경제전망(World Economic Outlook) 보고서를 근거로 답하는 경제 분석 도우미다.

[규칙]
1. 아래 [참고 문서]에 있는 내용만 근거로 답한다. 문서에 없는 내용은 추측하지 말고 "제공된 보고서에서 찾을 수 없습니다"라고 답한다.
2. 수치를 말할 때는 어느 보고서(발간 시점)의 수치인지 밝힌다. 보고서마다 전망이 다르면 발간 순서대로 비교해서 설명한다.
3. 근거로 쓴 문장 끝에 [출처 번호]를 붙인다. 예: 세계 성장률은 3.3%로 전망된다 [1].
4. 한국어로 답한다.

[참고 문서]
{context}"""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),   # 이전 대화도 보여줘야 "아까 말한 그 수치" 같은 표현을 이해한다
    ("human", "{question}"),
])


# ---------------------------------------------------------------
# 검색된 청크 → 프롬프트용 텍스트
# ---------------------------------------------------------------
def report_name(doc: Document) -> str:
    """PDF 메타데이터의 title에서 보고서 이름만 뽑는다.

    'World Economic Outlook, April 2026; Global Economy in the Shadow of War' → 'World Economic Outlook, April 2026'
    """
    title = doc.metadata.get("title") or doc.metadata["file_name"]
    return re.split(r"[:;]", title)[0].strip()


def source_label(doc: Document) -> str:
    """'보고서명 | 13페이지' 형태. 표에서 온 행이면 '(표)'를 붙인다."""
    label = f"{report_name(doc)} | {doc.metadata['page'] + 1}페이지"
    if doc.metadata.get("type") == "table":
        label += " (표)"
    return label


def format_docs(docs: list[Document]) -> str:
    """청크마다 [번호: 보고서명 | 페이지] 머리표를 붙여서 하나의 문자열로 합친다."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        header = f"[{i}] {source_label(doc)}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------
# 체인 조립 (LCEL: '|' 로 부품을 파이프처럼 연결)
# ---------------------------------------------------------------
def build_rewrite_chain():
    """⓪ 질문 재작성기: {"question", "chat_history"} → 영어 검색 질문 문자열

    gpt-5-mini는 답하기 전에 '생각(reasoning)'을 하는 모델.
    minimal로 두면 약 0.9초로 빠르지만 질문에 없는 국가·연도를 지어내는 일이 잦았다. low는 약 1.7초지만 안정적이다.
    """
    llm = ChatOpenAI(model=CHAT_MODEL, reasoning_effort="low")
    return rewrite_prompt | llm | StrOutputParser()


def build_rag_chain(file_name: str | None = None):
    """RAG 체인을 만든다.

    입력: {"question": 질문 문자열, "chat_history": [HumanMessage, AIMessage, ...]}
    출력: {"question", "chat_history", "search_query", "docs", "answer"}

    file_name을 주면 그 PDF 안에서만 검색한다. (Chroma 메타데이터 필터)
    """
    search_kwargs = {"k": TOP_K}
    if file_name:
        search_kwargs["filter"] = {"file_name": file_name}

    retriever = load_vectorstore().as_retriever(search_kwargs=search_kwargs)
    rewrite_chain = build_rewrite_chain()

    def rewrite_if_needed(x: dict) -> str:
        # 대화 기록도 없고 이미 영어인 질문(한글이 한 글자도 없음)이면 API를 부르지 않고 그대로 쓴다
        if not x["chat_history"] and not re.search(r"[가-힣]", x["question"]):
            return x["question"]
        return rewrite_chain.invoke(x)

    # 답변은 low로 두어 속도와 비용을 아낀다 (문서 요약·정리에는 깊은 추론이 필요 없다)
    answer_llm = ChatOpenAI(model=CHAT_MODEL, reasoning_effort="low")

    # ②+③ 청크 → 프롬프트 → GPT → 문자열
    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | answer_prompt
        | answer_llm
        | StrOutputParser()
    )

    # 입력 딕셔너리에 한 단계씩 값을 덧붙여 나간다.
    return (
        RunnablePassthrough.assign(search_query=RunnableLambda(rewrite_if_needed))  # ⓪ + search_query
        .assign(docs=itemgetter("search_query") | retriever)                        # ① + docs (재작성된 질문으로 검색)
        .assign(answer=answer_chain)                                                # ②③ + answer
    )


if __name__ == "__main__":
    from langchain_core.messages import AIMessage, HumanMessage

    chain = build_rag_chain()
    history = []

    # 후속 질문이 맥락을 이어받는지 두 번 연달아 물어본다
    for question in ["한국의 2026년 경제 성장률 전망은?", "그럼 일본은?"]:
        result = chain.invoke({"question": question, "chat_history": history})

        print(f"질문     : {question}")
        print(f"검색 질문: {result['search_query']}")
        print(f"답변     : {result['answer']}")
        print("출처     :", ", ".join(source_label(d) for d in result["docs"]))
        print()

        history += [HumanMessage(question), AIMessage(result["answer"])]
