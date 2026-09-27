"""
app.py — Gradio로 RAG 챗봇을 웹 화면에 띄우는 파일

    python app.py   →  브라우저에서 http://127.0.0.1:7860 접속

화면 구성: 채팅창 + 보고서 선택 메뉴 + 예시 질문
동작: 질문 → rag_chain(검색 + GPT) → 답변을 한 글자씩 스트리밍 → 마지막에 출처 표시
"""

import gradio as gr
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import MAX_HISTORY_TURNS
from rag_chain import build_rag_chain, report_name, source_label
from vectorstore import load_vectorstore

ALL = "all"  # '전체 보고서'를 뜻하는 선택값
SOURCES_MARK = "\n\n---\n**📚 출처**"  # 답변 뒤에 붙이는 출처 구역의 시작 표시


# ---------------------------------------------------------------
# 1. 보고서 선택 메뉴 만들기 — DB에 저장된 metadata에서 보고서 목록을 뽑는다
# ---------------------------------------------------------------
def list_reports() -> list[tuple[str, str]]:
    """[(화면에 보일 이름, 실제 값)] 목록. 예: ("World Economic Outlook, April 2026", "text3.pdf")"""
    metadatas = load_vectorstore().get(include=["metadatas"])["metadatas"]

    # 청크 1010개의 metadata 중 파일별로 하나씩만 남긴다
    by_file = {m["file_name"]: m for m in metadatas}

    # PDF 생성일(creationdate) 순으로 정렬 → 1월, 4월, 7월 순서
    files = sorted(by_file.values(), key=lambda m: m.get("creationdate", ""))
    choices = [("전체 보고서", ALL)]
    choices += [(report_name(Document(page_content="", metadata=m)), m["file_name"]) for m in files]
    return choices


# ---------------------------------------------------------------
# 2. 체인 캐시 — 보고서 선택이 바뀔 때마다 새로 만들지 않도록 한 번 만든 체인은 보관
# ---------------------------------------------------------------
_chains = {}


def get_chain(report: str):
    if report not in _chains:
        _chains[report] = build_rag_chain(file_name=None if report == ALL else report)
    return _chains[report]


def format_sources(docs: list[Document], search_query: str, question: str) -> str:
    text = SOURCES_MARK + "\n\n"
    # 질문이 재작성됐으면 실제로 무엇으로 검색했는지 보여준다 (동작 원리 확인용)
    if search_query != question:
        text += f"🔎 검색 질문: *{search_query}*\n\n"
    text += "\n".join(f"{i}. {source_label(d)}" for i, d in enumerate(docs, start=1))
    return text


# ---------------------------------------------------------------
# 3. Gradio 대화 기록 → LangChain 메시지로 변환
# ---------------------------------------------------------------
def to_langchain_messages(history: list[dict]) -> list[BaseMessage]:
    """
    Gradio history : [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    LangChain      : [HumanMessage("..."), AIMessage("..."), ...]
    """
    messages = []
    for msg in history[-MAX_HISTORY_TURNS * 2:]:   # 최근 N턴만 (1턴 = 질문 + 답변 = 메시지 2개)
        content = msg["content"]
        # Gradio 6은 content를 [{"type": "text", "text": "..."}] 목록으로 줄 때도 있다
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))

        if msg["role"] == "user":
            messages.append(HumanMessage(content))
        elif msg["role"] == "assistant":
            # 화면에 붙여둔 출처 목록은 GPT에게 필요 없으니 잘라낸다 (토큰 절약)
            messages.append(AIMessage(content.split(SOURCES_MARK)[0]))
    return messages


# ---------------------------------------------------------------
# 4. 채팅 함수 — Gradio가 사용자가 질문할 때마다 호출한다
# ---------------------------------------------------------------
def respond(message: str, history: list, report: str):
    """
    message : 사용자가 방금 입력한 질문
    history : 이전 대화 목록 (Gradio가 자동으로 관리해서 넘겨준다)
    report  : 보고서 선택 메뉴의 현재 값 (additional_inputs로 연결됨)

    return 대신 yield를 쓰면 Gradio가 '지금까지의 답변'을 화면에 계속 갱신한다 → 스트리밍
    """
    chain = get_chain(report)
    inputs = {"question": message, "chat_history": to_langchain_messages(history)}

    docs, search_query, answer = [], message, ""
    # stream()은 결과를 조각조각 내보낸다:
    #   {"question"}, {"chat_history"} → {"search_query"} → {"docs"} → {"answer": "글자"} 여러 번
    for chunk in chain.stream(inputs):
        if "search_query" in chunk:
            search_query = chunk["search_query"]
        if "docs" in chunk:
            docs = chunk["docs"]
        if "answer" in chunk:
            answer += chunk["answer"]
            yield answer

    yield answer + format_sources(docs, search_query, message)


# ---------------------------------------------------------------
# 5. 화면 조립
# ---------------------------------------------------------------
report_dropdown = gr.Dropdown(choices=list_reports(), value=ALL, label="검색할 보고서")

demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[report_dropdown],   # respond()의 세 번째 인자로 전달된다
    title="📈 IMF 세계경제전망 2026 Q&A",
    description="IMF World Economic Outlook 2026 보고서(1월·4월·7월)를 근거로 답합니다. 답변 아래에 출처 페이지가 표시됩니다. "
                "이전 대화를 기억하므로 \"그럼 일본은?\" 같은 후속 질문도 할 수 있습니다.",
    examples=[
        ["2026년 세계 경제 성장률 전망은?", ALL],
        ["세계 경제의 주요 하방 리스크는 무엇인가요?", ALL],
        ["AI 투자가 성장에 어떤 영향을 주나요?", ALL],
        ["한국 경제 전망은 어떤가요?", ALL],
    ],
)


if __name__ == "__main__":
    demo.launch()
