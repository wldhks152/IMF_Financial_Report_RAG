# 📈 IMF 세계경제전망 2026 RAG 챗봇

IMF **World Economic Outlook 2026** 보고서(1월 업데이트 · 4월 정식 보고서 · 7월 업데이트)를 근거로 질문에 답하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.
LangChain + OpenAI + Chroma로 만들고 Gradio로 웹에 띄웁니다.

- 답변마다 **출처(보고서 · 페이지)** 표시
- 보고서별 수치가 다르면 **발간 시점 순서대로 비교**
- **대화 기억**: "그럼 일본은?" 같은 후속 질문 가능
- **한국어 질문 → 영어 검색어로 재작성**해서 영어 보고서 검색 정확도 향상
- **표 처리**: 표의 각 행을 제목·열 이름이 붙은 문장으로 변환해서 국가별 수치 검색 가능

## 동작 구조

```
[색인: build_index.py — PDF가 바뀔 때만 실행]
data/*.pdf ─ loader.py ─┬─ 일반 페이지 ─ splitter.py (1000자 청크) ─┐
                        └─ 표 페이지 ─── tables.py (GPT로 행 → 문장) ─┴─ vectorstore.py ─▶ chroma_db/

[질의: app.py — 질문할 때마다]
질문 + 대화기록 ─▶ ⓪ 영어 검색어로 재작성 ─▶ ① Chroma 검색 (TOP 4) ─▶ ② 프롬프트 조립 ─▶ ③ GPT 답변 + 출처
```

| 파일 | 역할 |
|---|---|
| `config.py` | 경로, 모델명, 청크 크기, 검색 개수 등 설정값 |
| `loader.py` | `data/` 폴더의 PDF를 페이지 단위 Document로 읽기 |
| `splitter.py` | 페이지를 1000자(겹침 200자) 청크로 자르기 |
| `tables.py` | 표 페이지 감지 → GPT로 행 단위 문장 변환 → 숫자 검증 → 캐시 |
| `vectorstore.py` | OpenAI 임베딩 + Chroma DB 생성/불러오기 |
| `build_index.py` | 위 과정을 한 번에 실행해서 벡터 DB 생성 |
| `rag_chain.py` | 질문 재작성 + 검색 + 프롬프트 + GPT를 LCEL 체인으로 연결 |
| `app.py` | Gradio 채팅 화면 (스트리밍, 출처 표시, 보고서 선택) |
| `compare_search.py` | 한국어 검색 vs 영어 재작성 검색 비교 실험 |
| `table_cache.json` | 표 변환 결과 캐시 (저장소에는 없음 — 첫 `build_index.py` 실행 시 생성, 이후 GPT 재호출 없이 재사용) |

## 실행 방법

### 1. 설치 (Python 3.12 이상, Windows 기준)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
copy .env.example .env
```
`.env`를 열어 `OPENAI_API_KEY`에 본인 키를 넣습니다.

### 3. PDF 넣기

PDF는 저장소에 포함되어 있지 않습니다. [IMF World Economic Outlook](https://www.imf.org/en/Publications/WEO) 페이지에서 아래 보고서를 받아 `data/` 폴더에 넣으세요. (파일 이름은 자유)

- World Economic Outlook Update, January 2026
- World Economic Outlook, April 2026
- World Economic Outlook Update, July 2026

### 4. 벡터 DB 만들기 (PDF를 바꿨을 때만)

```bash
python build_index.py
```
처음 실행할 때는 표 페이지를 GPT로 변환하느라 4분 정도 걸립니다. 결과는 `table_cache.json`에 저장되어 다음부터는 빠릅니다.

### 5. 앱 실행

```bash
python app.py
```
브라우저에서 http://127.0.0.1:7860 접속

## 사용 모델

- 임베딩: `text-embedding-3-small`
- 답변 · 질문 재작성 · 표 변환: `gpt-5-mini`

## 알려진 한계

- 표 변환은 GPT가 하므로, 숫자를 **다른 열에 잘못 붙이는** 실수는 검증 단계에서 잡지 못할 수 있습니다. (원문에 없는 숫자는 걸러냄)
- 일부 표 행에서 실제값/전망치 구분 표시가 부정확할 수 있습니다.
- 페이지 머리글이 청크에 섞여 있고, 페이지 경계에서 문장이 끊깁니다.

## 출처

보고서 원문 © International Monetary Fund. 이 저장소는 학습 목적의 RAG 예제이며 IMF와 무관합니다.
