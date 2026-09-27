"""
config.py — 프로젝트 전체 설정값을 한곳에 모아둔 파일

다른 파일들은 여기서 값을 import 해서 쓴다.
(예: from config import PDF_PATH, CHUNK_SIZE)
실험하면서 값을 바꾸고 싶을 때 이 파일만 고치면 된다.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# .env 파일을 읽어서 OPENAI_API_KEY를 환경변수로 등록한다.
# langchain-openai는 이 환경변수를 자동으로 찾아 쓰기 때문에 코드에 키를 적을 필요가 없다.
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(".env 파일에 OPENAI_API_KEY가 없습니다.")


# ---------------------------------------------------------------
# 1. 경로
# ---------------------------------------------------------------
# __file__ = 이 파일(config.py)의 위치 → 어디서 실행하든 프로젝트 폴더 기준으로 경로가 잡힌다.
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"            # 원본 PDF를 넣어두는 폴더 (여기 있는 *.pdf를 전부 읽는다)
CHROMA_DIR = BASE_DIR / "chroma_db"     # 벡터 DB가 저장될 폴더 (build_index.py가 만든다)
COLLECTION_NAME = "imf_reports"         # Chroma 안의 '테이블' 이름 같은 것
TABLE_CACHE_PATH = BASE_DIR / "table_cache.json"  # GPT가 변환한 표 결과 저장 (다시 만들 때 API 재호출 방지)


# ---------------------------------------------------------------
# 2. 모델
# ---------------------------------------------------------------
# 임베딩 모델: 텍스트 → 숫자 벡터(1536차원)로 바꾸는 모델. 문서와 질문 모두 이 모델로 변환해야 비교가 가능하다.
EMBEDDING_MODEL = "text-embedding-3-small"

# 채팅 모델: 검색된 문서 조각을 읽고 최종 답변을 만드는 모델.
CHAT_MODEL = "gpt-5-mini"


# ---------------------------------------------------------------
# 3. 청크(Chunk) 설정 — 문서를 얼마나 잘게 자를지
# ---------------------------------------------------------------
# CHUNK_SIZE: 한 조각의 최대 글자 수.
#   너무 크면 → 한 조각에 여러 주제가 섞여서 검색이 부정확해진다.
#   너무 작으면 → 문맥이 잘려서 조각 하나만 보고는 의미를 알기 어렵다.
CHUNK_SIZE = 1000

# CHUNK_OVERLAP: 이웃한 조각끼리 겹치는 글자 수.
#   문장이 조각 경계에서 잘려도 앞뒤 조각 어딘가에는 온전히 남도록 하는 안전장치.
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------
# 4. 검색 설정
# ---------------------------------------------------------------
# 질문 하나당 가장 비슷한 조각을 몇 개 가져와 GPT에게 넘길지.
#   많을수록 → 정보는 풍부하지만 관련 없는 내용이 섞이고 비용이 늘어난다.
TOP_K = 4


# ---------------------------------------------------------------
# 5. 대화 기억 설정
# ---------------------------------------------------------------
# 최근 몇 번의 주고받음(질문+답변 = 1턴)까지 GPT에게 보여줄지.
#   많을수록 → 오래전 맥락까지 기억하지만, 매 질문마다 토큰(비용)이 늘어난다.
MAX_HISTORY_TURNS = 3


if __name__ == "__main__":
    # python config.py 로 직접 실행했을 때만 동작 (다른 파일에서 import 할 때는 실행 안 됨)
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    print("PDF 폴더     :", DATA_DIR, f"({len(pdfs)}개)")
    for p in pdfs:
        print("               -", p.name)
    print("벡터 DB 경로 :", CHROMA_DIR, f"(컬렉션: {COLLECTION_NAME})")
    print("임베딩 모델  :", EMBEDDING_MODEL)
    print("채팅 모델    :", CHAT_MODEL)
    print("청크 크기    :", CHUNK_SIZE, "/ 겹침:", CHUNK_OVERLAP)
    print("검색 개수    :", TOP_K)
    print("대화 기억    :", MAX_HISTORY_TURNS, "턴")
    print("API 키       : 설정됨")
