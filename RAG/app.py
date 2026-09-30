"""
소득세법 RAG 데모 — Streamlit 웹앱

실행:
    source .venv/bin/activate
    streamlit run app.py

income_tax_rag_demo.ipynb와 동일한 파이프라인(DOCX 인제스천 -> 청킹 -> 임베딩 하이브리드
검색 -> 규칙 기반 세금 계산 -> Gemini 무료 API로 근거 기반 설명 생성)을 웹 UI로 감쌌다.
무거운 부분(DOCX 파싱, 임베딩 모델 로드, 코퍼스 임베딩)은 st.cache_resource로 한 번만
실행되고 이후 요청부터는 재사용된다.
"""
import json
import os
import re
from pathlib import Path

import docx
import numpy as np
import streamlit as st
from docx.table import Table
from docx.text.paragraph import Paragraph
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

DOCX_PATH = Path("data/소득세법_통합.docx")
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"
GEMINI_MODEL = "gemini-3.5-flash-lite"  # 2026-07 기준 gemini-2.5-flash는 신규 사용자에게 차단됨

CIRCLED_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
CHUNK_MAX_CHARS = 900

CHAPTER_RE = re.compile(r"^제\d+(장|절|관)\b")
SOCIAL_INS_HEADING_RE = re.compile(r"^(?P<law>.+?)\s+(?P<article>제[^(]+)\((?P<topic>[^)]*)\)$")
INCOME_TAX_HEADING_RE = re.compile(r"^(?P<article>제[^(]+)\((?P<topic>[^)]*)\)$")

ITEM_QUERIES = {
    "근로소득공제": "근로소득공제",
    "세율": "세율",
    "근로소득세액공제": "근로소득세액공제",
    "국민연금": "국민연금료율",
    "건강보험": "건강보험료율",
    "장기요양보험": "장기요양보험료율",
    "고용보험": "고용보험료율",
    "지방소득세": "지방소득세",
}

TAX_BRACKETS = [
    (14_000_000, 0.06, 0),
    (50_000_000, 0.15, 1_260_000),
    (88_000_000, 0.24, 5_760_000),
    (150_000_000, 0.35, 15_440_000),
    (300_000_000, 0.38, 19_940_000),
    (500_000_000, 0.40, 25_940_000),
    (1_000_000_000, 0.42, 35_940_000),
    (float("inf"), 0.45, 65_940_000),
]


# ── DOCX 인제스천 ────────────────────────────────────────────
def table_to_text(table: Table) -> str:
    rows = [[c.text.strip() for c in row.cells] for row in table.rows]
    header, body_rows = rows[0], rows[1:]
    return "\n".join(" / ".join(f"{h}: {v}" for h, v in zip(header, row)) for row in body_rows)


def parse_docx(path: Path) -> list[dict]:
    document = docx.Document(path)
    articles: list[dict] = []
    current: dict | None = None
    current_위계 = None
    in_social_section = False

    def flush():
        nonlocal current
        if current is not None and current["text"].strip():
            articles.append(current)
        current = None

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            style = item.style.name if item.style else ""
            text = item.text.strip()
            if not text:
                continue
            if style == "Heading 1":
                flush()
                in_social_section = True
                current_위계 = None
                continue
            if style == "Title":
                continue
            if style == "Heading 3":
                flush()
                if in_social_section:
                    m = SOCIAL_INS_HEADING_RE.match(text)
                    law, article, topic = (m.group("law"), m.group("article"), m.group("topic")) if m else ("", text, "")
                else:
                    m = INCOME_TAX_HEADING_RE.match(text)
                    law, article, topic = ("소득세법", m.group("article"), m.group("topic")) if m else ("소득세법", text, "")
                current = {"law": law, "article": article, "topic": topic, "위계": current_위계, "text": ""}
                continue
            if CHAPTER_RE.match(text) and not in_social_section:
                current_위계 = text
                continue
            if current is not None:
                current["text"] += text + "\n"
        elif isinstance(item, Table):
            if current is not None:
                current["text"] += table_to_text(item) + "\n"

    flush()
    for a in articles:
        a["text"] = a["text"].strip()
    return articles


def split_into_chunks(text: str) -> list[str]:
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]
    parts = [p for p in re.split(f"(?=[{CIRCLED_NUMS}])", text) if p.strip()]
    if len(parts) <= 1:
        return [text[i:i + CHUNK_MAX_CHARS] for i in range(0, len(text), CHUNK_MAX_CHARS)]
    chunks, buf = [], ""
    for p in parts:
        if len(buf) + len(p) <= CHUNK_MAX_CHARS:
            buf += p
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def build_corpus(articles: list[dict]) -> list[dict]:
    corpus = []
    for idx, a in enumerate(articles):
        chunks = split_into_chunks(a["text"])
        for i, chunk_text in enumerate(chunks):
            corpus.append({
                "id": f"art{idx:03d}_{i}" if len(chunks) > 1 else f"art{idx:03d}",
                "law": a["law"], "article": a["article"], "topic": a["topic"],
                "text": chunk_text, "위계": a["위계"],
                "chunk_info": f"{i + 1}/{len(chunks)}" if len(chunks) > 1 else None,
            })
    return corpus


# ── 세금·4대보험 계산 ────────────────────────────────────────
def calc_근로소득공제(총급여: float) -> float:
    if 총급여 <= 5_000_000:
        공제 = 총급여 * 0.70
    elif 총급여 <= 15_000_000:
        공제 = 3_500_000 + (총급여 - 5_000_000) * 0.40
    elif 총급여 <= 45_000_000:
        공제 = 7_500_000 + (총급여 - 15_000_000) * 0.15
    elif 총급여 <= 100_000_000:
        공제 = 12_000_000 + (총급여 - 45_000_000) * 0.05
    else:
        공제 = 14_750_000 + (총급여 - 100_000_000) * 0.02
    return min(공제, 20_000_000)


def calc_산출세액(과세표준: float) -> float:
    if 과세표준 <= 0:
        return 0.0
    for 상한, 세율, 누진공제 in TAX_BRACKETS:
        if 과세표준 <= 상한:
            return max(과세표준 * 세율 - 누진공제, 0.0)
    return 0.0


# 법인세율표: (과세표준 상한, 세율, 누진공제) — 법인세법 제55조, 2026.1.1. 이후 개시 사업연도
# 국세청 공식 페이지 수치를 그대로 쓰되, 최고 구간 누진공제는 구간 경계 연속성을 직접
# 검산해 보정함(원 자료 9억4,200만원 → 94억2,000만원이 수학적으로 맞음).
CORP_TAX_BRACKETS = [
    (200_000_000, 0.10, 0),
    (20_000_000_000, 0.20, 20_000_000),
    (300_000_000_000, 0.22, 420_000_000),
    (float("inf"), 0.25, 9_420_000_000),
]


def calc_법인세(과세표준: float) -> float:
    """법인세법 제55조"""
    if 과세표준 <= 0:
        return 0.0
    for 상한, 세율, 누진공제 in CORP_TAX_BRACKETS:
        if 과세표준 <= 상한:
            return max(과세표준 * 세율 - 누진공제, 0.0)
    return 0.0


def to_kr_currency(amount: float) -> str:
    """숫자를 소득세법 조문에 나오는 표기 그대로 억/만원 단위 문자열로 바꾼다 (예: 17,406만원 -> "1억7,406만원")."""
    amount = round(amount)
    if amount == 0:
        return "0원"
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    억, 나머지 = divmod(amount, 100_000_000)
    만원 = round(나머지 / 10_000)
    if 만원 == 10_000:  # 나머지가 반올림으로 1억이 되는 경우 자리올림
        억, 만원 = 억 + 1, 0
    if 억 and 만원:
        return f"{sign}{억}억{만원:,}만원"
    if 억:
        return f"{sign}{억}억원"
    if 만원:
        return f"{sign}{만원:,}만원"
    return f"{sign}{amount:,}원"


def calc_구간_계산(과세표준: float, brackets: list[tuple[float, float, float]] = TAX_BRACKETS) -> dict | None:
    """법 조문이 실제로 쓰는 방식("기본세액 + 초과금액의 x퍼센트") 그대로 계산 과정을 풀어낸다.
    내부적으로 쓰는 calc_산출세액()의 "과세표준 × 세율 - 누진공제" 방식과 결과값은 항상 같다
    (기본세액 = 하한 × 세율 - 누진공제 로 정의되므로 수학적으로 동일한 계산)."""
    if 과세표준 <= 0:
        return None
    하한 = 0.0
    for 상한, 세율, 누진공제 in brackets:
        if 과세표준 <= 상한:
            기본세액 = max(하한 * 세율 - 누진공제, 0.0)
            초과금액 = 과세표준 - 하한
            가산세액 = 초과금액 * 세율
            구간 = f"{to_kr_currency(하한)} 초과 " if 하한 > 0 else ""
            구간 += f"{to_kr_currency(상한)} 이하" if 상한 != float("inf") else "(최고 구간)"
            return {
                "구간": 구간, "세율": 세율, "기본세액": 기본세액,
                "하한": 하한, "과세표준": 과세표준,
                "초과금액": 초과금액, "가산세액": 가산세액,
                "산출세액": 기본세액 + 가산세액,
            }
        하한 = 상한
    return None


def format_구간_table(detail: dict | None) -> dict | None:
    """calc_구간_계산() 결과를 render_answer가 표로 그릴 수 있는 {항목: [...], 내용: [...]} 형태로 변환."""
    if detail is None:
        return None
    return {
        "항목": ["적용 구간", "기본세액", "초과금액 (과세표준 − 구간 하한)", "가산세액 (초과금액 × 세율)",
                "산출세액 (기본세액 + 가산세액)"],
        "내용": [
            detail["구간"],
            to_kr_currency(detail["기본세액"]),
            f"{to_kr_currency(detail['과세표준'])} − {to_kr_currency(detail['하한'])} = "
            f"{to_kr_currency(detail['초과금액'])}",
            f"{to_kr_currency(detail['초과금액'])} × {detail['세율'] * 100:.0f}% = {to_kr_currency(detail['가산세액'])}",
            f"{to_kr_currency(detail['기본세액'])} + {to_kr_currency(detail['가산세액'])} = {to_kr_currency(detail['산출세액'])}",
        ],
    }


def format_과세표준_table(breakdown: dict) -> dict:
    """calculate_take_home_pay()가 돌려준 값들로 "과세표준이 왜 이 금액인지"를 뺄셈 과정
    그대로 풀어서 보여준다 (연간 총급여 -> 근로소득공제 차감 -> 기본공제·4대보험료 차감)."""
    인적공제수 = round(breakdown["기본공제(연)"] / 1_500_000)
    return {
        "항목": [
            "연간 총급여", "(−) 근로소득공제", "= 근로소득금액",
            f"(−) 기본공제 (인적공제 {인적공제수}명 × 150만원)", "(−) 4대보험료(연)", "= 과세표준",
        ],
        "내용": [
            to_kr_currency(breakdown["연간 총급여"]),
            to_kr_currency(breakdown["근로소득공제(연)"]),
            to_kr_currency(breakdown["근로소득금액(연)"]),
            to_kr_currency(breakdown["기본공제(연)"]),
            to_kr_currency(breakdown["4대보험료(연)"]),
            to_kr_currency(breakdown["과세표준(연)"]),
        ],
    }


def compare_biz_vs_corp(연간이익: float) -> dict:
    """개인사업자(종합소득세) vs 법인(법인세) 세부담 비교 — 법인 단계 세금만 비교.

    법인으로 하면 법인 단계 세금은 낮아 보여도, 대표자가 급여·배당으로 이익을 인출할
    때 근로소득세·배당소득세가 추가로 부과된다. 이 비교는 그 인출 단계 세금을 포함하지
    않으므로 "법인이 항상 유리하다"는 결론으로 직접 이어지지 않는다.
    """
    개인_산출세액 = calc_산출세액(연간이익)
    개인_지방소득세 = 개인_산출세액 * 0.10
    개인_세금합계 = 개인_산출세액 + 개인_지방소득세

    법인세 = calc_법인세(연간이익)
    법인_지방소득세 = 법인세 * 0.10  # 법인지방소득세 근사치(실제로는 별도 낮은 세율 구간 적용)
    법인_세금합계 = 법인세 + 법인_지방소득세

    return {
        "연간 이익": round(연간이익),
        "[개인사업자] 종합소득세(산출세액)": round(개인_산출세액),
        "[개인사업자] 지방소득세": round(개인_지방소득세),
        "[개인사업자] 세금 합계": round(개인_세금합계),
        "[개인사업자] 세후 이익": round(연간이익 - 개인_세금합계),
        "[법인] 법인세": round(법인세),
        "[법인] 법인지방소득세(근사치)": round(법인_지방소득세),
        "[법인] 세금 합계 (법인 단계만)": round(법인_세금합계),
        "[법인] 세후 이익 (전액 유보 시)": round(연간이익 - 법인_세금합계),
    }


def find_breakeven_income(lo: float = 100_000.0, hi: float = 1_000_000_000_000.0) -> float | None:
    """개인사업자 종합소득세 총액이 법인세 총액을 처음으로 넘어서는(=법인이 유리해지기
    시작하는) 연간 이익 지점을 이진탐색으로 찾는다.

    실제 계산(1만원 단위 스캔)으로 확인한 결과 교차는 단 한 번만 일어나고(약 2,600만원
    부근), 그 이후로는 법인이 계속 유리하게 유지된다 — 개인 소득세 최고세율(45%)이 법인
    최고세율(25%)보다 훨씬 높아서 소득이 커질수록 격차가 벌어지기 때문. 그래서 단순
    이진탐색으로 안전하게 찾을 수 있다.
    """
    def diff(x: float) -> float:
        c = compare_biz_vs_corp(x)
        return c["[개인사업자] 세금 합계"] - c["[법인] 세금 합계 (법인 단계만)"]

    if diff(lo) >= 0:
        return lo
    if diff(hi) < 0:
        return None  # 탐색 범위 내에서 교차점을 찾지 못함

    for _ in range(60):
        mid = (lo + hi) / 2
        if diff(mid) < 0:
            lo = mid
        else:
            hi = mid
    return hi


KOREAN_AMOUNT_RE_EOK = re.compile(r"(\d+(?:\.\d+)?)\s*억")
KOREAN_AMOUNT_RE_CHEONMAN = re.compile(r"(\d+(?:\.\d+)?)\s*천\s*만")
KOREAN_AMOUNT_RE_BAECKMAN = re.compile(r"(\d+(?:\.\d+)?)\s*백\s*만")
KOREAN_AMOUNT_RE_MAN = re.compile(r"(\d+(?:\.\d+)?)\s*만")
PLAIN_WON_RE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d{5,})\s*원")


def parse_korean_amount(text: str) -> float | None:
    """'10억', '1억 5000만원', '3000만원', '5백만원', '30,000,000원' 같은 다양한 한글/숫자
    금액 표현을 원 단위 숫자로 변환. 억 단위와 만 단위 표현은 함께 등장해도(예: '1억 5000만원')
    합산한다.
    """
    total = 0.0
    found = False

    m = KOREAN_AMOUNT_RE_EOK.search(text)
    if m:
        total += float(m.group(1)) * 100_000_000
        found = True

    m = KOREAN_AMOUNT_RE_CHEONMAN.search(text)
    if m:
        total += float(m.group(1)) * 10_000_000
        found = True
    else:
        m = KOREAN_AMOUNT_RE_BAECKMAN.search(text)
        if m:
            total += float(m.group(1)) * 1_000_000
            found = True
        else:
            m = KOREAN_AMOUNT_RE_MAN.search(text)
            if m:
                total += float(m.group(1)) * 10_000
                found = True

    if found:
        return total

    # 억/만 단위 한글 표현이 전혀 없으면 '30,000,000원'처럼 순수 숫자 표기를 시도
    m = PLAIN_WON_RE.search(text)
    if m:
        return float(m.group(1).replace(",", ""))

    return None


def is_biz_vs_corp_question(text: str) -> bool:
    has_corp = "법인" in text
    has_biz = any(k in text for k in ["사업자", "개인사업", "사업소득", "개인"])
    return has_corp and has_biz


def is_salary_question(text: str) -> bool:
    keywords = ["월급", "실수령액", "연봉", "세후", "월급여", "월 급여", "떼", "공제", "받는"]
    return any(k in text for k in keywords) and parse_korean_amount(text) is not None


ANNUAL_INDICATOR_RE = re.compile(r"연봉|연간|연\s*소득|연\s*수입|연\s*이익|연에|1년에|일\s*년에|년에|한\s*해에")
MONTHLY_INDICATOR_RE = re.compile(r"월급|월\s*급여|월\s*소득|월에|매달|매월")


def extract_monthly_salary(text: str) -> float | None:
    """질문에서 금액을 뽑아 월급 기준으로 환산한다.

    '연봉/연간/1년에/연에'처럼 연 단위임을 나타내는 표현이 있으면 12로 나눈다. '월급/월에'
    처럼 명시적으로 월 단위라고 밝힌 경우에만 그대로 월급으로 쓰고, 아무 단위 표현도 없으면
    (예: '3000만원 세금이 얼마야') 더 흔한 해석인 월급으로 기본 처리한다.
    """
    amount = parse_korean_amount(text)
    if amount is None:
        return None
    if ANNUAL_INDICATOR_RE.search(text):
        return amount / 12
    return amount


def calc_근로소득세액공제(산출세액: float, 총급여: float) -> float:
    if 산출세액 <= 1_300_000:
        공제 = 산출세액 * 0.55
    else:
        공제 = 715_000 + (산출세액 - 1_300_000) * 0.30
    if 총급여 <= 33_000_000:
        한도 = 740_000
    elif 총급여 <= 70_000_000:
        한도 = max(740_000 - (총급여 - 33_000_000) * 0.008, 660_000)
    elif 총급여 <= 120_000_000:
        한도 = max(660_000 - (총급여 - 70_000_000) * 0.5, 500_000)
    else:
        한도 = max(500_000 - (총급여 - 120_000_000) * 0.5, 200_000)
    return min(공제, 한도)


def calc_4대보험료(월급여: float) -> dict:
    국민연금_상한 = 5_900_000
    연금_기준소득 = min(월급여, 국민연금_상한)
    건강보험 = 월급여 * 0.03545
    return {
        "국민연금": round(연금_기준소득 * 0.045),
        "건강보험": round(건강보험),
        "장기요양보험": round(건강보험 * 0.1295),
        "고용보험": round(월급여 * 0.009),
    }


def calculate_take_home_pay(월급여: float, 인적공제수: int = 1) -> dict:
    총급여 = 월급여 * 12
    근로소득공제 = calc_근로소득공제(총급여)
    근로소득금액 = 총급여 - 근로소득공제

    사대보험_월 = calc_4대보험료(월급여)
    사대보험_연 = sum(사대보험_월.values()) * 12

    기본공제 = 인적공제수 * 1_500_000
    과세표준 = max(근로소득금액 - 기본공제 - 사대보험_연, 0)

    산출세액 = calc_산출세액(과세표준)
    근로소득세액공제 = calc_근로소득세액공제(산출세액, 총급여)
    결정세액 = max(산출세액 - 근로소득세액공제, 0)
    지방소득세 = 결정세액 * 0.10

    소득세_월 = 결정세액 / 12
    지방소득세_월 = 지방소득세 / 12
    공제총액_월 = 소득세_월 + 지방소득세_월 + sum(사대보험_월.values())

    return {
        "월급여(세전)": round(월급여),
        "연간 총급여": round(총급여),
        "근로소득공제(연)": round(근로소득공제),
        "근로소득금액(연)": round(근로소득금액),
        "기본공제(연)": round(기본공제),
        "4대보험료(연)": round(사대보험_연),
        "과세표준(연)": round(과세표준),
        "산출세액(연)": round(산출세액),
        "근로소득세액공제(연)": round(근로소득세액공제),
        "결정세액(연)": round(결정세액),
        "소득세(월)": round(소득세_월),
        "지방소득세(월)": round(지방소득세_월),
        **{f"{k}(월)": v for k, v in 사대보험_월.items()},
        "공제총액(월)": round(공제총액_월),
        "실수령액(월)": round(월급여 - 공제총액_월),
    }


# ── 캐시된 리소스 (모델/코퍼스는 최초 1회만 로드) ────────────
@st.cache_resource(show_spinner="법령 DOCX 파싱 + 임베딩 모델 로드 중 (최초 1회, 1~2분 소요)...")
def load_pipeline():
    articles = parse_docx(DOCX_PATH)
    corpus = build_corpus(articles)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    corpus_texts = [f"{d['topic']}: {d['text']}" for d in corpus]
    corpus_embeddings = embedder.encode(corpus_texts, normalize_embeddings=True)
    return corpus, embedder, corpus_embeddings


def retrieve(query: str, corpus, embedder, corpus_embeddings, top_k: int = 1, keyword_boost: float = 0.5):
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    scores = corpus_embeddings @ query_emb
    for i, doc in enumerate(corpus):
        if doc["topic"] == query:
            scores[i] += keyword_boost
        elif doc["topic"] and doc["topic"] in query:
            scores[i] += keyword_boost * 0.6
    top_idx = np.argsort(-scores)[:top_k]
    return [(corpus[i], float(scores[i])) for i in top_idx]


def format_citation(doc: dict) -> str:
    citation = f"[{doc['law']} {doc['article']}] {doc['text']}"
    if doc.get("chunk_info"):
        citation += f" (일부 발췌 {doc['chunk_info']})"
    return citation


TONE_INSTRUCTION = """말투 지침: 번호를 매긴 딱딱한 보고서 형식(1. 2. 3. 같은 목차, ### 소제목 여러 개, 불릿
나열)은 쓰지 마세요. 옆에서 설명해주는 친한 선배처럼, 자연스럽게 이어지는 문장으로
답변하세요. 핵심 결론(예: 실수령액, 손익분기점 금액)은 답변 맨 앞이나 첫 문장에서 바로
말하고, 그다음에 왜 그런지를 풀어서 설명하세요. 숫자는 **굵게** 강조하되, 전체 분량은
짧은 문단 1~2개(4~6문장) 정도로 간결하게 정리하세요."""

NUMBER_ACCURACY_INSTRUCTION = """숫자 정확성 지침: 답변에 등장하는 모든 금액(원 단위 숫자)은 아래 [계산
결과]에 있는 값을 한 글자도 바꾸지 말고 그대로 옮겨 적으세요. 절대로 직접 덧셈·뺄셈·나눗셈 등을
암산해서 새로운 숫자를 만들어내지 마세요 — 특히 최종 실수령액, 세금 합계, 손익분기점처럼 사용자가
가장 주목하는 숫자는 [계산 결과]에 있는 것과 정확히 일치해야 합니다. 필요한 숫자가 [계산 결과]에
없다면 만들어내지 말고 "제공된 정보에 없다"고 말하세요."""


def build_llm_prompt(월급여: float, breakdown: dict, evidence: dict) -> str:
    근거_텍스트 = "\n".join(f"- {format_citation(v['doc'])}" for v in evidence.values())
    return f"""당신은 친절한 세무 설명 도우미입니다. 아래 계산 결과와 법조문 근거를 바탕으로,
일반인이 이해하기 쉽게 한국어로 설명해 주세요. 반드시 아래 제공된 근거 조문만 인용하고,
제공되지 않은 정보는 추측하지 마세요. 마지막엔 이 계산이 교육용 단순화 모델이라는 점을
한 문장으로 짧게 덧붙이세요.

**꼭 짚어야 할 부분**: 사람들이 특히 헷갈려 하는 게 "과세표준(연)"이 왜 그 금액인지예요.
[계산 결과]의 연간 총급여에서 근로소득공제(연)를 뺀 게 근로소득금액(연)이고, 거기서
기본공제(연)와 4대보험료(연)를 추가로 뺀 게 과세표준(연)이라는 흐름을 한 문장으로 짧게
짚어주세요 — 그냥 "과세표준은 얼마예요"라고 결과만 말하지 말고, 총급여에서 어떤 항목들이
얼마씩 빠져서 그 금액이 됐는지가 드러나야 합니다.

{TONE_INSTRUCTION}

{NUMBER_ACCURACY_INSTRUCTION}

[사용자 정보]
월급(세전): {월급여:,.0f}원

[계산 결과]
{json.dumps(breakdown, ensure_ascii=False, indent=2)}

[근거 법조문]
{근거_텍스트}
"""


def build_biz_vs_corp_prompt(연간이익: float, comparison: dict, evidence: dict) -> str:
    근거_텍스트 = "\n".join(f"- {format_citation(v['doc'])}" for v in evidence.values())
    return f"""당신은 친절한 세무 설명 도우미입니다. 개인사업자와 법인 중 어느 쪽이 유리한지
고민하는 사용자에게, 아래 계산 결과와 법조문 근거를 바탕으로 이해하기 쉽게 설명해 주세요.
반드시 아래 제공된 근거 조문만 인용하고, 제공되지 않은 정보는 추측하지 마세요.

**중요**: 이 계산은 법인 단계의 세금(법인세)까지만 비교합니다. 법인으로 하면 법인 단계
세금은 낮아 보일 수 있지만, 대표자가 그 이익을 급여나 배당으로 실제로 인출할 때는
근로소득세 또는 배당소득세가 추가로 부과됩니다. 법인이 무조건 유리하다고 단정하지 말고,
이 비교는 "이익을 법인에 유보(재투자)할 경우"에 한정된 비교라는 점과, 실제 인출까지
고려한 총 세부담은 인출 방식·시기·배당 여부에 따라 달라진다는 점을 반드시 언급하세요.
그리고 이 계산이 교육용 단순화 모델이라는 점도 한 문장으로 짧게 덧붙이세요.

{TONE_INSTRUCTION}

{NUMBER_ACCURACY_INSTRUCTION}

[사용자 정보]
연간 이익(추정): {연간이익:,.0f}원

[계산 결과 — 개인사업자(종합소득세) vs 법인(법인세) 비교]
{json.dumps(comparison, ensure_ascii=False, indent=2)}

[근거 법조문]
{근거_텍스트}
"""


def build_breakeven_prompt(breakeven: float, comparison_around: dict, evidence: dict) -> str:
    근거_텍스트 = "\n".join(f"- {format_citation(v['doc'])}" for v in evidence.values())
    return f"""당신은 친절한 세무 설명 도우미입니다. "연간 이익이 얼마부터 법인이 개인사업자보다
유리할까?"를 궁금해하는 사용자에게, 아래 계산된 손익분기점과 근거 조문을 바탕으로 설명해
주세요. 반드시 아래 제공된 근거 조문만 인용하고, 제공되지 않은 정보는 추측하지 마세요.

**중요**: 이 손익분기점은 법인 단계의 세금(법인세)과 개인사업자의 종합소득세만 비교한
결과입니다. 법인으로 전환하면 법인 단계 세금은 낮아져도, 대표자가 그 이익을 급여나
배당으로 실제로 인출할 때는 근로소득세 또는 배당소득세가 추가로 부과됩니다. 그래서 실제
로는 이 손익분기점보다 더 높은 소득에서 법인 전환이 유리해지는 경우가 많습니다 — 이
계산은 "이익을 법인에 유보(재투자)하는 경우"에 한정된 참고치라는 점을 반드시 언급하세요.
그리고 이 계산이 교육용 단순화 모델이라는 점도 한 문장으로 짧게 덧붙이세요.

{TONE_INSTRUCTION}

{NUMBER_ACCURACY_INSTRUCTION}

[계산된 손익분기점]
연간 이익 약 {breakeven:,.0f}원부터 법인 단계 세금이 개인사업자 종합소득세보다 낮아지기
시작합니다 (그 이전에는 개인사업자가 세금이 더 적습니다).

[손익분기점 근처 비교 예시]
{json.dumps(comparison_around, ensure_ascii=False, indent=2)}

[근거 법조문]
{근거_텍스트}
"""


def build_free_question_prompt(question: str, evidence: list[dict]) -> str:
    근거_텍스트 = "\n".join(f"- {format_citation(doc)}" for doc in evidence)
    return f"""당신은 친절한 세무 설명 도우미입니다. 아래 검색된 법조문 근거만 사용해서
사용자의 질문에 한국어로 답변해 주세요. 근거에 없는 내용은 추측하지 말고, 답을 알 수
없으면 모른다고 솔직히 말하세요. 필요하면 인용한 조문을 자연스럽게 문장 속에서 언급하세요.
마지막엔 이 답변이 교육용 참고 정보이며 실제 세무 신고·상담은 세무 전문가나 국세청
홈택스를 통해 확인해야 한다는 점을 한 문장으로 짧게 덧붙이세요.

{TONE_INSTRUCTION}

[사용자 질문]
{question}

[검색된 근거 법조문]
{근거_텍스트}
"""


def explain_evidence_quality(doc: dict, query: str | None = None, score: float | None = None) -> str:
    """검색된 근거 조문을 얼마나 신뢰할 수 있는지 코드로 직접 판단해 설명한다(LLM의 자기
    평가에 의존하지 않음 — 실제 코사인 유사도/키워드 일치 여부를 그대로 보여준다)."""
    label = f"**{doc['law']} {doc['article']}** ({doc['topic']})"
    if query is not None and doc["topic"] == query:
        return f"✅ {label} — 찾고 있던 항목(\"{query}\")과 조문 제목이 정확히 일치합니다."
    if score is None:
        return f"ℹ️ {label}"
    if score >= 0.6:
        return f"✅ {label} — 질문과 관련성이 높습니다 (유사도 {score:.2f})."
    if score >= 0.4:
        return f"🟡 {label} — 관련 있을 가능성이 있습니다 (유사도 {score:.2f})."
    return f"⚠️ {label} — 관련성이 낮을 수 있습니다 (유사도 {score:.2f}) — 참고만 하세요."


def render_answer(
    prompt: str,
    headline: list[tuple[str, str]] | None = None,
    detail_table: dict | None = None,
    detail_title: str = "\U0001f4ca 계산 상세 보기",
    bracket_tables: list[tuple[str, dict]] | None = None,
    evidence: list[tuple[dict, float | None, str | None]] | None = None,
    extra_warning: str | None = None,
    must_include: list[str] | None = None,
):
    """AI 설명을 맨 위에 크게, 계산 상세와 근거 조문은 접힌 expander로 아래에 배치하는
    공통 렌더링 순서. raw 표를 답변보다 먼저 쏟아내지 않기 위한 것.

    must_include: 답변에 반드시 그대로 나와야 하는 핵심 숫자 문자열 목록(예: 실수령액).
    없으면 검증 없이 한 번만 호출한다.
    """
    if headline:
        cols = st.columns(len(headline))
        for col, (label, value) in zip(cols, headline):
            col.metric(label, value)

    st.markdown("#### \U0001f4ac 답변")
    if GEMINI_API_KEY:
        with st.spinner("생각하는 중..."):
            st.markdown(call_gemini_verified(prompt, must_include))
    else:
        st.info("Gemini API 키가 없어 AI 설명은 생략합니다 — 아래 계산 결과와 근거 조문을 참고하세요.")

    if extra_warning:
        st.warning(extra_warning)

    if detail_table:
        with st.expander(detail_title):
            st.table(detail_table)

    for title, bracket_table in bracket_tables or []:
        with st.expander(f"\U0001f9ee {title}"):
            st.table(bracket_table)

    if evidence:
        with st.expander("\U0001f50d 근거 조문 — 어떻게 찾았는지"):
            for doc, score, query in evidence:
                st.markdown(explain_evidence_quality(doc, query, score))
                st.caption(format_citation(doc))


# ── Streamlit UI ─────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # 프론트에는 노출하지 않음 — .env로만 설정


def call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt,
        # temperature=0에 가깝게 — 이 앱은 창작이 아니라 이미 계산된 숫자를 그대로 옮겨 적는
        # 작업이라, 온도가 높으면 자릿수가 비슷한 다른 숫자로 잘못 옮겨 적는 경우가 있었다.
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text


def call_gemini_verified(prompt: str, must_include: list[str] | None = None, max_tries: int = 3) -> str:
    """call_gemini()를 감싸서, 답변에 반드시 들어가야 할 핵심 숫자(예: 실수령액)가 그대로
    적혔는지 확인한다. temperature를 낮춰도 8~9자리 숫자를 옮겨 적다가 몇 자리를 틀리는
    경우가 있었기 때문 — 프롬프트 지침만으로는 100% 막지 못해 재시도로 한 번 더 검증한다.
    max_tries를 다 써도 못 맞추면 마지막 답변에 경고를 붙여 상단 지표 카드를 참고하라고 안내한다."""
    text = ""
    for _ in range(max_tries):
        text = call_gemini(prompt)
        if not must_include or all(n in text for n in must_include):
            return text
    return text + "\n\n⚠️ 위 설명 속 숫자가 실제 계산 결과와 다를 수 있습니다 — 상단 지표와 아래 계산 상세 표를 기준으로 확인해 주세요."


st.set_page_config(page_title="소득세법 RAG 데모", page_icon="\U0001f4b0")
st.title("\U0001f4b0 세금, 뭐든 물어보세요")
st.caption("RAG 데모 — 소득세법 349개 조문(법제처 원문) + 4대보험 + 법인세·부가가치세 핵심 조문 기반, LLM은 설명 생성에만 사용")

corpus, embedder, corpus_embeddings = load_pipeline()
st.caption(f"코퍼스: {len(corpus)}개 청크 로드 완료")
if not GEMINI_API_KEY:
    st.caption(
        "ℹ️ AI 설명 기능을 쓰려면 프로젝트 폴더의 `.env` 파일에 `GEMINI_API_KEY=...`를 "
        "설정하세요 ([무료 발급](https://aistudio.google.com/apikey)). 키가 없어도 계산 "
        "결과와 근거 조문은 표시됩니다."
    )

question = st.text_area(
    "무엇이든 물어보세요",
    placeholder="예: 월급 500만원인데 세금 얼마나 떼? / 10억 버는 사업자인데 법인으로 할지 고민돼",
    height=100,
)

if st.button("물어보기", type="primary") and question.strip():
    if is_biz_vs_corp_question(question):
        income = parse_korean_amount(question)
        세율_doc, 세율_score = retrieve("세율", corpus, embedder, corpus_embeddings, top_k=1)[0]
        법인세율_doc, 법인세율_score = retrieve("법인세율", corpus, embedder, corpus_embeddings, top_k=1)[0]
        evidence = [(세율_doc, 세율_score, "세율"), (법인세율_doc, 법인세율_score, "법인세율")]

        if income is not None:
            # 구체적인 금액이 있으면 그 금액 기준으로 비교
            comparison = compare_biz_vs_corp(income)
            render_answer(
                prompt=build_biz_vs_corp_prompt(
                    income, comparison,
                    {"세율": {"doc": 세율_doc}, "법인세율": {"doc": 법인세율_doc}},
                ),
                headline=[
                    ("[개인사업자] 세금 합계", f"{comparison['[개인사업자] 세금 합계']:,}원"),
                    ("[법인] 세금 합계 (법인 단계만)", f"{comparison['[법인] 세금 합계 (법인 단계만)']:,}원"),
                ],
                detail_table={"항목": list(comparison.keys()), "금액": [f"{v:,}" for v in comparison.values()]},
                detail_title="\U0001f4ca 계산 상세 보기 (연간 이익 인식값: "
                             f"{income:,.0f}원)",
                bracket_tables=[
                    t for t in [
                        ("개인사업자(종합소득세) — 세율 구간 계산식",
                         format_구간_table(calc_구간_계산(income, TAX_BRACKETS))),
                        ("법인(법인세) — 세율 구간 계산식",
                         format_구간_table(calc_구간_계산(income, CORP_TAX_BRACKETS))),
                    ] if t[1] is not None
                ],
                evidence=evidence,
                extra_warning="⚠️ 위 비교는 **법인 단계 세금만** 비교합니다. 법인 이익을 대표자가 "
                              "급여나 배당으로 인출하면 근로소득세·배당소득세가 추가로 붙습니다 — "
                              "법인이 무조건 유리하다는 뜻이 아닙니다.",
                must_include=[
                    f"{comparison['[개인사업자] 세금 합계']:,}", f"{comparison['[법인] 세금 합계 (법인 단계만)']:,}",
                ],
            )

        else:
            # 구체적인 금액이 없으면 "얼마부터 유리한가"로 해석해 손익분기점을 계산
            breakeven = find_breakeven_income()
            if breakeven is None:
                st.warning("탐색 범위 내에서 손익분기점을 찾지 못했습니다.")
            else:
                comparison_around = {
                    "더 적게 벌 때": compare_biz_vs_corp(breakeven * 0.8),
                    "손익분기점": compare_biz_vs_corp(breakeven),
                    "더 많이 벌 때": compare_biz_vs_corp(breakeven * 1.5),
                }
                # 표를 시나리오별로 3개 늘어놓지 않고, 핵심 항목만 뽑아 열로 비교
                summary_rows = ["연간 이익", "[개인사업자] 세금 합계", "[법인] 세금 합계 (법인 단계만)",
                                "[개인사업자] 세후 이익", "[법인] 세후 이익 (전액 유보 시)"]
                detail_table = {"항목": summary_rows}
                for label, comp in comparison_around.items():
                    detail_table[label] = [f"{comp[row]:,}" for row in summary_rows]

                render_answer(
                    prompt=build_breakeven_prompt(
                        breakeven, comparison_around,
                        {"세율": {"doc": 세율_doc}, "법인세율": {"doc": 법인세율_doc}},
                    ),
                    headline=[("손익분기점 (연간 이익)", f"{breakeven:,.0f}원")],
                    detail_table=detail_table,
                    detail_title="\U0001f4ca 손익분기점 근처 비교 상세",
                    evidence=evidence,
                    extra_warning="⚠️ 이 손익분기점은 **법인 단계 세금만** 비교한 결과입니다. 실제로 "
                                  "법인 이익을 급여·배당으로 인출하면 근로소득세·배당소득세가 추가로 "
                                  "붙기 때문에, 인출까지 고려하면 실제 손익분기점은 이보다 더 높을 "
                                  "수 있습니다.",
                    must_include=[f"{breakeven:,.0f}"],
                )

    elif is_salary_question(question):
        월급여 = extract_monthly_salary(question)
        breakdown = calculate_take_home_pay(월급여)
        evidence = [
            (retrieve(query, corpus, embedder, corpus_embeddings, top_k=1)[0][0], None, query)
            for item, query in ITEM_QUERIES.items()
        ]

        render_answer(
            prompt=build_llm_prompt(월급여, breakdown, {
                item: {"doc": retrieve(query, corpus, embedder, corpus_embeddings, top_k=1)[0][0]}
                for item, query in ITEM_QUERIES.items()
            }),
            headline=[
                ("실수령액 (월)", f"{breakdown['실수령액(월)']:,}원"),
                ("공제총액 (월)", f"{breakdown['공제총액(월)']:,}원"),
            ],
            detail_table={"항목": list(breakdown.keys()), "금액": [f"{v:,}" for v in breakdown.values()]},
            detail_title=f"\U0001f4ca 계산 상세 보기 (월급 인식값: {월급여:,.0f}원)",
            bracket_tables=[
                t for t in [
                    ("과세표준 산출 내역", format_과세표준_table(breakdown)),
                    ("소득세 — 세율 구간 계산식", format_구간_table(calc_구간_계산(breakdown["과세표준(연)"]))),
                ]
                if t[1] is not None
            ],
            evidence=evidence,
            must_include=[f"{breakdown['실수령액(월)']:,}"],
        )

    else:
        with st.spinner("관련 법조문 검색 중..."):
            results = retrieve(question, corpus, embedder, corpus_embeddings, top_k=5)
            evidence_docs = [doc for doc, _score in results]

        render_answer(
            prompt=build_free_question_prompt(question, evidence_docs),
            evidence=[(doc, score, None) for doc, score in results],
        )
