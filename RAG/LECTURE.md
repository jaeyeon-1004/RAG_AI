# 소득세 RAG 데모, 어떻게 만들었나 — 코드 워크스루

법제처 원문 소득세법 349개 조문 + 4대보험 + 법인세·부가가치세 핵심 조문을 근거로 실수령액을
계산하고 설명하는 RAG 파이프라인. LangChain도, 벡터DB도 없다 — 왜 없이도 되는지, 그리고 그
과정에서 실제로 터진 버그들까지 코드 단위로 훑는다.

**파이프라인**: `소득세법.jsonl + 공식 PDF` → `preprocessing.ipynb` → `소득세법_통합.docx`
→ `income_tax_rag_demo.ipynb` / `app.py (Streamlit)` → `Gemini API (설명 생성만)`

---

## 목차

- [00 · 아키텍처 개요](#00--아키텍처-개요)
- [01 · preprocessing.ipynb](#01--preprocessingipynb)
- [02 · income_tax_rag_demo.ipynb](#02--income_tax_rag_demoipynb)
- [03 · app.py](#03--apppy)
- [04 · 발견한 버그 8개](#04--발견한-버그-8개)
- [05 · 예상 질문 대비](#05--예상-질문-대비)

---

## 00 · 아키텍처 개요

이 프로젝트는 의도적으로 **LangChain도, 벡터DB(FAISS/Chroma/Pinecone)도 쓰지 않는다**.
코퍼스가 357개 청크 규모라 `sentence-transformers`로 임베딩한 뒤 numpy 배열에 올려두고
코사인 유사도(`corpus_embeddings @ query_emb`)만 계산해도 충분히 빠르다. 이 규모에서
벡터DB를 넣는 건 오버엔지니어링이고, LangChain의 체인 추상화도 이 정도 로직(파싱 → 청킹
→ 검색 → 계산 → 프롬프트)엔 득보다 실이 크다 — 매 단계가 뭘 하는지 코드 3줄로 바로 보이는
편이 디버깅에 유리하다.

> 💡 **핵심 설계 원칙**: 정확성이 중요한 부분(세금 계산)은 코드에, 자연스러움이 중요한
> 부분(설명)은 LLM에 맡긴다. LLM은 세금을 계산하지 않는다 — 이미 계산된 결과와 검색된
> 조문을 받아 "설명"만 한다. 그래서 계산 결과에 환각이 낄 여지가 없다 (LLM이 숫자를
> *옮겨 적다가* 실수하는 문제는 남아있다 — 03-6에서 다룬다).

### 비용 구조 — 이 앱은 왜 무료인가

| 구성요소 | 사용 기술 | 실행 위치 | 비용 |
|---|---|---|---|
| DOCX 파싱 | `python-docx` | 로컬 | 무료 |
| 임베딩 | `ko-sroberta-multitask` | 로컬 CPU | 무료 |
| 벡터 검색 | numpy 코사인 + 키워드 하이브리드 | 로컬 | 무료 |
| 세금 계산 | 순수 파이썬 함수 | 로컬 | 무료 |
| 설명 생성 | Gemini 무료 티어 | API | 무료 (요청량 제한) |

원래는 Groq 무료 API로 시작했지만, 특정 네트워크에서 API 자체가 차단되는 걸 확인하고
(키 유무와 무관하게 `Access denied`) 접속이 확인된 Gemini로 전환했다.

---

## 01 · preprocessing.ipynb

**RAG가 읽어들일 원본 데이터(DOCX) 한 개를 만드는 노트북 · 5개 셀**

RAG 코퍼스의 소스가 두 갈래다: **소득세법 본문**은 사용자가 직접 업로드한
`data/소득세법.jsonl`(법제처 국가법령정보센터 원문, 349개 조문)을 쓰고, **4대보험·법인세·
부가가치세처럼 소득세법 파일에 없는 법령**은 이 노트북이 직접 만든 보충 PDF로 채운다.
최종 산출물은 `data/소득세법_통합.docx` 하나 — RAG 파이프라인은 이 파일만 읽는다.

### 1. 보충 조문 PDF 생성

law.go.kr의 조문 페이지는 자바스크립트로 렌더링되고, 다운로드 가능한 PDF URL도 예측
가능한 패턴이 없어 자동 스크래핑이 불가능하다. 그래서 검증된 조문 텍스트로 PDF를 **직접
구성**한다 — `reportlab`의 내장 CJK 폰트(`HYSMyeongJo-Medium`)를 쓰면 폰트 파일 없이
한글 PDF를 만들 수 있다.

```python
TAX_SUPPLEMENT_ENTRIES = [
    ("국민연금법", "제88조", "국민연금료율", "..."),
    ("국민건강보험법", "제73조", "건강보험료율", "..."),
    ("노인장기요양보험법", "제9조", "장기요양보험료율", "..."),
    ("고용보험법", "제13조 및 시행령", "고용보험료율", "..."),
    ("지방세법", "제92조", "지방소득세", "..."),
    ("법인세법", "제55조", "법인세율", "..."),
    ("법인세법", "제19조", "손금의 범위", "..."),
    ("부가가치세법", "제48조", "예정신고와 납부", "..."),
    ("부가가치세법", "제49조", "확정신고와 납부", "..."),
]

# PDF 안에 "■ 법령명 | 조번호 (조제목)" 헤더 + 본문 순서로 찍는다 —
# 이 고정 포맷을 나중에 정규식으로 다시 파싱해서 DOCX에 조문 단위로 넣는다.
for law, article, topic, text in TAX_SUPPLEMENT_ENTRIES:
    story.append(Paragraph(f"■ {law} | {article} ({topic})", styles["header"]))
    story.append(Paragraph(text, styles["body"]))
pdf_doc.build(story)
```

9개 항목 중 6개(4대보험·지방세·법인세율)는 세션 초반부터 있었고, 나머지 3개(법인세법
손금 조문, 부가가치세법 신고·납부 기한 2개)는 나중에 추가됐다 — 이유는 아래 버그 7번에서.

### 2. 소득세법 원문 적재·정제

```python
FUTURE_EFFECTIVE_RE = re.compile(r"\[시행일\s*:\s*\d{4}\.")

def dedupe_by_effective_date(articles: list[dict]) -> list[dict]:
    by_id = defaultdict(list)
    for d in articles:
        by_id[d["id"]].append(d)
    result = []
    for id_, versions in by_id.items():
        if len(versions) == 1:
            result.append(versions[0])
            continue
        # 같은 id가 여러 건이면 "아직 시행 안 된" 버전을 걸러내고 현재 버전만 채택
        current = [v for v in versions if not FUTURE_EFFECTIVE_RE.search(v["본문_원본"])]
        result.append(current[0] if current else versions[0])
    return result

deduped = dedupe_by_effective_date(raw_articles)
active_articles = [d for d in deduped if not d["삭제여부"]]
# 349 -> 326건(중복 제거) -> 259건(삭제조문 제외)
```

JSONL 349건 중 23건은 같은 조문 id가 두 번 등장한다 — 개정 전/후 버전이 함께 실려 있어서다.
`본문_원본`에 `[시행일: 2027...]` 같은 미래 날짜가 박혀 있으면 "아직 적용 안 되는 버전"으로
판단해 제외한다. 그다음 `삭제여부: true`인 67건(폐지 조문)도 뺀다. dict가 삽입 순서를 보존한다는
점을 이용해 **정렬을 따로 하지 않는다** — JSONL 원본 순서가 곧 법령 원문 순서이기 때문.

### 3. 표 이미지 직접 검증 결과 반영

제47조(근로소득공제)·제55조(종합소득세율)·제59조(근로소득세액공제) — 이 3개 조문은 원문
PDF에서 표가 **이미지로만** 렌더링돼 있어서 JSONL의 `본문` 필드에 실제 수치가 통째로 빠져
있었다. 표를 pdf2docx로 변환해 이미지로 추출한 뒤 **직접 육안으로 대조**해 값을 옮겨 적었다.

```python
TABLE_55 = [
    ("종합소득 과세표준", "세율"),
    ("1,400만원 이하", "과세표준의 6퍼센트"),
    ("1,400만원 초과 5,000만원 이하", "84만원 + (1,400만원을 초과하는 금액의 15퍼센트)"),
    # ... 8개 구간
    ("10억원 초과", "3억8,406만원 + (10억원을 초과하는 금액의 45퍼센트)"),
]

# id -> (표 앞 문단들, 표, 표 뒤 문단들) — 실제 조문 흐름 그대로 재조립
ARTICLE_BODY_OVERRIDES = {
    "소득세법_제47조": ([...], TABLE_47, [...]),
    "소득세법_제55조": ([...], TABLE_55, [...]),
    "소득세법_제59조": ([...], TABLE_59, [...]),
}
```

> 🐛 **이 검증에서 실제로 잡은 버그**: 제59조 ②항의 근로소득세액공제 한도 감소 계산식 —
> 총급여 7천만원 초과 구간의 비율을 기억(암묵지)에 의존해 **5%**로 가정하고 있었는데,
> 표 이미지를 직접 대조하고 웹 검색으로 교차 확인한 결과 실제로는 **1/2(50%)**이 맞았다.
> "그럴듯해 보이는 숫자"와 "원문에 실제로 적힌 숫자"는 다를 수 있다는 걸 보여준 사례.

### 4. DOCX 조립

소득세법 259개 조문(표 3개는 진짜 Word `Table` 객체로 원위치에 삽입) + 보충 조문 9개를
`data/소득세법_통합.docx` 하나로 저장한다. **편집자 주석이나 색상 표시는 넣지 않는다** —
RAG가 곧바로 읽어들일 원본이므로 실제 법령 문서 형태 그대로 유지하는 게 목표다.

```python
def parse_law_pdf(text: str) -> list[dict]:
    # PDF에서 뽑은 "■ 법령 | 조번호 (조제목)" 포맷을 정규식으로 역파싱
    blocks = re.split(r"(?m)^■\s*", text)
    header_re = re.compile(r"(?P<law>.+?)\s*\|\s*(?P<article>.+?)\s*\((?P<topic>[^)]*)\)\s*")
    ...

current_chapter = None
for a in active_articles:
    if a["장"] != current_chapter:            # 장 전환 감지
        current_chapter = a["장"]
        docx_doc.add_paragraph(current_chapter)
    docx_doc.add_heading(f"{a['조번호']}({a['조제목']})", level=3)   # Heading 3 = 조문 경계

    if a["id"] in ARTICLE_BODY_OVERRIDES:
        before, table_rows, after = ARTICLE_BODY_OVERRIDES[a["id"]]
        ...
        add_table(docx_doc, table_rows)        # 진짜 Word Table로 삽입
    else:
        for line in a["본문"].split("\n"):
            docx_doc.add_paragraph(line.strip())

docx_doc.add_heading("4대보험·지방소득세·법인세율 관련 법령 (보충)", level=1)  # Heading 1 = 섹션 전환
```

여기서 쓰는 `Heading 3`(조문 경계)과 `Heading 1`(소득세법→보충 법령 섹션 전환)이 다음
챕터의 인제스천 파서가 그대로 의존하는 구조적 신호다.

---

## 02 · income_tax_rag_demo.ipynb

**RAG 5단계를 순서대로 보여주는 교육용 노트북 · 10개 셀**

`data/소득세법_통합.docx` 하나를 읽어서 인제스천 → 청킹 → 검색 → 계산 → 생성까지 전체
파이프라인을 실행한다. 월급 3가지 예시(300/500/1000만원)로 끝까지 돌아간다.

### 1. 인제스천 — 문서를 원문 순서 그대로 읽기

`python-docx`의 `iter_inner_content()`가 핵심이다 — 문단(Paragraph)과 표(Table)를
**문서에 실제로 등장하는 순서 그대로** 순회한다. 미리 손질된 JSON을 읽는 게 아니라, 진짜
문서 구조(제목 스타일 · 문단 · 표)를 그대로 훑으며 조문 단위 청크를 재구성한다.

```python
for item in document.iter_inner_content():
    if isinstance(item, Paragraph):
        style = item.style.name if item.style else ""
        ...
        if style == "Heading 1":      # 소득세법 -> 보충 법령 섹션 전환
            flush(); in_social_section = True; continue
        if style == "Heading 3":      # 새 조문 시작
            flush()
            # in_social_section 여부에 따라 다른 정규식으로 (법령, 조번호, 조제목) 파싱
            current = {"law": law, "article": article, "topic": topic, ...}
            continue
        if CHAPTER_RE.match(text) and not in_social_section:
            current_위계 = text; continue   # "제1장 총칙" 같은 건 본문이 아니라 메타데이터
        current["text"] += text + "\n"
    elif isinstance(item, Table):
        current["text"] += table_to_text(item) + "\n"   # 표 linearize해서 본문에 통합
```

표는 `table_to_text()`가 헤더 행 기준으로 각 데이터 행을 `"헤더1: 값1 / 헤더2: 값2"`
텍스트로 풀어써서(linearize) 본문에 이어붙인다 — 임베딩 모델은 표 구조를 이해 못 하지만,
이렇게 풀어쓴 문장은 이해한다.

### 2. 청킹

법 조문은 길이 편차가 크다. 너무 긴 조문을 통째로 임베딩하면 검색 정확도가 떨어지므로,
항(①②③...) 단위로 쪼갠 뒤 너무 잘게 쪼개지지 않도록 900자 이하로 다시 묶는다.

```python
CIRCLED_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
CHUNK_MAX_CHARS = 900

def split_into_chunks(text: str) -> list[str]:
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]
    parts = [p for p in re.split(f"(?=[{CIRCLED_NUMS}])", text) if p.strip()]
    # 항 표시가 아예 없는 조문은 그냥 900자 단위로 강제 분할
    if len(parts) <= 1:
        return [text[i:i + CHUNK_MAX_CHARS] for i in range(0, len(text), CHUNK_MAX_CHARS)]
    chunks, buf = [], ""
    for p in parts:
        if len(buf) + len(p) <= CHUNK_MAX_CHARS:
            buf += p                 # 계속 이어붙일 수 있으면 이어붙임
        else:
            chunks.append(buf); buf = p
    ...
```

### 3. 하이브리드 검색 (Retrieval)

`ko-sroberta-multitask`로 코퍼스를 벡터화하고 코사인 유사도로 가장 관련 있는 청크를
찾는다. 하지만 **순수 임베딩만으로는 한계가 있다** — "근로소득공제"와 "근로소득세액공제"처럼
표현이 비슷한 조문은 코사인 유사도만으로 종종 헷갈린다. 그래서 조제목(topic) 일치 여부로
부스트를 주는 **키워드 하이브리드**를 섞는다.

```python
def retrieve(query: str, top_k: int = 3, keyword_boost: float = 0.5):
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    scores = corpus_embeddings @ query_emb   # 정규화됐으므로 내적 = 코사인 유사도
    for i, doc in enumerate(corpus):
        if doc["topic"] == query:
            scores[i] += keyword_boost              # 완전 일치 → 강한 부스트
        elif doc["topic"] and doc["topic"] in query:
            scores[i] += keyword_boost * 0.6         # 조제목이 질문 문구에 통째로 포함 → 약한 부스트
    top_idx = np.argsort(-scores)[:top_k]
    return [(corpus[i], float(scores[i])) for i in top_idx]
```

> 💡 **방향성이 비대칭인 이유**: `doc["topic"] in query`는 부스트하지만
> `query in doc["topic"]`는 절대 부스트하지 않는다. 반대로 했다면 "세율"이라는 짧은
> 질문이 "양도소득세의 세율"이라는 긴 조제목의 *부분 문자열*이라는 이유만으로 부스트를
> 받아, 훨씬 더 관련 있는 제55조(종합소득세율)를 밀어내는 오탐이 났을 것이다 (버그 2번).

### 4. 세금·4대보험료 계산 (Rule-based)

```python
def calc_근로소득세액공제(산출세액: float, 총급여: float) -> float:
    # 소득세법 제59조. ①항은 55%/30%, ②항 한도는 7천만원 초과부터 초과분의 1/2씩 감소
    if 산출세액 <= 1_300_000:
        공제 = 산출세액 * 0.55
    else:
        공제 = 715_000 + (산출세액 - 1_300_000) * 0.30

    if 총급여 <= 33_000_000:
        한도 = 740_000
    elif 총급여 <= 70_000_000:
        한도 = max(740_000 - (총급여 - 33_000_000) * 0.008, 660_000)
    elif 총급여 <= 120_000_000:
        한도 = max(660_000 - (총급여 - 70_000_000) * 0.5, 500_000)   # ← 1/2 (버그 수정 반영)
    else:
        한도 = max(500_000 - (총급여 - 120_000_000) * 0.5, 200_000)
    return min(공제, 한도)
```

`TAX_BRACKETS`는 `(과세표준 상한, 세율, 누진공제)` 튜플 리스트로 소득세법 제55조를
코드화한다. `calc_산출세액()`은 `과세표준 × 세율 − 누진공제` 방식(구간 스캔 한 번으로
끝나는 계산)을 쓰는데, 이건 법 조문이 실제로 쓰는 "기본세액 + 초과분×세율" 표현과
수학적으로 동일하다 — `app.py`에서 이 등가성을 이용해 사용자에게 법 조문 그대로의 표현을
다시 보여준다 (03-3).

### 5. 근거 기반 설명 생성

```python
def generate_explanation(월급여: float) -> str:
    breakdown = calculate_take_home_pay(월급여)
    evidence = gather_evidence()

    if gemini_client is None:
        # API 키 없으면 LLM 없이 계산 결과 + 근거만 정리해서 출력 (노트북은 끝까지 실행됨)
        ...
        return "\n".join(lines)

    prompt = build_llm_prompt(월급여, breakdown, evidence)
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text
```

`ITEM_QUERIES`는 계산 항목(근로소득공제, 세율...)을 코퍼스의 정식 조제목으로 매핑해둔
딕셔너리다 — 자유 서술형 질문이 아니라 "이 계산 항목의 근거가 필요하다"는 구조화된
조회라서, 하이브리드 부스트와 맞물려 비슷한 조문끼리 헷갈리지 않는다.

---

## 03 · app.py

**같은 파이프라인을 자연어 질문 하나로 감싼 Streamlit 웹앱 · ~850줄**

노트북의 인제스천/청킹/검색/계산 로직은 그대로 가져오고, 그 위에 **의도 분류**, **자연어
금액 파싱**, **UX 렌더링 순서**, **LLM 신뢰성 엔지니어링**을 얹었다 — 이 챕터가 세션
대부분의 반복 작업이 일어난 곳이다.

### 1. 의도 분류 — 분류 모델 없이 키워드로

```python
def is_biz_vs_corp_question(text: str) -> bool:
    has_corp = "법인" in text
    has_biz = any(k in text for k in ["사업자", "개인사업", "사업소득", "개인"])
    return has_corp and has_biz

def is_salary_question(text: str) -> bool:
    keywords = ["월급", "실수령액", "연봉", "세후", "월급여", "월 급여", "떼", "공제", "받는"]
    return any(k in text for k in keywords) and parse_korean_amount(text) is not None
```

3-way 라우팅(개인사업자 vs 법인 비교 → 급여 계산기 → 일반 RAG 폴백)을 분류 모델 없이
순수 키워드 휴리스틱으로 처리한다. 코퍼스가 소득세 도메인 하나로 좁고 질문 패턴이
한정적이라, 분류기를 학습·서빙하는 비용이 이 휴리스틱보다 이득이 크지 않다고 판단했다.

### 2. 자연어 금액 파싱 & 연/월 판별

```python
ANNUAL_INDICATOR_RE = re.compile(r"연봉|연간|연\s*소득|연\s*수입|연\s*이익|연에|1년에|일\s*년에|년에|한\s*해에")

def extract_monthly_salary(text: str) -> float | None:
    # '연봉/연간/1년에/연에'처럼 연 단위 표현이 있으면 12로 나눈다.
    # '월급/월에'처럼 명시적으로 월 단위인 경우만 그대로 쓰고,
    # 아무 단위 표현도 없으면 더 흔한 해석인 월급으로 기본 처리한다.
    amount = parse_korean_amount(text)
    if amount is None:
        return None
    if ANNUAL_INDICATOR_RE.search(text):
        return amount / 12
    return amount
```

> 🐛 **이 정규식이 넓어진 이유**: "연에 100억 버는데 세금 떼고 얼마 받아?"를 처음엔
> **월급 100억**으로 오인했다 — 초기 정규식이 "연봉"·"연 소득" 같은 정확한 문구만 잡고
> "연에"는 놓쳤기 때문. 자연어는 사용자가 실제로 말하는 방식을 다 받아야 해서, 정규식을
> 계속 넓혀가는 식으로 대응했다 (버그 6번).

### 3. 세율 구간 계산식 — 법 조문 그대로 보여주기

내부 계산은 `과세표준 × 세율 − 누진공제` 방식이 빠르지만, 사용자가 "왜 이 세금이
나왔는지" 검증하려면 법 조문 표현("84만원 + 초과분의 15%")이 훨씬 직관적이다. 두 표현이
수학적으로 동일하다는 걸 이용해, 사용자에게는 후자를 보여주는 **표시 전용 변환 레이어**를
하나 더 뒀다.

```python
def calc_구간_계산(과세표준: float, brackets=TAX_BRACKETS) -> dict | None:
    # 기본세액 = 하한 × 세율 − 누진공제 로 정의하면 calc_산출세액()과 결과가 항상 같다
    하한 = 0.0
    for 상한, 세율, 누진공제 in brackets:
        if 과세표준 <= 상한:
            기본세액 = max(하한 * 세율 - 누진공제, 0.0)
            초과금액 = 과세표준 - 하한
            가산세액 = 초과금액 * 세율
            return {
                "구간": ..., "기본세액": 기본세액, "하한": 하한, "과세표준": 과세표준,
                "초과금액": 초과금액, "가산세액": 가산세액, "산출세액": 기본세액 + 가산세액,
            }
        하한 = 상한
    return None
```

`to_kr_currency()`는 숫자를 법 조문 표기 그대로 억/만원 단위 문자열로 바꾼다(예:
`154377448` → `1억5,438만원`). 표에는 "초과금액 = 과세표준 − 구간 하한" 뺄셈 과정을
그대로 노출한다 — 처음엔 "초과금액"이라는 라벨만 있어서 사용자가 *구간 하한(3억원) 자체*가
초과금액인 줄 오해한 적이 있었다. 뺄셈식을 표에 직접 써주는 걸로 해결했다.

| 항목 | 내용 (과세표준 450,793,620원 예시) |
|---|---|
| 적용 구간 | 3억원 초과 5억원 이하 |
| 기본세액 | 9,406만원 |
| 초과금액 (과세표준 − 구간 하한) | 4억5,079만원 − 3억원 = 1억5,079만원 |
| 가산세액 (초과금액 × 세율) | 1억5,079만원 × 40% = 6,032만원 |
| 산출세액 (기본세액 + 가산세액) | 9,406만원 + 6,032만원 = 1억5,438만원 |

### 4. 개인사업자 vs 법인 비교 & 손익분기점

```python
def find_breakeven_income(lo=100_000.0, hi=1_000_000_000_000.0) -> float | None:
    # 1만원 단위 스캔으로 "교차가 딱 한 번만 일어난다"를 먼저 검증한 뒤에야
    # 이진탐색을 신뢰해서 썼다 — 다봉 함수였다면 이진탐색은 틀린 답을 냈을 것.
    def diff(x):
        c = compare_biz_vs_corp(x)
        return c["[개인사업자] 세금 합계"] - c["[법인] 세금 합계 (법인 단계만)"]
    ...
    for _ in range(60):
        mid = (lo + hi) / 2
        if diff(mid) < 0: lo = mid
        else: hi = mid
    return hi
```

개인 소득세 최고세율(45%)이 법인세 최고세율(25%)보다 훨씬 높아서, 소득이 커질수록 격차가
벌어지기만 하고 다시 좁혀지지 않는다 — 그래서 교차점이 하나뿐이라는 가정이 성립한다. 이
가정을 코드 주석이 아니라 **사전 스캔으로 실제 검증**한 다음 이진탐색을 적용한 게 포인트.

이 비교는 **법인 단계 세금만** 비교한다 — 대표자가 급여·배당으로 이익을 인출할 때 추가로
붙는 근로소득세/배당소득세는 포함하지 않는다. 그래서 모든 관련 출력에 이 경고를 강제로
붙인다(`extra_warning`).

### 5. render_answer — 답변이 먼저, 표는 나중에

```python
def render_answer(prompt, headline=None, detail_table=None, bracket_tables=None,
                   evidence=None, extra_warning=None, must_include=None):
    # AI 설명을 맨 위에 크게, 계산 상세·근거 조문은 접힌 expander로 아래에 배치.
    # raw 표를 답변보다 먼저 쏟아내지 않기 위한 렌더링 순서 고정.
    if headline:
        cols = st.columns(len(headline))
        for col, (label, value) in zip(cols, headline):
            col.metric(label, value)
    st.markdown("#### 💬 답변")
    st.markdown(call_gemini_verified(prompt, must_include))
    if extra_warning: st.warning(extra_warning)
    if detail_table:
        with st.expander(detail_title): st.table(detail_table)
    for title, bracket_table in bracket_tables or []:
        with st.expander(f"🧮 {title}"): st.table(bracket_table)
    if evidence:
        with st.expander("🔍 근거 조문 — 어떻게 찾았는지"): ...
```

> 💡 **렌더링 순서가 이렇게 굳어진 이유**: 원래는 raw 계산 표 3개를 답변 텍스트보다 먼저
> 늘어놓았다 — 사용자 피드백은 "뭐라는 건지 모르겠다"였다. 지금은 **지표 카드 → AI
> 답변(결론 먼저) → 계산 상세(접힘) → 근거(접힘)** 순서로 고정하고, 모든 라우팅 분기가
> 이 하나의 함수를 거치게 해서 UX가 갈라지지 않게 했다.

### 6. LLM 신뢰성 엔지니어링

계산은 100% 결정론적이지만, **LLM이 그 결과를 설명문으로 옮겨 적는 과정**에서도 오류가
생길 수 있다는 걸 뒤늦게 발견했다 — 8~9자리 숫자를 그대로 베끼라고 프롬프트에 명시해도,
가끔 자릿수를 틀리게 옮겨 적었다 (버그 8번). 세 겹으로 방어한다.

```python
NUMBER_ACCURACY_INSTRUCTION = """숫자 정확성 지침: 답변에 등장하는 모든 금액은
아래 [계산 결과]에 있는 값을 한 글자도 바꾸지 말고 그대로 옮겨 적으세요.
절대로 직접 덧셈·뺄셈·나눗셈 등을 암산해서 새로운 숫자를 만들어내지 마세요..."""

def call_gemini(prompt: str) -> str:
    ...
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt,
        # 창작이 아니라 계산된 숫자를 그대로 옮겨 적는 작업 → 온도를 낮춘다
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text

def call_gemini_verified(prompt, must_include=None, max_tries=3) -> str:
    # 답변에 핵심 숫자(예: 실수령액)가 정확히 들어있는지 확인 후 없으면 재시도.
    # 3번 다 실패하면(확률상 0.1~0.3% 수준) 경고문을 붙여 상단 지표를 보라고 안내.
    for _ in range(max_tries):
        text = call_gemini(prompt)
        if not must_include or all(n in text for n in must_include):
            return text
    return text + "\n\n⚠️ 위 설명 속 숫자가 다를 수 있습니다 — 상단 지표를 기준으로 확인하세요."
```

1. **프롬프트 지침**(`NUMBER_ACCURACY_INSTRUCTION`) — 계산하지 말고 그대로 옮기라고 명시
2. **temperature=0.1** — 창작 온도를 낮춰 자릿수 전사 오류 확률 자체를 낮춤
3. **검증-재시도**(`call_gemini_verified`) — 그래도 남는 확률적 오류를 결과 검사로 잡아냄

세 겹 다 프롬프트 엔지니어링만으론 부족했던 이유: 온도를 낮춰도 실패 확률이 완전히 0이
되지 않는(체감상 10~15%) 확률적 현상이었다 — 그래서 마지막 방어선은 반드시 **결과를
검사하는 코드**여야 했다.

---

## 04 · 발견한 버그 8개

세션 전체에서 실제로 터진 문제와 고친 방법 — 시간 순.

1. **Groq API 네트워크 차단** — 특정 네트워크에서 API 키 유무와 무관하게 `Access denied`가
   떴다. `curl`로 직접 찔러서 네트워크 레벨 차단임을 확인하고, 접속 가능한 Gemini 무료
   티어로 전환.

2. **하이브리드 검색 방향성 오탐** — "세율"로 검색하면 짧은 질의가 "양도소득세의
   세율"(제104조)의 부분 문자열이라는 이유로 부스트를 받아 더 관련 있는 제55조를 밀어냈다.
   `query in doc["topic"]` 방향의 부스트를 완전히 제거하고 `doc["topic"] in query` 방향만
   남겨 해결.

3. **근로소득세액공제 한도 비율 오기억 (5% → 1/2)** — 표 이미지 직접 대조 + 웹 검색 교차
   확인으로 잡음. "그럴듯한 숫자"가 아니라 원문 대조가 유일하게 믿을 수 있는 검증 방법이라는
   걸 보여준 사례.

4. **법인세 누진공제 10배 오류** — 국세청 페이지에서 가져온 최고 구간 누진공제가
   "9억4,200만원"이었는데, 구간 경계에서 세액이 끊기지 않는지 직접 검산
   (`300e9*0.22-420e6 == 300e9*0.25-X`)해보니 **94억2,000만원**이 수학적으로 맞았다 —
   출처 페이지 자체의 표기 오류로 추정.

5. **Gemini 모델 사용 중단 (gemini-2.5-flash)** — "신규 사용자에게 더 이상 제공되지
   않음" 404. 공식 모델 목록을 다시 확인해 `gemini-3.5-flash-lite`로 교체.

6. **"연에 100억"이 월급으로 오인됨** — 연 단위 인디케이터 정규식이 "연봉"·"연 소득" 같은
   정확한 문구만 잡고 "연에"는 놓쳤다. 자연어 표현을 계속 넓혀가며 `ANNUAL_INDICATOR_RE`를
   보강.

7. **코퍼스 커버리지 누락 (검색 알고리즘 문제로 오인)** — 사용자가 "부가가치세 신고
   기한", "법인 비용 인정 기준" 질문이 다른 세목 조문으로 잘못 검색된다며 리트리버 개선을
   제안했다. 실제로 `Counter(d['law'] for d in corpus)`를 찍어보니 **부가가치세법 0건,
   법인세법 1건(세율만)** — 데이터가 아예 없었다. 검색 알고리즘이 아니라 코퍼스 누락이
   원인이라는 걸 실측으로 반증하고, 원문 PDF를 직접 fetch해 조문 2~3개를 추가해 해결.

8. **LLM이 정답 숫자를 옮겨 적다가 실수** — 프롬프트에 정확한 `25224869`가 그대로 들어
   있는데도 AI 답변엔 `25,224,692`로 몇 자리가 바뀌어 나온 걸 발견 — 계산 버그가 아니라
   LLM의 자릿수 전사 오류. temperature 하향 + 검증-재시도 루프로 대응 (03-6).

---

## 05 · 예상 질문 대비

**벡터DB(FAISS/Chroma/Pinecone) 왜 안 썼어요?**
코퍼스가 357개 청크뿐이라 numpy 배열 하나(`corpus_embeddings`)에 다 올라간다. 검색은
`corpus_embeddings @ query_emb` 행렬곱 한 번 — 수만~수십만 청크로 늘어나면 그때 벡터DB를
도입할 이유가 생긴다. 지금 규모에서 벡터DB는 인프라 비용만 늘리는 오버엔지니어링이다.

**LangChain은요?**
이 파이프라인의 각 단계(파싱→청킹→검색→계산→프롬프트)가 서로 로직이 다르고 LLM 호출은
마지막 한 단계뿐이라, LangChain의 체인/에이전트 추상화가 코드를 더 읽기 쉽게 만들어주지
않는다. 오히려 각 함수가 뭘 하는지 3줄 안에 바로 보이는 지금 구조가 디버깅에 유리했다.

**세금 계산도 LLM한테 시키면 안 돼요?**
LLM은 확률적으로 그럴듯한 답을 생성하지, 정확한 산술을 보장하지 않는다 (버그 8번이 그
증거 — 이미 계산된 숫자를 *옮겨 적기만* 하는데도 가끔 틀렸다). 세금 계산처럼 정답이 하나로
고정된 문제는 결정론적 코드에 맡기고, LLM은 "그 결과를 사람 말로 풀어 설명"하는 역할만 준다.

**하이브리드 검색이 정확히 뭘 더하는 거예요?**
임베딩 코사인 유사도(dense) 위에 조제목 완전/포함 일치 시 점수를 더하는 키워드 부스트
(sparse)를 얹은 것 — 실무의 dense+sparse 결합(예: BM25+embedding)의 축소판이다. 단, 부스트
방향을 비대칭으로 설계한 게 포인트(버그 2번 참고) — 안 그러면 짧은 질의가 긴 조제목의
부분 문자열이라는 이유로 엉뚱하게 부스트된다.

**이거 실제 서비스로 쓸 수 있어요?**
아니다 — 교육/시연용이다. 4대보험 요율은 매년 바뀌고, 실제 원천징수는 국세청 근로소득
간이세액표를 따르며 연말정산으로 최종 정산된다. 부양가족 공제, 비과세소득 등 실제 변수도
반영 안 됐다. 실제 서비스로 확장하려면 (1) 남은 조문(양도소득세율 등)도 동일한 방식으로
원문 검증, (2) 요율 자동 업데이트, (3) 간이세액표 반영, (4) 조문 메타데이터 기반 질의
분류가 필요하다.

**변수·함수 이름이 왜 한글이에요?**
`과세표준`, `근로소득공제` 같은 법률 용어를 그대로 변수명으로 쓰면, 코드와 법 조문을
나란히 놓고 대조하기가 훨씬 쉽다. "이 변수가 법의 어느 개념에 대응하는가"를 번역 없이
바로 확인할 수 있다는 게 이 도메인(세법 계산)에서는 가독성 이득이 더 크다고 판단했다.
