"""
Lead Funnel Analytics API — Production-Ready
=============================================
Architecture:
  - LLM (IBM Watsonx / OpenAI-compatible) handles intent + date extraction
  - Rule-based regex acts as a fast-path fallback (never the primary path)
  - All date parsing goes through a unified DateResolver
  - All funnel computation goes through FunnelEngine
  - FastAPI endpoint is thin: parse → query → compute → respond
"""

from __future__ import annotations

import json
import logging
import os
import re
from json import JSONDecodeError
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import prestodb
from dotenv import load_dotenv
from fastapi import Body, FastAPI
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("lead_funnel.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("funnel_api")

# ─────────────────────────────────────────────
# Environment / Config
# ─────────────────────────────────────────────
CATALOG       = os.getenv("CATALOG", "salesforcereport")
FUNNEL_SCHEMA = os.getenv("FUNNEL_SCHEMA", "funnel")
FUNNEL_TABLE  = os.getenv("FUNNEL_TABLE", "sales_funnel_monthly")

PRESTO_HOST   = os.getenv("PRESTO_HOST")
PRESTO_PORT   = int(os.getenv("PRESTO_PORT", "443"))
PRESTO_USER   = os.getenv("PRESTO_USERNAME")
PRESTO_PASS   = os.getenv("PRESTO_PASSWORD")

WATSONX_URL        = os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
MODEL_ID           = os.getenv("MODEL_ID", "meta-llama/llama-3-2-3b-instruct")

# ─────────────────────────────────────────────
# LLM Client (Watsonx)
# ─────────────────────────────────────────────
_llm_model: Optional[ModelInference] = None

def _get_llm() -> Optional[ModelInference]:
    global _llm_model
    if _llm_model is not None:
        return _llm_model
    try:
        creds = Credentials(url=WATSONX_URL, api_key=PRESTO_PASS)
        _llm_model = ModelInference(
            model_id=MODEL_ID,
            credentials=creds,
            project_id=WATSONX_PROJECT_ID,
            params={"temperature": 0, "max_new_tokens": 1024},
        )
        logger.info("LLM client initialised: %s", MODEL_ID)
        return _llm_model
    except Exception as exc:
        logger.warning("LLM unavailable (%s) — falling back to regex", exc)
        return None


def llm_extract_intent(question: str) -> Dict[str, Any]:
    """
    Ask the LLM to parse the user question and return structured intent JSON.

    Returns a dict with keys:
        analysis_type : str  one of: single_period | mom | qoq | yoy |
                                      multi_month | multi_quarter | multi_year
        periods       : list of {"label", "start_date", "end_date"}  (YYYY-MM-DD)
        raw_question  : str  (echo)

    On any failure returns {"analysis_type": "unknown"} so the regex
    fallback is triggered.
    """
    model = _get_llm()
    if model is None:
        return {"analysis_type": "unknown"}

    today_str = datetime.today().strftime("%Y-%m-%d")
    current_fy = datetime.today().year if datetime.today().month >= 4 else datetime.today().year - 1

    system_prompt = f"""
You are a date-range extraction assistant for a sales funnel analytics system.
Today's date is {today_str}.  The financial year runs April 1 – March 31.
The current financial year starts April 1, {current_fy}.

Given a user question, return ONLY a valid JSON object (no markdown, no prose) with:
{{
  "analysis_type": "<type>",
  "periods": [
    {{"label": "<human label>", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}
  ]
}}

analysis_type must be exactly one of:
  single_period   – one date range
  mom             – month-on-month (one entry per calendar month)
  qoq             – quarter-on-quarter (one entry per quarter Q1–Q4 of FY)
  yoy             – year-on-year (one entry per financial year)
  multi_month     – discrete months (e.g. "april and june")
  multi_quarter   – discrete quarters (e.g. "Q1 and Q3")
  multi_year      – discrete or range of years

Rules:
- Dates always YYYY-MM-DD.
- Financial year quarters: Q1=Apr-Jun, Q2=Jul-Sep, Q3=Oct-Dec, Q4=Jan-Mar.
- **A bare year (e.g., "2025" or "2024") ALWAYS means a financial year, NOT a calendar year. So "2025" = April 1, 2025 to March 31, 2026.**
- "last week" = the previous completed Monday–Sunday week (end_date is the most recent Sunday).
- "last N weeks" = the last N completed Monday–Sunday weeks, excluding the current week.
- For week-based requests like "last week" or "past 4 weeks", return only completed Monday-Sunday weeks and never include the current partial week.
- "this week" = Monday of the current week to today (end_date = today, never a future date).
- "last day" / "last days" = today only.
- "last N days" = the last N days excluding today.
- "last month"  = full previous calendar month.
- "last N months" = the last N completed calendar months, excluding the current month.
- "this month"  = 1st of current month to today.
- "last quarter" = the most recently completed FY quarter.
- "last N quarters" = the last N completed FY quarters, excluding the current quarter.
- "this quarter" / "current quarter" = the FY quarter containing today, from its start date to today.
  Use FY quarter mapping: Q1=Apr–Jun, Q2=Jul–Sep, Q3=Oct–Dec, Q4=Jan–Mar.
  Example: today=2026-06-09 → Q1 FY2026 → start_date=2026-04-01, end_date=2026-06-09.
- "last year" / "last FY" = previous complete financial year.
- "last N years" = the last N completed financial years, excluding the current financial year.
- "MOM last year" / "last FY" = previous complete financial year per month.
- "this year" / "this FY" = April 1 {current_fy} to today.
- "MTD" = month to date (1st of current month to today).
- "YTD" = April 1 {current_fy} to today.
- For MOM: generate one period per month in the requested range.
- For QOQ: generate one period per quarter in the requested FY.
- For YOY: include the current FY-to-date by default, along with prior FYs for comparison.
- Exclude the current FY only when the user explicitly asks for "last year", "previous year", "last FY", or "last N years".
- For MOM Last Quarter: generate one period per month in the requested quarter.
- Supported natural-language date formats include but are not limited to:
    DD Month YYYY, DD/MM/YYYY, DD-MM-YYYY, Month YYYY,
    "5th June", "Q2 FY24", "FY2023", "2023-24", "last 30 days",
    "april to june 2024", "jan and march", "Q1 and Q3 last year".

Examples of valid JSON responses:
1. Question: "What was the lead funnel performance in April and June 2024?"
   JSON:
   {{
     "analysis_type": "multi_month",
     "periods": [
       {{"label": "Apr 2024", "start_date": "2024-04-01", "end_date": "2024-04-30"}},
       {{"label": "Jun 2024", "start_date": "2024-06-01", "end_date": "2024-06-30"}}
     ]
   }}
2. Question: "What was the lead funnel performance in last 3 years?"
   JSON:
   {{
     "analysis_type": "multi_year",
     "periods": [
      
       {{"label": "FY2023", "start_date": "2023-04-01", "end_date": "2024-03-31"}},
        {{"label": "FY2024", "start_date": "2024-04-01", "end_date": "2025-03-31"}},
       {{"label": "FY2025", "start_date": "2025-04-01", "end_date": "2026-03-31"}}
     ]
   }}

3. Question: "What was the lead funnel performance in last 3 quarters?"
   JSON:
   {{
     "analysis_type": "multi_quarter",
     "periods": [
       {{"label": "Q2 FY2025", "start_date": "2025-07-01", "end_date": "2025-09-30"}},
       {{"label": "Q3 FY2025", "start_date": "2025-10-01", "end_date": "2025-12-31"}},
       {{"label": "Q4 FY2025", "start_date": "2026-01-01", "end_date": "2026-03-31"}}
     ]
   }}

4. Question: "What was the lead funnel performance in last 3 months?"
   JSON:
   {{
     "analysis_type": "multi_month",
     "periods": [
       {{"label": "Feb 2026", "start_date": "2026-02-01", "end_date": "2026-02-28"}},
       {{"label": "Mar 2026", "start_date": "2026-03-01", "end_date": "2026-03-31"}},
       {{"label": "Apr 2026", "start_date": "2026-04-01", "end_date": "2026-04-30"}}
     ]
   }}

4b. Question: "show me lead funnel for this quarter"  (asked on any date in April–June, e.g. 2026-06-09)
   JSON:
   {{
     "analysis_type": "single_period",
     "periods": [
       {{"label": "Q1 FY2026 (QTD)", "start_date": "2026-04-01", "end_date": "2026-06-09"}}
     ]
   }}

   REMINDER — FY quarter boundaries (Q1=Apr-Jun, Q2=Jul-Sep, Q3=Oct-Dec, Q4=Jan-Mar):
   - If today is in April, May, or June  → current quarter is Q1 of current FY
   - If today is in July, Aug, or Sep    → current quarter is Q2 of current FY
   - If today is in Oct, Nov, or Dec     → current quarter is Q3 of current FY
   - If today is in Jan, Feb, or Mar     → current quarter is Q4 of previous FY

4c. Question: "show me lead funnel for this week"  (asked on 2026-06-09, a Tuesday)
   JSON:
   {{
     "analysis_type": "single_period",
     "periods": [
       {{"label": "This Week", "start_date": "2026-06-08", "end_date": "2026-06-09"}}
     ]
   }}

   REMINDER — "this week" = Monday of the current week to today (never future dates).
   "last week" = full Monday–Sunday of the immediately preceding week.

5. Question: "Show me mom lead funnel for last year?"
   JSON:
   {{
     "analysis_type": "mom",
     "periods": [
       {{"label": "Apr 2025", "start_date": "2025-04-01", "end_date": "2025-04-30"}},
       {{"label": "May 2025", "start_date": "2025-05-01", "end_date": "2025-05-31"}},
       {{"label": "Jun 2025", "start_date": "2025-06-01", "end_date": "2025-06-30"}},
       {{"label": "Jul 2025", "start_date": "2025-07-01", "end_date": "2025-07-31"}},
       {{"label": "Aug 2025", "start_date": "2025-08-01", "end_date": "2025-08-31"}},
       {{"label": "Sep 2025", "start_date": "2025-09-01", "end_date": "2025-09-30"}},
       {{"label": "Oct 2025", "start_date": "2025-10-01", "end_date": "2025-10-31"}},
       {{"label": "Nov 2025", "start_date": "2025-11-01", "end_date": "2025-11-30"}},
       {{"label": "Dec 2025", "start_date": "2025-12-01", "end_date": "2025-12-31"}},
       {{"label": "Jan 2026", "start_date": "2026-01-01", "end_date": "2026-01-31"}},
       {{"label": "Feb 2026", "start_date": "2026-02-01", "end_date": "2026-02-28"}},
       {{"label": "Mar 2026", "start_date": "2026-03-01", "end_date": "2026-03-31"}}
     ]
   }}

6. Question: "Show me qoq lead funnel for last year?"
   JSON:
   {{
     "analysis_type": "multi_quarter",
     "periods": [
       {{"label": "Q1 2025", "start_date": "2025-04-01", "end_date": "2025-06-30"}},
       {{"label": "Q2 2025", "start_date": "2025-07-01", "end_date": "2025-09-30"}},
       {{"label": "Q3 2025", "start_date": "2025-10-01", "end_date": "2025-12-31"}},
       {{"label": "Q4 2025", "start_date": "2026-01-01", "end_date": "2026-03-31"}}
     ]
   }}

7. Question: "What was the lead funnel performance in year on year?"
   JSON:
   {{
     "analysis_type": "multi_year",
     "periods": [
      
     {{"label": "FY2020", "start_date": "2020-04-01", "end_date": "2021-03-31"}},
     {{"label": "FY2021", "start_date": "2021-04-01", "end_date": "2022-03-31"}},
     {{"label": "FY2022", "start_date": "2022-04-01", "end_date": "2023-03-31"}},
       {{"label": "FY2023", "start_date": "2023-04-01", "end_date": "2024-03-31"}},
        {{"label": "FY2024", "start_date": "2024-04-01", "end_date": "2025-03-31"}},
       {{"label": "FY2025", "start_date": "2025-04-01", "end_date": "2026-03-31"}},
       {{"label": "FY2026", "start_date": "2026-04-01", "end_date": "2027-03-31"}}
     ]
   }}

8. Question: "What was the lead funnel performance in last 5 days?"
   JSON:
   {{
     "analysis_type": "last_n_days",
     "periods": [
       {{"label": "last 5 days", "start_date": "2026-05-17", "end_date": "2026-05-21"}},
     ]
   }}



""".strip()

    user_prompt = f'Question: "{question}"\n\nJSON:'

    try:
        raw = model.generate_text(prompt=f"{system_prompt}\n\n{user_prompt}")
        raw = raw.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            parsed = json.loads(raw)
        except JSONDecodeError:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(raw)
        parsed["raw_question"] = question
        logger.info("LLM intent: %s", parsed.get("analysis_type"))
        return parsed
    except Exception as exc:
        logger.warning("LLM parse error (%s) — falling back to regex", exc)
        return {"analysis_type": "unknown"}


# ─────────────────────────────────────────────
# Presto Helper
# ─────────────────────────────────────────────
def query_presto(catalog: str, schema: str, sql: str) -> pd.DataFrame:
    logger.info("Presto → %s.%s", catalog, schema)
    logger.debug("SQL:\n%s", sql)
    print(sql)
    try:
        conn = prestodb.dbapi.connect(
            host=PRESTO_HOST,
            port=PRESTO_PORT,
            user=PRESTO_USER,
            catalog=catalog,
            schema=schema,
            http_scheme="https",
            auth=prestodb.auth.BasicAuthentication(PRESTO_USER, PRESTO_PASS),
        )
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        logger.info("Fetched %d rows", len(df))
        return df
    except Exception as exc:
        logger.error("Presto error: %s", exc, exc_info=True)
        raise


def fetch_funnel_data(start: date, end: date) -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM {CATALOG}.{FUNNEL_SCHEMA}.{FUNNEL_TABLE}
        WHERE DATE(period_month)
              BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'
    """
    return query_presto(CATALOG, FUNNEL_SCHEMA, sql)


# ─────────────────────────────────────────────
# Funnel Engine
# ─────────────────────────────────────────────
# Fields that must NOT be summed (ratios / percentages)
_NON_ADDITIVE = {"%", ":"}


def _is_additive(key: str) -> bool:
    return not any(m in key for m in _NON_ADDITIVE)


def compute_funnel(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate a funnel DataFrame into a typed metrics dict.
    All column names are normalised to lower-snake so the function is
    resilient to minor schema changes (e.g. trailing spaces, mixed case).
    """
    if df is None or df.empty:
        return {}

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    def _sum(col: str) -> int:
        return int(df[col].sum()) if col in df.columns else 0

    def _pct(num: int, den: int) -> str:
        return f"{round(num / den * 100, 1)}%" if den else "0.0%"

    def _ratio(num: int, den: int) -> float:
        return round(num / den, 2) if den else 0.0

    tl  = _sum("total_leads")
    vl  = _sum("valid_leads")
    jl  = _sum("junk_leads")
    sol = _sum("sol_leads_interested")
    mb  = _sum("meeting_booked")
    md  = _sum("meeting_done")
    sd  = _sum("sales_done")

    return {
        "Total Leads":            tl,
        "Valid Leads":            vl,
        "Junk Leads":             jl,
        "SOL Leads (Interested)": sol,
        "Meeting Booked":         mb,
        "Meeting Done":           md,
        "Sales Done":             sd,
        # — percentages —
        "Junk %":   _pct(jl, tl),
        # — conversion ratios —
        "TL:VL":    _ratio(tl, vl),
        "VL:SOL":   _ratio(vl, sol),
        "SOL:MB":   _ratio(sol, mb),
        "MB:MD":    _ratio(mb, md),
        "MD:SD":    _ratio(md, sd),
        "TL:SD":    _ratio(tl, sd),
        "VL:SD":    _ratio(vl, sd),
        "SOL:SD":   _ratio(sol, sd),
        "MB:SD":    _ratio(mb, sd),
    }


def aggregate_funnels(funnels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Column-wise sum of additive metrics across multiple funnel dicts."""
    totals: Dict[str, float] = {}
    for f in funnels:
        for k, v in f.items():
            if _is_additive(k) and isinstance(v, (int, float)):
                totals[k] = totals.get(k, 0.0) + v
    return {
        k: int(v) if float(v).is_integer() else round(v, 2)
        for k, v in totals.items()
    }


# ─────────────────────────────────────────────
# Date Utilities  (used as fallback only)
# ─────────────────────────────────────────────

def _today() -> datetime:
    return datetime.today()


def _current_fy() -> int:
    t = _today()
    return t.year if t.month >= 4 else t.year - 1


def _fy_quarter(month: int) -> int:
    """
    Determine fiscal quarter (1-4) from calendar month.
    
    Fiscal year: April 1 – March 31
    - Q1: April (4), May (5), June (6)
    - Q2: July (7), August (8), September (9)
    - Q3: October (10), November (11), December (12)
    - Q4: January (1), February (2), March (3)
    
    Args:
        month: Calendar month (1-12)
    
    Returns:
        Quarter number (1-4)
        
    Example:
        June 9, 2026 → month=6 → Q1 ✓
    """
    for quarter, rng in enumerate((range(4, 7), range(7, 10), range(10, 13)), start=1):
        if month in rng:
            return quarter
    return 4


def _quarter_dates(q: int, fy: int) -> Tuple[date, date]:
    """
    Return (start_date, end_date) for a fiscal quarter.
    
    Fiscal year numbering:
        FY2026 = April 1, 2026 – March 31, 2027
        FY2025 = April 1, 2025 – March 31, 2026
    
    Quarter structure within FY:
        Q1 (fy): Apr 1 (fy) – Jun 30 (fy)
        Q2 (fy): Jul 1 (fy) – Sep 30 (fy)
        Q3 (fy): Oct 1 (fy) – Dec 31 (fy)
        Q4 (fy): Jan 1 (fy+1) – Mar 31 (fy+1)
    
    Args:
        q: Quarter (1-4)
        fy: Fiscal year as 4-digit number (e.g., 2026)
    
    Returns:
        Tuple of (start_date, end_date) as date objects
        
    Example:
        Q1 FY2026: (2026-04-01, 2026-06-30) ✓
        Q4 FY2025: (2026-01-01, 2026-03-31) ✓
    """
    mapping = {
        1: (date(fy, 4, 1),      date(fy, 6, 30)),
        2: (date(fy, 7, 1),      date(fy, 9, 30)),
        3: (date(fy, 10, 1),     date(fy, 12, 31)),
        4: (date(fy + 1, 1, 1),  date(fy + 1, 3, 31)),
    }
    return mapping[q]


def _month_end(year: int, month: int) -> date:
    return date(year, month, monthrange(year, month)[1])


MONTH_MAP: Dict[str, int] = {
    **{m: i + 1 for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
    )},
    **{m: i + 1 for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december"]
    )},
    "sept": 9,
}

WORD_NUM: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

_MONTH_RE = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)"
)


# ── Keyword-based intent detection (fast path before regex) ──────────────────

class AnalysisIntent(str, Enum):
    SINGLE   = "single_period"
    MOM      = "mom"
    QOQ      = "qoq"
    YOY      = "yoy"
    MULTI_M  = "multi_month"
    MULTI_Q  = "multi_quarter"
    MULTI_Y  = "multi_year"
    UNKNOWN  = "unknown"


# Keyword → intent mapping (order matters — more-specific first)
_INTENT_KEYWORDS: List[Tuple[AnalysisIntent, List[str]]] = [
    (AnalysisIntent.YOY,    ["yoy", "year on year", "year-on-year", "year over year",
                              "yearly comparison", "yoy performance", "year-wise", "year wise"]),
    (AnalysisIntent.QOQ,    ["qoq", "quarter on quarter", "quarter-on-quarter",
                              "quarter wise", "quarter-wise", "quarterly trend",
                              "quarterly comparison", "quarterwise", "quater wise",
                              "quarter over quarter"]),
    (AnalysisIntent.MOM,    ["mom", "month on month", "month-on-month",
                              "monthly trend", "month wise", "month-wise",
                              "monthly comparison", "month over month", "month-over-month"]),
]

_COMPARISON_KEYWORDS: Dict[AnalysisIntent, List[str]] = {
    AnalysisIntent.MOM: ["mom", "month on month", "month-on-month", "month over month"],
    AnalysisIntent.QOQ: ["qoq", "quarter on quarter", "quarter-on-quarter"],
    AnalysisIntent.YOY: ["yoy", "year on year", "year-on-year", "year over year"],
}

_MULTI_Q_RE = re.compile(r"q[1-4].{0,20}(and|,).{0,20}q[1-4]", re.I)
_MULTI_M_RE = re.compile(
    _MONTH_RE[1:-1] + r".{0,30}(and|,).{0,30}" + _MONTH_RE[1:-1],
    re.I
)
_MULTI_Y_RE = re.compile(
    r"(?:fy\s*)?(?:20\d{2}|\d{2}).{0,20}(and|,).{0,20}(?:fy\s*)?(?:20\d{2}|\d{2})",
    re.I,
)


def detect_intent_from_keywords(question: str) -> AnalysisIntent:
    q = question.lower()
    for intent, kws in _INTENT_KEYWORDS:
        if any(kw in q for kw in kws):
            return intent
    if _MULTI_Q_RE.search(q):
        return AnalysisIntent.MULTI_Q
    if _MULTI_M_RE.search(q):
        return AnalysisIntent.MULTI_M
    if _MULTI_Y_RE.search(q):
        return AnalysisIntent.MULTI_Y
    return AnalysisIntent.UNKNOWN


def detect_comparison_intents(question: str) -> List[AnalysisIntent]:
    q = question.lower()
    found: List[AnalysisIntent] = []
    for intent, kws in _COMPARISON_KEYWORDS.items():
        if any(kw in q for kw in kws):
            found.append(intent)
    return found


def detect_quarter_keywords(question: str) -> bool:
    """
    Detect if question contains quarter-related keywords.
    Returns True if any quarter-specific keywords are found.
    """
    q = question.lower()
    quarter_keywords = [
        "quarter", "quarters", "qtr", "qtrs", "q1", "q2", "q3", "q4",
        "q 1", "q 2", "q 3", "q 4",
        "first quarter", "second quarter", "third quarter", "fourth quarter",
        "1st quarter", "2nd quarter", "3rd quarter", "4th quarter",
    ]
    return any(kw in q for kw in quarter_keywords)


def has_explicit_period_context(question: str) -> bool:
    q = question.lower()
    patterns = [
        r"\b(last|previous|this|current)\b",
        r"\b\d+\s+(day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
        r"\b(" + _MONTH_RE[1:-1] + r")\b",
        r"\bq[1-4]\b",
        r"\bfy\s*\d{2,4}\b",
        r"\b20\d{2}\b",
        r"\b(mtd|qtd|ytd)\b",
        r"\b(today|yesterday|week)\b",
    ]
    return any(re.search(pattern, q, re.I) for pattern in patterns)


# ─────────────────────────────────────────────
# DateResolver  — unified, priority-ordered
# ─────────────────────────────────────────────

@dataclass
class Period:
    label: str
    start: date
    end: date

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label":      self.label,
            "start_date": self.start.isoformat(),
            "end_date":   self.end.isoformat(),
        }


class DateResolver:
    """
    Converts a natural-language question into a list of Period objects.

    Priority:
      1. LLM extraction  (returns structured JSON)
      2. Keyword intent  (decides which regex sub-parser to call)
      3. Regex sub-parsers (used as final fallback)
      4. Default: current FY
    """

    # ── public entry point ────────────────────────────────────────────────
    def resolve(self, question: str, llm_intent: Optional[Dict] = None) -> List[Period]:
        # 1. Use LLM result if available and valid
        if llm_intent and llm_intent.get("analysis_type", "unknown") != "unknown":
            periods = self._from_llm(llm_intent)
            if periods:
                return periods
            logger.warning("LLM intent conversion failed; falling back to regex")

        # 2. Keyword + regex
        return self._from_regex(question)

    # ── LLM → Period conversion ───────────────────────────────────────────
    def _from_llm(self, intent: Dict) -> List[Period]:
        out: List[Period] = []
        for p in intent.get("periods", []):
            try:
                out.append(Period(
                    label=p["label"],
                    start=date.fromisoformat(p["start_date"]),
                    end=date.fromisoformat(p["end_date"]),
                ))
            except (KeyError, ValueError) as exc:
                logger.warning("Bad LLM period %s: %s", p, exc)
        return out

    # ── Regex fallback ────────────────────────────────────────────────────
    def _from_regex(self, question: str) -> List[Period]:
        q   = question.lower().strip()
        now = _today()
        fy  = _current_fy()

        # — single date / range —
        pair = self._parse_date_pair(q)
        if pair:
            s, e = pair
            return [Period(f"{s} to {e}", s, e)]

        # — "this week / month / quarter / year / YTD / MTD / QTD" —
        # Must come BEFORE last_n_* so "this week" isn't consumed by week patterns
        p = self._this_period(q, now, fy)
        if p:
            return [p]

        # — last N weeks / last week —
        p = self._last_n_weeks(q, now)
        if p:
            return [p]

        # — last N days / last day —
        p = self._last_n_days(q, now)
        if p:
            return [p]

        # — last N months —
        p = self._last_n_months(q, now)
        if p:
            return [p]

        # — last N quarters —
        p = self._last_n_quarters(q, now, fy)
        if p:
            return [p]

        # — last N years —
        p = self.last_n_years(q, now, fy)
        if p:
            return [p]

        # — last quarter —
        p = self._last_quarter(q, now, fy)
        if p:
            return [p]

        # — named month with / without year —
        p = self._single_month(q, fy)
        if p:
            return [p]

        # — month range —
        p = self._month_range(q, fy)
        if p:
            return [p]

        # — quarter (Q1/Q2/Q3/Q4) —
        p = self._single_quarter(q, fy)
        if p:
            return [p]

        # — FY explicit (fy2025, fy25, fy 2025) —
        p = self._explicit_fy(q)
        if p:
            return [p]

        # — year range (2022 to 2024) —
        p = self._year_range(q)
        if p:
            return [p]

        # — bare year treated as FY start year —
        yr = self._extract_year(q)
        if yr:
            return [Period(f"FY{yr}", date(yr, 4, 1), date(yr + 1, 3, 31))]

        # — last / previous year —
        if re.search(r"\b(last|previous)\s+(financial\s+)?year\b", q):
            return [Period(f"FY{fy - 1}", date(fy - 1, 4, 1), date(fy, 3, 31))]

        # — Default: current FY to date —
        return [Period(f"FY{fy} YTD", date(fy, 4, 1), now.date())]

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        m = re.search(r"\b(20\d{2})\b", text)
        return int(m.group(1)) if m else None

    @staticmethod
    def _extract_month(text: str) -> Optional[int]:
        for key, num in sorted(MONTH_MAP.items(), key=lambda x: -len(x[0])):
            if re.search(rf"\b{key}\b", text):
                return num
        return None

    @staticmethod
    def _parse_date_pair(q: str) -> Optional[Tuple[date, date]]:
        """
        Handles:
          DD/MM/YYYY, DD-MM-YYYY, DD Month YYYY,
          "15 to 30 april 2024", "15 apr to 20 may"
        Returns (start, end) or None.
        """
        # slash / hyphen pair
        slash = re.search(
            r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\s+to\s+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
            q,
        )
        if slash:
            d1 = DateResolver._parse_dmy(slash.group(1))
            d2 = DateResolver._parse_dmy(slash.group(2))
            if d1 and d2:
                return min(d1, d2), max(d1, d2)

        # natural language range
        nl = re.search(
            r"(\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+(?:\s+\d{2,4})?)"
            r"\s+to\s+"
            r"(\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+(?:\s+\d{2,4})?)",
            q,
        )
        if nl:
            d1 = DateResolver._parse_natural(nl.group(1))
            d2 = DateResolver._parse_natural(nl.group(2))
            if d1 and d2:
                return min(d1, d2), max(d1, d2)

        # same-month range: "15 to 30 april 2024"
        sm = re.search(
            r"(\d{1,2})\s+to\s+(\d{1,2})\s+" + _MONTH_RE[1:-1] + r"(?:\s+(\d{4}))?",
            q,
        )
        if sm:
            d1_raw = f"{sm.group(1)} {sm.group(3)}"
            d2_raw = f"{sm.group(2)} {sm.group(3)}"
            yr = sm.group(4)
            if yr:
                d1_raw += f" {yr}"
                d2_raw += f" {yr}"
            d1 = DateResolver._parse_natural(d1_raw)
            d2 = DateResolver._parse_natural(d2_raw)
            if d1 and d2:
                return min(d1, d2), max(d1, d2)

        # single date
        single = DateResolver._parse_natural(q)
        if single:
            return single, single

        return None

    @staticmethod
    def _parse_dmy(s: str) -> Optional[date]:
        s = s.strip()
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass
        return None

    @staticmethod
    def _parse_natural(s: str) -> Optional[date]:
        """Parse 'DD Month [YY[YY]]' style."""
        s = re.sub(r"(st|nd|rd|th)", "", s).strip()
        m = re.search(
            r"(\d{1,2})\s+" + _MONTH_RE[1:-1] + r"(?:\s+(\d{2,4}))?",
            s,
        )
        if not m:
            return None
        day   = int(m.group(1))
        month = MONTH_MAP.get(m.group(2)[:3])
        if not month:
            return None
        yr_raw = m.group(3)
        if yr_raw:
            yr = int(yr_raw)
            if yr < 100:
                yr += 2000
        else:
            fy = _current_fy()
            yr = fy if month >= 4 else fy + 1
        try:
            return date(yr, month, day)
        except ValueError:
            return None

    @staticmethod
    def _last_n_days(q: str, now: datetime) -> Optional[Period]:
        today = now.date()
        m = re.search(r"(?:last|past)\s+(\d+|" + "|".join(WORD_NUM) + r")\s+days?", q)
        if m:
            n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUM[m.group(1)]
            # "last N days" excludes today — end is yesterday
            end = today - timedelta(days=1)
            start = today - timedelta(days=n)
            return Period(f"Last {n} days", start, end)
        if re.search(r"\b(?:last|past)\s+days?\b", q):
            # "last day" = yesterday
            yesterday = today - timedelta(days=1)
            return Period("Last day", yesterday, yesterday)
        return None
    @staticmethod
    def _last_n_weeks(q: str, now: datetime) -> Optional[Period]:
        today = now.date()
        # Monday of the current week
        this_week_monday = today - timedelta(days=today.weekday())

        m = re.search(r"(?:last|past)\s+(\d+|" + "|".join(WORD_NUM) + r")\s+weeks?", q)
        if m:
            n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUM[m.group(1)]
            # end = Sunday just before this week's Monday
            end   = this_week_monday - timedelta(days=1)
            # start = Monday of N weeks ago
            start = this_week_monday - timedelta(weeks=n)
            return Period(f"Last {n} weeks", start, end)

        if re.search(r"\b(?:last|previous|past)\s+week\b", q):
            # Last week: Monday → Sunday of the immediately preceding week
            end   = this_week_monday - timedelta(days=1)          # last Sunday
            start = this_week_monday - timedelta(days=7)          # last Monday
            return Period("Last Week", start, end)

        return None

    @staticmethod
    def _this_period(q: str, now: datetime, fy: int) -> Optional[Period]:
        today = now.date()
        if re.search(r"\b(this|current)\s+week\b", q):
            monday = today - timedelta(days=today.weekday())
            return Period("This Week", monday, today)
        if re.search(r"\b(this|current)\s+month\b|\bmtd\b", q):
            return Period("This Month (MTD)", today.replace(day=1), today)
        if re.search(r"\b(this|current)\s+(fy|financial\s+year|year)\b|\bytd\b", q):
            return Period(f"FY{fy} YTD", date(fy, 4, 1), today)
        if re.search(r"\b(this|current)\s+quarter\b|\bqtd\b", q):
            q_num = _fy_quarter(today.month)
            s, _e = _quarter_dates(q_num, fy)
            # QTD: from quarter start to today (not the full quarter end)
            return Period(f"Q{q_num} FY{fy} (QTD)", s, today)
        return None

    @staticmethod
    def _last_n_months(q: str, now: datetime) -> Optional[Period]:
        m = re.search(
            r"last\s+(\d+|" + "|".join(WORD_NUM) + r")\s+months?", q
        )
        if not m:
            return None
        n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUM[m.group(1)]
        first_this = now.date().replace(day=1)
        end = first_this - timedelta(days=1)
        start = end.replace(day=1)
        for _ in range(n - 1):
            start = (start - timedelta(days=1)).replace(day=1)
        return Period(f"Last {n} months", start, end)
    @staticmethod
    def _last_n_quarters(q: str, now: datetime, fy: int) -> Optional[Period]:
        m = re.search(r"last\s+(\d+|" + "|".join(WORD_NUM) + r")\s+quarters?", q)
        if not m:
            return None
        n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUM[m.group(1)]
        curr_q = _fy_quarter(now.month)
        end_q = curr_q - 1 if curr_q > 1 else 4
        end_fy = fy if curr_q > 1 else fy - 1
        end_s, end_e = _quarter_dates(end_q, end_fy)
        start_q = end_q
        start_fy = end_fy
        for _ in range(n - 1):
            start_q -= 1
            if start_q == 0:
                start_q = 4
                start_fy -= 1
        start_s, start_e = _quarter_dates(start_q, start_fy)
        return Period(f"Last {n} quarters", start_s, end_e)
    
    @staticmethod
    def last_n_years(q: str, now: datetime, fy: int) -> Optional[Period]:
        # "last years" (no number) → previous single FY
        # Use negative lookahead so "last 3 years" doesn't match this branch
        if re.search(r"\blast\s+years\b", q) and not re.search(r"\blast\s+\d", q):
            return Period("Last year", date(fy - 1, 4, 1), date(fy, 3, 31))
        m = re.search(r"last\s+(\d+|" + "|".join(WORD_NUM) + r")\s+years?", q)
        if not m:
            return None
        n        = int(m.group(1)) if m.group(1).isdigit() else WORD_NUM[m.group(1)]
        end_fy   = fy - 1
        start_fy = end_fy - n + 1
        return Period(f"Last {n} years", date(start_fy, 4, 1), date(end_fy + 1, 3, 31))


    @staticmethod
    def _last_quarter(q: str, now: datetime, fy: int) -> Optional[Period]:
        """
        Resolve "last quarter" or "previous quarter" to the most recently completed quarter.
        
        For fiscal year April 1 – March 31:
        - Q1 = Apr-Jun (months 4-6)
        - Q2 = Jul-Sep (months 7-9)
        - Q3 = Oct-Dec (months 10-12)
        - Q4 = Jan-Mar (months 1-3, of next calendar year)
        
        Examples on June 9, 2026 (Q1 FY2026):
        - "last quarter" → Q4 FY2025 (Jan-Mar 2026)
        - "this quarter" → Q1 FY2026 (Apr-Jun 2026, to-date)
        """
        if not re.search(r"\blast\s+quarter\b|\bprevious\s+quarter\b", q):
            return None
        
        today = now.date()
        curr_q = _fy_quarter(today.month)
        
        # Calculate previous quarter
        if curr_q == 1:
            # In Q1, previous quarter is Q4 of previous FY
            prev_q = 4
            prev_fy = fy - 1
        else:
            # In Q2-Q4, previous quarter is just the previous number in same FY
            prev_q = curr_q - 1
            prev_fy = fy
        
        s, e = _quarter_dates(prev_q, prev_fy)
        return Period(f"Q{prev_q} FY{prev_fy}", s, e)

    @staticmethod
    def _single_month(q: str, fy: int) -> Optional[Period]:
        m_num = DateResolver._extract_month(q)
        if not m_num:
            return None
        yr = DateResolver._extract_year(q)
        if not yr:
            yr = fy if m_num >= 4 else fy + 1
        return Period(
            datetime(yr, m_num, 1).strftime("%b %Y"),
            date(yr, m_num, 1),
            _month_end(yr, m_num),
        )

    @staticmethod
    def _month_range(q: str, fy: int) -> Optional[Period]:
        sep = r"\s*(?:to|till|through|–|-|—|and)\s*"
        pat = re.compile(_MONTH_RE[1:-1] + sep + _MONTH_RE[1:-1] + r"(?:\s+(\d{4}))?", re.I)
        m = pat.search(q)
        if not m:
            return None
        m1 = MONTH_MAP.get(m.group(1)[:3])
        m2 = MONTH_MAP.get(m.group(2)[:3])
        if not m1 or not m2:
            return None
        yr_raw = m.group(3)
        yr = int(yr_raw) if yr_raw else fy
        y1 = yr if m1 >= 4 else yr + 1
        y2 = yr if m2 >= 4 else yr + 1
        if m2 < m1:
            y2 += 1
        return Period(
            f"{datetime(y1, m1, 1).strftime('%b %Y')} – {datetime(y2, m2, 1).strftime('%b %Y')}",
            date(y1, m1, 1),
            _month_end(y2, m2),
        )

    @staticmethod
    def _single_quarter(q: str, fy: int) -> Optional[Period]:
        m = re.search(r"\bq([1-4])\b", q, re.I)
        if not m:
            return None
        q_num = int(m.group(1))
        # optional explicit FY
        fy_m = re.search(r"\bfy\s*(\d{2,4})\b", q, re.I)
        if fy_m:
            v = int(fy_m.group(1))
            fy = v if v > 100 else 2000 + v
        if re.search(r"\b(last|previous)\s+(year|fy)\b", q):
            fy -= 1
        s, e = _quarter_dates(q_num, fy)
        return Period(f"Q{q_num} FY{fy}", s, e)

    @staticmethod
    def _explicit_fy(q: str) -> Optional[Period]:
        # Handles: fy2025, fy 2025, fy25, fy 25, financial year 2025
        m = re.search(r"\bfy\s*(\d{2,4})\b", q, re.I)
        if not m:
            if "financial year" not in q:
                return None
            m = re.search(r"financial\s+year\s+(\d{4})", q, re.I)
            if not m:
                return None
        v  = int(m.group(1))
        yr = v if v > 100 else 2000 + v

        # "until FY2025" / "till FY25" → start of current FY to end of named FY
        if re.search(r"\b(until|till|upto|up\s+to)\b", q):
            fy_start = _current_fy()
            return Period(
                f"FY{fy_start} to FY{yr}",
                date(fy_start, 4, 1),
                date(yr + 1, 3, 31),
            )

        return Period(f"FY{yr}", date(yr, 4, 1), date(yr + 1, 3, 31))

    @staticmethod
    def _year_range(q: str) -> Optional[Period]:
        m = re.search(r"\b(20\d{2})\s*(?:to|–|-|—)\s*(20\d{2})\b", q)
        if not m:
            m = re.search(r"\bfy\s*(\d{2})\s*(?:to|-)\s*fy?\s*(\d{2})\b", q, re.I)
            if not m:
                return None
            y1, y2 = 2000 + int(m.group(1)), 2000 + int(m.group(2))
        else:
            y1, y2 = int(m.group(1)), int(m.group(2))
        return Period(
            f"FY{min(y1, y2)} to FY{max(y1, y2)}",
            date(min(y1, y2), 4, 1),
            date(max(y1, y2) + 1, 3, 31),
        )

def _extract_all_fy_years(text: str) -> List[int]:
    """
    Extract ALL fiscal-year references from text and return sorted list of
    4-digit start years.  Handles: FY2025, FY 2025, FY25, FY 25, plain 2025.
    """
    text_lower = text.lower()
    seen: set = set()
    results: List[int] = []

    # Pass 1: fy + 4-digit
    for m in re.finditer(r"\bfy\s*(20\d{2})\b", text_lower):
        yr = int(m.group(1))
        if yr not in seen:
            seen.add(yr); results.append(yr)

    # Pass 2: fy + 2-digit shorthand
    for m in re.finditer(r"\bfy\s*(\d{2})\b", text_lower):
        yr = 2000 + int(m.group(1))
        if yr not in seen:
            seen.add(yr); results.append(yr)

    # Pass 3: bare 4-digit year (only when no FY-prefixed found)
    if not results:
        for m in re.finditer(r"\b(20\d{2})\b", text):
            yr = int(m.group(1))
            if yr not in seen:
                seen.add(yr); results.append(yr)

    return sorted(results)
    """Lower-case, collapse whitespace, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # punctuation → space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_quarter_number(tokens: list[str]) -> int | None:
    """
    Return 1-4 if any token sequence looks like a quarter specifier.
    Handles: q1, q 1, qtr1, qtr 1, quarter1, quarter 1, quarter one …
    """
    QUARTER_WORDS = {"q", "qtr", "quarter", "quarters"}
    ORDINALS = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
                "first": 1, "second": 2, "third": 3, "fourth": 4,
                "one": 1, "two": 2, "three": 3, "four": 4}

    for i, tok in enumerate(tokens):
        # Compact forms: "q1", "q2", "qtr3", "quarter4"
        for prefix in ("quarter", "qtr", "q"):
            if tok.startswith(prefix) and tok[len(prefix):].isdigit():
                n = int(tok[len(prefix):])
                if 1 <= n <= 4:
                    return n

        # Two-token forms: ("q","1"), ("quarter","2"), ("quarter","first") …
        if tok in QUARTER_WORDS and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.isdigit() and 1 <= int(nxt) <= 4:
                return int(nxt)
            if nxt in ORDINALS:
                return ORDINALS[nxt]

    return None


def _extract_fy_year(tokens: list[str], fallback: int) -> int:
    """
    Return a 4-digit FY year if explicitly mentioned, else *fallback*.
    Handles: fy2026, fy 26, fy 2026, 2024, …
    """
    for i, tok in enumerate(tokens):
        # Compact: "fy2026", "fy26"
        if tok.startswith("fy") and tok[2:].isdigit():
            v = int(tok[2:])
            return v if v > 100 else 2000 + v
        # Two-token: ("fy", "2026") or ("fy", "26")
        if tok == "fy" and i + 1 < len(tokens) and tokens[i + 1].isdigit():
            v = int(tokens[i + 1])
            return v if v > 100 else 2000 + v
        # Bare 4-digit year anywhere
        if tok.isdigit() and len(tok) == 4:
            return int(tok)

    return fallback


def _has_any(tokens: set[str], *keywords: str) -> bool:
    """True if the token set contains *any* of the given keywords."""
    return bool(tokens & set(keywords))

def _mom_periods(question: str) -> list[Period]:
    """
    Generate a list of month-level Period objects for MOM analysis.

    Financial Year (FY) system
    --------------------------
    FY2026  →  Apr 1 2025 – Mar 31 2026
    Q1 Apr-Jun  |  Q2 Jul-Sep  |  Q3 Oct-Dec  |  Q4 Jan-Mar

    Accepted quarter phrasings (case-insensitive, spacing-insensitive)
    ------------------------------------------------------------------
    Q1 / Q 1 / q1 / Qtr1 / Qtr 1 / Quarter1 / Quarter 1 /
    Quarter one / 1st quarter / first quarter  … (and 2-4 equivalents)
    """
    dr = DateResolver()
    base = dr.resolve(question)
    now = _today()
    fy = _current_fy()
    today = now.date()

    # ── token-set preparation ────────────────────────────────────────────────
    norm = _normalize(question)
    tokens: list[str] = norm.split()
    tok_set: set[str] = set(tokens)

    # Convenient intent flags derived purely from token membership
    has_last     = _has_any(tok_set, "last", "previous", "prev", "prior")
    has_this     = _has_any(tok_set, "this", "current")
    has_quarter  = _has_any(tok_set, "quarter", "quarters", "qtr", "q")
    has_year     = _has_any(tok_set, "year", "yr", "fy", "annual")

    quarter_num  = _extract_quarter_number(tokens)   # 1-4 or None
    explicit_fy  = _extract_fy_year(tokens, fallback=-1)  # -1 = not found

    # ── period resolution ────────────────────────────────────────────────────

    if has_last and has_quarter and quarter_num is None:
        # "last quarter" / "previous quarter" / "prior qtr"
        curr_q = _fy_quarter(today.month)
        if curr_q == 1:
            prev_q, prev_fy = 4, fy - 1
        else:
            prev_q, prev_fy = curr_q - 1, fy
        s, e = _quarter_dates(prev_q, prev_fy)

    elif has_this and has_quarter and quarter_num is None:
        # "this quarter" / "current quarter"
        curr_q = _fy_quarter(today.month)
        s, e = _quarter_dates(curr_q, fy)
        e = min(e, today)

    elif quarter_num is not None:
        # Explicit quarter: "Q2", "quarter 3", "qtr 1 fy2025", "2nd quarter" …
        fy_for_q = explicit_fy if explicit_fy != -1 else fy
        # "… of last year / previous FY"
        if has_last and has_year:
            fy_for_q -= 1
        s, e = _quarter_dates(quarter_num, fy_for_q)

    elif has_last and has_year:
        # "last year" / "previous FY" / "last fy"
        s, e = date(fy - 1, 4, 1), date(fy, 3, 31)

    elif has_this and has_year:
        # "this year" / "current FY"
        s, e = date(fy, 4, 1), today

    elif base:
        s, e = base[0].start, base[0].end

    else:
        s, e = date(fy, 4, 1), today

    # ── generate one Period per calendar month ───────────────────────────────
    periods: list[Period] = []
    cur = s.replace(day=1)

    while cur <= e:
        me  = _month_end(cur.year, cur.month)
        end = min(me, e, today)
        lbl = cur.strftime("%b %Y")
        if cur.year == today.year and cur.month == today.month:
            lbl += " (MTD)"
        periods.append(Period(lbl, cur, end))

        cur = (date(cur.year + 1, 1, 1) if cur.month == 12
               else date(cur.year, cur.month + 1, 1))

    return periods


def _qoq_periods(question: str) -> List[Period]:
    q   = question.lower()
    fy  = _current_fy()
    now = _today()

    years         = _extract_all_fy_years(question)
    is_year_range = (
        len(years) >= 2
        and (" to " in q or re.search(r"(20\d{2}|fy\s*\d{2,4})\s*[-\u2013\u2014]\s*(20\d{2}|fy\s*\d{2,4})", q, re.I))
    )
    if is_year_range:
        periods: List[Period] = []
        for year in range(years[0], years[-1] + 1):
            for quarter in range(1, 5):
                periods.append(Period(f"Q{quarter} FY{year}", *_quarter_dates(quarter, year)))
        return periods

    if (last_n := DateResolver._last_n_quarters(q, now, fy)):
        periods = []
        cur = last_n.start
        while cur <= last_n.end:
            q_num = _fy_quarter(cur.month)
            q_fy  = cur.year if cur.month >= 4 else cur.year - 1
            periods.append(Period(f"Q{q_num} FY{q_fy}", *_quarter_dates(q_num, q_fy)))
            next_month = cur.month + 3
            next_year  = cur.year
            if next_month > 12:
                next_month -= 12
                next_year  += 1
            cur = date(next_year, next_month, 1)
        return periods

    if re.search(r"\blast\s+quarter\b|\bprevious\s+quarter\b", q):
        curr_q  = _fy_quarter(now.month)
        prev_q  = curr_q - 1 if curr_q > 1 else 4
        prev_fy = fy if curr_q > 1 else fy - 1
        return [Period(f"Q{prev_q} FY{prev_fy}", *_quarter_dates(prev_q, prev_fy))]

    # Explicit FY: fy25, fy2025, fy 2025 ...  (FY keyword wins over bare year)
    fy_m = re.search(r"\bfy\s*(\d{2,4})\b", q, re.I)
    if fy_m:
        v  = int(fy_m.group(1))
        fy = v if v > 100 else 2000 + v
    elif years:
        fy = years[0]

    if re.search(r"\b(last|previous)\s+(year|fy)\b", q):
        fy -= 1

    return [
        Period(f"Q{i} FY{fy}", *_quarter_dates(i, fy))
        for i in range(1, 5)
    ]


def _yoy_periods(question: str) -> List[Period]:
    fy    = _current_fy()
    q     = question.lower()
    today = _today().date()

    if re.search(r"\b(last|previous)\s+(financial\s+)?(year|fy)\b|\blast\s+years\b", q):
        return [Period(f"FY{fy - 1}", date(fy - 1, 4, 1), date(fy, 3, 31))]

    if (last_n := DateResolver.last_n_years(q, _today(), fy)):
        start_fy = last_n.start.year if last_n.start.month >= 4 else last_n.start.year - 1
        # last_n.end is March 31 of (end_fy+1), so calendar year is end_fy+1
        end_fy   = last_n.end.year - 1
        return [
            Period(f"FY{year}", date(year, 4, 1), date(year + 1, 3, 31))
            for year in range(start_fy, end_fy + 1)
        ]

    # FY-prefixed year references (FY2025, FY25 …)
    fy_years = _extract_all_fy_years(question)
    if fy_years:
        return [
            Period(f"FY{y}", date(y, 4, 1), date(y + 1, 3, 31))
            for y in fy_years
        ]

    periods = [
        Period(f"FY{past_fy}", date(past_fy, 4, 1), date(past_fy + 1, 3, 31))
        for past_fy in [fy - 3, fy - 2, fy - 1]
    ]
    periods.append(Period(f"FY{fy} YTD", date(fy, 4, 1), today))
    return periods


def _multi_quarter_periods(question: str) -> List[Period]:
    q  = question.lower()
    fy = _current_fy()
    if re.search(r"\b(last|previous)\s+(year|fy)\b", q): fy -= 1
    yr_m = re.search(r"\b(20\d{2})\b", q)
    if yr_m: fy = int(yr_m.group(1))
    fy_m = re.search(r"\bfy\s*(\d{2,4})\b", q, re.I)
    if fy_m:
        v = int(fy_m.group(1)); fy = v if v > 100 else 2000 + v

    q_nums = [int(x) for x in re.findall(r"\bq([1-4])\b", q, re.I)]
    if not q_nums:
        return []
    is_range = " to " in q
    if is_range:
        q_nums = list(range(min(q_nums), max(q_nums) + 1))
    return [Period(f"Q{n} FY{fy}", *_quarter_dates(n, fy)) for n in sorted(set(q_nums))]


def _multi_month_periods(question: str) -> List[Period]:
    q  = question.lower()
    fy = _current_fy()
    if re.search(r"\b(last|previous)\s+(year|fy)\b", q): fy -= 1
    yr_m = re.search(r"\b(20\d{2})\b", q)
    if yr_m: fy = int(yr_m.group(1))

    month_names = re.findall(_MONTH_RE, q)
    periods: List[Period] = []
    for mn in month_names:
        m_num = MONTH_MAP.get(mn[:3])
        if not m_num: continue
        yr = fy if m_num >= 4 else fy + 1
        s  = date(yr, m_num, 1)
        e  = _month_end(yr, m_num)
        periods.append(Period(datetime(yr, m_num, 1).strftime("%b %Y"), s, e))
    return periods


def _multi_year_periods(question: str) -> List[Period]:
    q     = question.lower()
    years = _extract_all_fy_years(question)
    if not years:
        return []
    is_range = " to " in q or re.search(r"(20\d{2}|fy\s*\d{2,4})\s*[-–—]\s*(20\d{2}|fy\s*\d{2,4})", q, re.I)
    if is_range:
        years = list(range(years[0], years[-1] + 1))
    return [
        Period(f"FY{y}", date(y, 4, 1), date(y + 1, 3, 31))
        for y in years
    ]


def handle_quarter_intent(question: str) -> Optional[List[Period]]:
    """
    Handle quarter-related intents with proper fiscal year alignment.
    
    Supports:
    - "last quarter" → most recently completed quarter
    - "this quarter" → current quarter to today
    - "current quarter" → current quarter to today
    - "Q1/Q2/Q3/Q4" → specific quarter of current FY
    - "Q1 FY2026" → specific quarter of specific FY
    - "last quarter of FY2026" → Q4 of specified fiscal year
    - "first/second/third/fourth quarter" → ordinal quarter names
    - Quarter ranges: "Q1 to Q3", "Q2 and Q4"
    
    Returns list of Period objects or None if not a quarter-related intent.
    """
    q = question.lower()
    now = _today()
    today = now.date()
    fy = _current_fy()
    
    # Check if this is a quarter-related question
    if not detect_quarter_keywords(question):
        return None
    
    # Determine explicit FY if specified
    explicit_fy = _extract_fy_year(q.split(), -1)
    target_fy = explicit_fy if explicit_fy != -1 else fy
    
    # Handle "last quarter" / "previous quarter"
    if re.search(r"\b(last|previous|prior)\s+quarter\b", q):
        curr_q = _fy_quarter(today.month)
        if curr_q == 1:
            prev_q, prev_fy = 4, fy - 1
        else:
            prev_q, prev_fy = curr_q - 1, fy
        s, e = _quarter_dates(prev_q, prev_fy)
        return [Period(f"Q{prev_q} FY{prev_fy}", s, e)]
    
    # Handle "this quarter" / "current quarter"
    if re.search(r"\b(this|current)\s+quarter\b", q):
        curr_q = _fy_quarter(today.month)
        s, e = _quarter_dates(curr_q, fy)
        return [Period(f"Q{curr_q} FY{fy} (current)", s, min(e, today))]
    
    # Handle "next quarter"
    if re.search(r"\bnext\s+quarter\b", q):
        curr_q = _fy_quarter(today.month)
        next_q = curr_q + 1 if curr_q < 4 else 1
        next_fy = fy if curr_q < 4 else fy + 1
        s, e = _quarter_dates(next_q, next_fy)
        return [Period(f"Q{next_q} FY{next_fy} (upcoming)", s, e)]
    
    # Handle ordinal quarter names: "first quarter", "second quarter", etc.
    ordinal_map = {
        "first": 1, "1st": 1,
        "second": 2, "2nd": 2,
        "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4,
    }
    for word, q_num in ordinal_map.items():
        if re.search(rf"\b{word}\s+quarter\b", q):
            s, e = _quarter_dates(q_num, target_fy)
            label = f"Q{q_num} FY{target_fy}"
            if "last" in q and target_fy == fy:
                label += " (last year)" if explicit_fy == -1 else ""
            return [Period(label, s, e)]
    
    # Handle explicit quarter numbers with optional FY: "Q1", "Q2 FY2024", etc.
    q_matches = re.findall(r"\bq\s*([1-4])\b", q, re.I)
    if q_matches:
        quarters = sorted(set(int(m) for m in q_matches))
        
        # Check if it's a range: "Q1 to Q3" or "Q1 and Q3"
        is_range = bool(re.search(r"q\s*\d\s*(?:to|through)\s*q\s*\d", q, re.I))
        if is_range and len(quarters) >= 2:
            quarters = list(range(min(quarters), max(quarters) + 1))
        
        # Handle "of last year / previous FY"
        if re.search(r"(?:of|in)\s+(?:last|previous)\s+(?:year|fy)", q):
            target_fy = fy - 1
        
        return [Period(f"Q{q_num} FY{target_fy}", *_quarter_dates(q_num, target_fy)) 
                for q_num in quarters]
    
    return None


# ─────────────────────────────────────────────
# Response Builders
# ─────────────────────────────────────────────

def _run_period(period: Period) -> Dict[str, Any]:
    """Fetch data for one period and compute funnel metrics."""
    try:
        df = fetch_funnel_data(period.start, period.end)
    except Exception as exc:
        logger.error("DB error for %s: %s", period.label, exc)
        return {"label": period.label, "error": str(exc)}

    if df.empty:
        logger.warning("No data for %s", period.label)

    funnel = compute_funnel(df)
    return {
        "label":  period.label,
        "period": f"{period.start.isoformat()} to {period.end.isoformat()}",
        "funnel": funnel,
    }


def _build_single_response(period: Period) -> Dict[str, Any]:
    result = _run_period(period)
    return {
        "status":        "success",
        "analysis_type": AnalysisIntent.SINGLE,
        "filter":        result.get("period"),
        "lead_funnel":   result.get("funnel", {}),
        "totals":        aggregate_funnels([result.get("funnel", {})]),
    }


def _build_multi_response(
    analysis_type: str,
    periods: List[Period],
    label_key: str = "label",
) -> Dict[str, Any]:
    results = [_run_period(p) for p in periods]
    funnels = [r.get("funnel", {}) for r in results]
    return {
        "status":        "success",
        "analysis_type": analysis_type,
        "data":          results,
        "totals":        aggregate_funnels(funnels),
    }


def _build_comparison_sections(
    question: str,
    intents: List[AnalysisIntent],
    llm_intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    seen: Set[AnalysisIntent] = set()
    llm_analysis_type = (llm_intent or {}).get("analysis_type")
    llm_periods = _resolver._from_llm(llm_intent) if llm_intent else []

    for intent in intents:
        if intent in seen:
            continue
        seen.add(intent)

        if intent == AnalysisIntent.MOM:
            periods = llm_periods if llm_analysis_type == AnalysisIntent.MOM and llm_periods else _mom_periods(question)
            comparisons["mom"] = _build_multi_response("month_on_month", periods)
        elif intent == AnalysisIntent.QOQ:
            periods = llm_periods if llm_analysis_type in {AnalysisIntent.QOQ, AnalysisIntent.MULTI_Q} and llm_periods else _qoq_periods(question)
            comparisons["qoq"] = _build_multi_response("quarter_on_quarter", periods)
        elif intent == AnalysisIntent.YOY:
            periods = llm_periods if llm_analysis_type in {AnalysisIntent.YOY, AnalysisIntent.MULTI_Y} and llm_periods else _yoy_periods(question)
            comparisons["yoy"] = _build_multi_response("year_on_year", periods)

    return comparisons


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Lead Funnel Analytics API",
    description="IBM Watsonx-powered sales funnel with LLM intent extraction",
    version="2.0.0",
)

_resolver = DateResolver()


@app.post("/funnel/leadfunnel/question")
async def lead_funnel_from_question(payload: dict = Body(...)):
    question = payload.get("question", "").strip()
    if not question:
        return {"status": "error", "message": "question is required"}

    logger.info("Request: %s", question)
    comparison_intents = detect_comparison_intents(question)

    # ── Step 1: LLM intent extraction ─────────────────────────────────────
    llm_intent = llm_extract_intent(question)
    analysis_type = llm_intent.get("analysis_type", "unknown")

    print(f"LLM intent: {llm_intent}")
    
    # ── Step 1a: Check for quarter-specific intent (before generic LLM handling) ──
    quarter_periods = handle_quarter_intent(question)
    if quarter_periods and analysis_type == "unknown":
        # Only use quarter handler if LLM didn't provide intent
        if len(quarter_periods) == 1:
            return _build_single_response(quarter_periods[0])
        return _build_multi_response("quarter_analysis", quarter_periods)
    
    # ── Step 2: If LLM gave a valid multi-period intent, use it directly ──
    if analysis_type != "unknown" and not comparison_intents:
        # Convert LLM periods to Period objects
        llm_periods = _resolver._from_llm(llm_intent)

        if llm_periods:
            if analysis_type == AnalysisIntent.SINGLE:
                return _build_single_response(llm_periods[0])

            return _build_multi_response(analysis_type, llm_periods)

    # ── Step 3: Keyword-based intent detection (fast path) ────────────────
    kw_intent = detect_intent_from_keywords(question)

    if comparison_intents:
        response: Dict[str, Any] = {
            "status": "success",
            "analysis_type": "comparison_bundle",
            "comparisons": _build_comparison_sections(question, comparison_intents, llm_intent),
        }

        if has_explicit_period_context(question):
            periods = _resolver.resolve(question)
            if len(periods) == 1:
                response["primary"] = _build_single_response(periods[0])
            elif periods:
                response["primary"] = _build_multi_response("multi_period", periods)

        return response

    if kw_intent == AnalysisIntent.YOY:
        periods = _yoy_periods(question)
        return _build_multi_response("year_on_year", periods)

    if kw_intent == AnalysisIntent.QOQ:
        periods = _qoq_periods(question)
        return _build_multi_response("quarter_on_quarter", periods)

    if kw_intent == AnalysisIntent.MOM:
        periods = _mom_periods(question)
        return _build_multi_response("month_on_month", periods)

    if kw_intent == AnalysisIntent.MULTI_Q:
        periods = _multi_quarter_periods(question)
        if periods:
            return _build_multi_response("multi_quarter", periods)

    if kw_intent == AnalysisIntent.MULTI_M:
        periods = _multi_month_periods(question)
        if periods:
            return _build_multi_response("multi_month", periods)

    if kw_intent == AnalysisIntent.MULTI_Y:
        periods = _multi_year_periods(question)
        if periods:
            return _build_multi_response("multi_year", periods)

    # ── Step 4: Generic date resolution (regex fallback) ──────────────────
    periods = _resolver.resolve(question, llm_intent)
    if len(periods) == 1:
        return _build_single_response(periods[0])

    return _build_multi_response("multi_period", periods)


@app.get("/")
async def health_check():
    return {
        "status":  "ok",
        "service": "Lead Funnel Analytics API v2",
        "llm":     MODEL_ID,
    }


@app.get("/supported-formats")
async def supported_formats():
    """Return all supported natural-language date formats."""
    return {
        "single_date": [
            "15 April 2024", "15/04/2024", "15-04-2024",
            "5th June", "Jun 2024", "April",
        ],
        "date_range": [
            "15 April to 30 June 2024",
            "15/04/2024 to 30/06/2024",
            "April to June",
            "April to June 2024",
            "April 2022 to June 2025",
        ],
        "relative": [
            "last 30 days", "last 3 months", "last week",
            "last month", "last quarter", "last year", "last FY",
            "this month", "this quarter", "this year", "this FY",
            "MTD", "QTD", "YTD",
        ],
        "quarter": [
            "Q1", "Q2 FY24", "Q3 2024",
            "Q1 and Q3", "Q1 to Q3", "last quarter",
        ],
        "financial_year": [
            "FY24", "FY2024", "FY 2024",
            "2023-24", "2022 to 2024",
            "financial year 2024",
        ],
        "comparative": [
            "MOM", "month on month", "month-wise",
            "QOQ", "quarter on quarter", "quarterly",
            "YOY", "year on year", "year-over-year",
        ],
    }