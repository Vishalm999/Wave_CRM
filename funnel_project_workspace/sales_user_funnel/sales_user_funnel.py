"""
Production-Ready Sales User-Wise Funnel Analytics API
======================================================
Architecture:
  - LLM (IBM Watsonx / OpenAI-compatible) handles intent + date extraction (primary path)
  - Rule-based keyword + regex acts as fallback (never the primary path)
  - All date parsing goes through a unified DateResolver (mirrors lead_user_funnel.py)
  - User-wise funnel computation maps Salesforce users → opportunity/event owners → metrics
  - FastAPI endpoint: POST /funnel/salesuser/question
"""

from __future__ import annotations

import json
import logging
import os
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import prestodb
import requests
from dotenv import load_dotenv
from fastapi import Body, FastAPI
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# --------------------------------------------
# Logging & Environment
# --------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("sales_user_funnel_tool.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("sales_user_funnel_api")

# --------------------------------------------
# Configuration
# --------------------------------------------
CATALOG            = os.getenv("CATALOG", "salesforcereport")
OPPORTUNITY_SCHEMA = os.getenv("OPPORTUNITY_SCHEMA", "opportunity_sf_report")
OPPORTUNITY_TABLE  = os.getenv("OPPORTUNITY_TABLE", "opportunity_report")
EVENT_SCHEMA       = os.getenv("EVENT_SCHEMA", "event_sf_report")
EVENT_TABLE        = os.getenv("EVENT_TABLE", "event_report")

PRESTO_HOST = os.getenv("PRESTO_HOST")
PRESTO_PORT = int(os.getenv("PRESTO_PORT", "443"))
PRESTO_USER = os.getenv("PRESTO_USERNAME")
PRESTO_PASS = os.getenv("PRESTO_PASSWORD")

WATSONX_URL        = os.getenv("WATSONX_URL")
WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
MODEL_ID           = os.getenv("MODEL_ID", "meta-llama/llama-3-2-3b-instruct")

SALESFORCE_USER_API      = os.getenv("SALESFORCE_USER_API") or os.getenv("user_api")
SALESFORCE_TOKEN_URL     = os.getenv("SALESFORCE_TOKEN_URL") or os.getenv("TOKEN_URL")
SALESFORCE_CLIENT_ID     = os.getenv("SALESFORCE_CLIENT_ID") or os.getenv("CLIENT_ID")
SALESFORCE_CLIENT_SECRET = os.getenv("SALESFORCE_CLIENT_SECRET") or os.getenv("CLIENT_SECRET")

# --------------------------------------------
# Keyword / Alias Mappings
# --------------------------------------------
PROJECT_ALIASES: Dict[str, str] = {
    "wave city":     "wave city",
    "wavecity":      "wave city",
    "wave_city":     "wave city",
    "wmcc sec 2":    "wmcc sec 2",
    "wmcc":          "wmcc sec 2",
    "wmcc sector 2": "wmcc sec 2",
    "wave estate":   "wave estate",
    "estate":        "wave estate",
    "wave one":      "wave one",
    "waveone":       "wave one",
    "wave highrise": "wave highrise",
}

ALL_PROJECT_PHRASES: List[str] = [
    "all project", "all projects", "every project",
    "project wise", "projects wise", "by project",
    "per project", "overall project",
    "all user sales funnel for project", "user wise sales funnel for project",
]

ACTIVE_STATUS_MAP: Dict[str, bool] = {
    "active":         True,
    "active users":   True,
    "active user":    True,
    "inactive":       False,
    "inactive users": False,
    "inactive user":  False,
    "not active":     False,
    "disabled":       False,
    "non active":     False,
}

# --------------------------------------------
# LLM Client (Watsonx) — lazy singleton
# --------------------------------------------
_llm_model: Optional[ModelInference] = None

_RANK_SORT_METRIC = "Meeting Booked (MB)"
_RANK_PATTERNS: List[Tuple[str, str]] = [
    (r"\bbottom\s+(\d+)\b", "bottom"),
    (r"\btop\s+(\d+)\b",    "top"),
    (r"\bworst\s+(\d+)\b",  "bottom"),
    (r"\bbest\s+(\d+)\b",   "top"),
]


def _get_llm() -> Optional[ModelInference]:
    global _llm_model
    if _llm_model is not None:
        return _llm_model
    try:
        creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
        _llm_model = ModelInference(
            model_id=MODEL_ID,
            credentials=creds,
            project_id=WATSONX_PROJECT_ID,
            params={"temperature": 0, "max_new_tokens": 1024},
        )
        logger.info("LLM client initialised: %s", MODEL_ID)
        return _llm_model
    except Exception as exc:
        logger.warning("LLM unavailable (%s) - falling back to regex", exc)
        return None


def _extract_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            parsed, _ = decoder.raw_decode(raw[match.start():])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("No JSON object found in LLM response", raw, 0)


def llm_extract_intent(question: str) -> Dict[str, Any]:
    """
    Ask the LLM to parse the user question and return structured intent JSON.

    Returns a dict with keys:
        analysis_type : str  one of: single_period | mom | qoq | yoy |
                                      multi_month | multi_quarter | multi_year
        periods       : list of {"label", "start_date", "end_date"}  (YYYY-MM-DD)
        raw_question  : str

    On failure returns {"analysis_type": "unknown"}.
    """
    model = _get_llm()
    if model is None:
        return {"analysis_type": "unknown"}

    today_str  = datetime.today().strftime("%Y-%m-%d")
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
- "last month"  = full previous calendar month.
- "last week" = the previous completed Monday-Sunday week.
- "last N weeks" = the last N completed Monday-Sunday weeks, excluding the current week.
- For week-based requests like "last week" or "past 4 weeks", return only completed Monday-Sunday weeks and never include the current partial week.
- "this week" = Monday of the current week to today.
- "last day" / "last days" = today only.
- "last N days" = the last N days excluding today.
- "last N months" = the last N completed calendar months, excluding the current month.
- "this month"  = 1st of current month to today.
- "last quarter" = the most recently completed FY quarter.
- "last N quarters" = the last N completed FY quarters, excluding the current quarter.
- "this quarter" / "current quarter" = the FY quarter containing today, from its start date to today.
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
- **DEFAULT**: If the question contains NO date-related keywords, return the ENTIRE current financial year: April 1, {current_fy} to March 31, {current_fy + 1}.
- Supported natural-language date formats include but are not limited to:
    DD Month YYYY, DD/MM/YYYY, DD-MM-YYYY, Month YYYY,
    "5th June", "Q2 FY24", "FY2023", "2023-24", "last 30 days",
    "last week", "last 2 weeks", "this week",
    "april to june 2024", "jan and march", "Q1 and Q3 last year".
Examples of valid JSON responses:
1. Question: "What was the user funnel performance in April and June 2024?"
   JSON:
   {{
     "analysis_type": "multi_month",
     "periods": [
       {{"label": "Apr 2024", "start_date": "2024-04-01", "end_date": "2024-04-30"}},
       {{"label": "Jun 2024", "start_date": "2024-06-01", "end_date": "2024-06-30"}}
     ]
   }}
2. Question: "What was the user funnel performance in last 3 years?"
   JSON:
   {{
     "analysis_type": "multi_year",
     "periods": [
       {{"label": "FY2023", "start_date": "2023-04-01", "end_date": "2024-03-31"}},
       {{"label": "FY2024", "start_date": "2024-04-01", "end_date": "2025-03-31"}},
       {{"label": "FY2025", "start_date": "2025-04-01", "end_date": "2026-03-31"}}
     ]
   }}

3. Question: "What was the user funnel performance in last 2 quarters?"
   JSON:
   {{
     "analysis_type": "multi_quarter",
     "periods": [
       {{"label": "Q3 FY2025", "start_date": "2025-10-01", "end_date": "2025-12-31"}},
       {{"label": "Q4 FY2025", "start_date": "2026-01-01", "end_date": "2026-03-31"}}
     ]
   }}

   REMINDER — FY quarter boundaries (Q1=Apr-Jun, Q2=Jul-Sep, Q3=Oct-Dec, Q4=Jan-Mar):
   - If today is in April, May, or June  → last quarter is Q4 of previous FY
   - If today is in July, Aug, or Sep    → last quarter is Q1 of current FY
   - If today is in Oct, Nov, or Dec     → last quarter is Q2 of current FY
   - If today is in Jan, Feb, or Mar     → last quarter is Q3 of current FY
   - adjust accordingly for "last N quarters" (e.g. if today is in May 2026, last 2 quarters are Q4 FY2025 and Q1 FY2026)


4. Question: "What was the user funnel performance in last 3 months?"
   JSON:
   {{
     "analysis_type": "multi_month",
     "periods": [
       {{"label": "Feb 2026", "start_date": "2026-02-01", "end_date": "2026-02-28"}},
       {{"label": "Mar 2026", "start_date": "2026-03-01", "end_date": "2026-03-31"}},
       {{"label": "Apr 2026", "start_date": "2026-04-01", "end_date": "2026-04-30"}}
     ]
   }}

4b. Question: "show me user funnel for this quarter"  (asked on any date in April–June, e.g. 2026-06-09)
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

4c. Question: "show me user funnel for this week"  (asked on 2026-06-09, a Tuesday)
   JSON:
   {{
     "analysis_type": "single_period",
     "periods": [
       {{"label": "This Week", "start_date": "2026-06-08", "end_date": "2026-06-09"}}
     ]
   }}

   REMINDER — "this week" = Monday of the current week to today (never future dates).
   "last week" = full Monday–Sunday of the immediately preceding week.

5. Question: "Show me mom user funnel for last year?"
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

6. Question: "Show me qoq user funnel for last year?"
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

7. Question: "What was the user funnel performance year on year?"
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

8. Question: "What was the user funnel performance in last 5 days?"
   JSON:
   {{
     "analysis_type": "last_n_days",
     "periods": [
       {{"label": "last 5 days", "start_date": "2026-05-17", "end_date": "2026-05-21"}}
     ]
   }}

    """.strip()

    user_prompt = f'Question: "{question}"\n\nJSON:'

    try:
        raw    = model.generate_text(prompt=f"{system_prompt}\n\n{user_prompt}")
        parsed = _extract_json_object(raw)
        parsed["raw_question"] = question
        logger.info("LLM intent: %s", parsed.get("analysis_type"))
        return parsed
    except Exception as exc:
        logger.warning("LLM parse error (%s) - falling back to regex", exc)
        return {"analysis_type": "unknown"}


# --------------------------------------------
# Salesforce Bearer-Token Cache
# --------------------------------------------
_bearer_token_cache: Dict[str, Any] = {"token": None, "expires_at": None}


def get_salesforce_bearer_token() -> str:
    if (
        _bearer_token_cache["token"]
        and _bearer_token_cache["expires_at"]
        and datetime.now() < _bearer_token_cache["expires_at"] - timedelta(minutes=5)
    ):
        logger.info("Using cached bearer token")
        return _bearer_token_cache["token"]

    logger.info("Generating new Salesforce bearer token ...")
    try:
        resp = requests.post(
            SALESFORCE_TOKEN_URL,
            data={
                "grant_type":    "client_credentials",
                "client_id":     SALESFORCE_CLIENT_ID,
                "client_secret": SALESFORCE_CLIENT_SECRET,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if not token:
            raise ValueError("No access_token in response")
        _bearer_token_cache["token"]      = token
        _bearer_token_cache["expires_at"] = datetime.now() + timedelta(hours=2)
        logger.info("New bearer token obtained")
        return token
    except Exception as exc:
        logger.error("Bearer token generation failed: %s", exc, exc_info=True)
        raise


def fetch_users_from_salesforce() -> pd.DataFrame:
    logger.info("Fetching users from Salesforce API ...")
    token = get_salesforce_bearer_token()
    resp  = requests.get(
        SALESFORCE_USER_API,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "records" not in data:
        logger.error("No 'records' field in Salesforce response")
        return pd.DataFrame()
    df = pd.DataFrame(data["records"])
    logger.info("Fetched %d users from Salesforce", len(df))
    return df


# --------------------------------------------
# Presto Helper
# --------------------------------------------
def query_presto(catalog: str, schema: str, sql: str) -> pd.DataFrame:
    logger.info("Presto → %s.%s", catalog, schema)
    logger.debug("SQL:\n%s", sql)
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
        df   = pd.DataFrame(rows, columns=cols)
        logger.info("Fetched %d rows", len(df))
        return df
    except Exception as exc:
        logger.error("Presto error: %s", exc, exc_info=True)
        raise


def fetch_data_for_period(start_date: date, end_date: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch opportunities and events for a given date range using ISO date objects."""
    date_filter = f"""
        date_parse(replace(trim(created_date_c), '/', '-'), '%d-%m-%Y')
        BETWEEN date_parse('{start_date.strftime("%d-%m-%Y")}', '%d-%m-%Y')
            AND date_parse('{end_date.strftime("%d-%m-%Y")}', '%d-%m-%Y')
    """
    opportunity_sql = f"""
        SELECT Sales_Order_Number_c, created_date_c, owner_name_c, ownerid
        FROM {CATALOG}.{OPPORTUNITY_SCHEMA}.{OPPORTUNITY_TABLE}
        WHERE {date_filter}
    """
    event_sql = f"""
        SELECT Subject_c, Appointment_Status_c, created_date_c, OwnerName_c, ownerid
        FROM {CATALOG}.{EVENT_SCHEMA}.{EVENT_TABLE}
        WHERE {date_filter}
    """
    opportunities = query_presto(CATALOG, OPPORTUNITY_SCHEMA, opportunity_sql)
    events        = query_presto(CATALOG, EVENT_SCHEMA, event_sql)
    return opportunities, events


# --------------------------------------------
# Aggregation Helpers
# --------------------------------------------
NON_ADDITIVE_MARKERS = ["%", ":"]


def _is_additive_key(key: str) -> bool:
    return not any(m in key for m in NON_ADDITIVE_MARKERS)


def _normalize_to_rows(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        return list(data.values())
    return []


def calculate_master_totals(data: Any) -> Dict[str, Union[int, float]]:
    totals: Dict[str, float] = {}
    for row in _normalize_to_rows(data):
        for key, value in row.items():
            if _is_additive_key(key) and isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    return {k: int(v) if float(v).is_integer() else round(v, 2) for k, v in totals.items()}


def extract_rank_filter(question: str) -> Optional[Tuple[str, int]]:
    """
    Parse a top-N / bottom-N intent from the question.

    Returns (rank_type, n) where rank_type is "top" or "bottom",
    or None if no ranking intent is detected.
    """
    q = question.lower()
    for pattern, rank_type in _RANK_PATTERNS:
        m = re.search(pattern, q)
        if m:
            n = int(m.group(1))
            if n > 0:
                logger.info("Rank filter detected: %s %d", rank_type, n)
                return rank_type, n
    return None


def apply_rank_filter(
    funnel_list: List[Dict[str, Any]],
    rank_type:   str,
    n:           int,
    sort_metric: str = _RANK_SORT_METRIC,
) -> List[Dict[str, Any]]:
    """
    Slice a pre-built funnel list to the top-N or bottom-N users.
    """
    if not funnel_list or n <= 0:
        return funnel_list

    def _metric_value(row: Dict[str, Any]) -> float:
        v = row.get(sort_metric, 0)
        return v if isinstance(v, (int, float)) else 0.0

    sorted_desc = sorted(funnel_list, key=_metric_value, reverse=True)

    if rank_type == "top":
        return sorted_desc[:n]

    if rank_type == "bottom":
        sliced = sorted_desc[-n:] if n < len(sorted_desc) else sorted_desc
        return list(reversed(sliced))

    return funnel_list


def sort_funnel_by_numeric_desc(data: Any, return_as_list: bool = False) -> Any:
    if not isinstance(data, dict) or not data:
        return data
    first_val = next(iter(data.values()))
    if isinstance(first_val, dict):
        first_inner = next(iter(first_val.values()), None)
        if isinstance(first_inner, dict):
            return {k: sort_funnel_by_numeric_desc(v, return_as_list) for k, v in sorted(data.items())}

        def _sort_key(item: Tuple) -> float:
            for k, v in item[1].items():
                if isinstance(v, (int, float)) and _is_additive_key(k):
                    return -(v if k == _RANK_SORT_METRIC else v)
            return 0.0

        sorted_items = sorted(data.items(), key=_sort_key)
        if return_as_list:
            return [{"user_name": name, **metrics} for name, metrics in sorted_items]
        return dict(sorted_items)
    return data


# --------------------------------------------
# Sales Funnel Computation (per user)
# --------------------------------------------
def _get_empty_metrics() -> Dict[str, Any]:
    return {
        "Meeting Booked (MB)": 0,
        "Meeting Done (MD)":   0,
        "Sales Done (SD)":     0,
        "MB:MD": 0.0,
        "MB:SD": 0.0,
        "MD:SD": 0.0,
    }


def compute_user_metrics(
    user_opportunities: pd.DataFrame,
    user_events:        pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Compute MB, MD, SD metrics for a single user's opportunities and events."""
    sales_done = meeting_booked = meeting_done = 0

    if not user_opportunities.empty:
        for col in user_opportunities.columns:
            user_opportunities[col] = user_opportunities[col].fillna("").astype(str)
        son = user_opportunities.get(
            "Sales_Order_Number_c",
            pd.Series([""] * len(user_opportunities)),
        ).str.strip().str.lower()
        sales_done = int(((son != "") & (son != "null")).sum())

    if not user_events.empty:
        for col in user_events.columns:
            user_events[col] = user_events[col].fillna("").astype(str)
        subj   = user_events.get("Subject_c", pd.Series([""] * len(user_events))).str.strip().str.lower()
        status = user_events.get("Appointment_Status_c", pd.Series([""] * len(user_events))).str.strip().str.lower()
        meeting_booked = int((subj == "personal appointment booked").sum())
        meeting_done   = int(((subj == "personal appointment booked") & (status == "completed")).sum())

    if sales_done == 0 and meeting_booked == 0 and meeting_done == 0:
        return None

    def _r(n: int, d: int) -> float:
        return round(n / d, 2) if d else 0.0

    return {
        "Meeting Booked (MB)": int(meeting_booked),
        "Meeting Done (MD)":   int(meeting_done),
        "Sales Done (SD)":     int(sales_done),
        "MB:MD": _r(meeting_booked, meeting_done),
        "MB:SD": _r(meeting_booked, sales_done),
        "MD:SD": _r(meeting_done, sales_done),
    }


# --------------------------------------------
# User-Wise Funnel (Owner-ID Mapping)
# --------------------------------------------
def explode_user_projects(users: pd.DataFrame) -> pd.DataFrame:
    u = users.copy()
    u["Project"] = u["Projects__c"].fillna("").str.lower().str.split(";")
    return u.explode("Project").assign(Project=lambda df: df["Project"].str.strip())


def _build_user_id_maps(users_f: pd.DataFrame) -> Tuple[Dict, Set]:
    users_f = users_f.copy()
    users_f["FullName"] = (
        users_f["FirstName"].fillna("").str.strip() + " " +
        users_f["LastName"].fillna("").str.strip()
    ).str.strip()
    id_to_name = dict(zip(users_f["Id"], users_f["FullName"]))
    valid_ids  = set(users_f["Id"])
    return id_to_name, valid_ids


def _per_user_funnel(
    opps_f:     pd.DataFrame,
    events_f:   pd.DataFrame,
    id_to_name: Dict,
    valid_ids:  Set,
) -> Dict[str, Any]:
    """Core per-user aggregation shared by both normal and all-projects paths."""
    opps_f   = opps_f[opps_f["ownerid"].isin(valid_ids)].copy()
    events_f = events_f[events_f["ownerid"].isin(valid_ids)].copy()

    output: Dict[str, Any] = {}
    all_owner_ids = set(opps_f["ownerid"]).union(set(events_f["ownerid"]))

    for owner_id in all_owner_ids:
        opps_user   = opps_f[opps_f["ownerid"] == owner_id]
        events_user = events_f[events_f["ownerid"] == owner_id]
        metrics = compute_user_metrics(opps_user, events_user)
        if metrics is None:
            continue
        display_name = id_to_name.get(owner_id, owner_id)
        output[display_name] = metrics

    return output


def compute_user_wise_funnel(
    opportunities:     pd.DataFrame,
    events:            pd.DataFrame,
    users:             pd.DataFrame,
    start_date:        date,
    end_date:          date,
    active_users_only: bool,
    projects:          Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute user-wise sales funnel metrics for a single period.
    Accepts date objects — date filtering is applied here against the
    raw Presto data which carries DD-MM-YYYY string columns.
    """
    opportunities = opportunities.copy()
    events        = events.copy()
    users         = users.copy()

    def _parse_flex(series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.strip().str.replace("/", "-", regex=False)
        return pd.to_datetime(cleaned, format="%d-%m-%Y", errors="coerce")

    opportunities["_date"] = _parse_flex(opportunities.get("created_date_c", pd.Series(dtype=str)))
    events["_date"]        = _parse_flex(events.get("created_date_c", pd.Series(dtype=str)))

    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)

    opps_f   = opportunities[(opportunities["_date"] >= start_ts) & (opportunities["_date"] <= end_ts)].copy()
    events_f = events[(events["_date"] >= start_ts) & (events["_date"] <= end_ts)].copy()

    logger.info("After date filter → opportunities: %d, events: %d", len(opps_f), len(events_f))

    # User role + active filter
    users_f = users[users["Role_Name__c"] == "Sales Agent"].copy()
    if active_users_only:
        users_f = users_f[users_f["IsActive"] == True].copy()

    # Normalise ownerid in data frames
    opps_f["ownerid"]   = opps_f["ownerid"].astype(str).str.strip()
    events_f["ownerid"] = events_f["ownerid"].astype(str).str.strip()
    opps_f   = opps_f[opps_f["ownerid"] != "nan"]
    events_f = events_f[events_f["ownerid"] != "nan"]

    # Project filter
    if projects == ["all"]:
        return _funnel_all_projects(opps_f, events_f, users_f)

    if projects:
        canonical = [p.strip().lower() for p in projects]
        users_f = users_f[
            users_f["Projects__c"].fillna("").str.lower()
            .apply(lambda x: any(p in x for p in canonical))
        ].copy()

    logger.info("Users after filtering: %d", len(users_f))

    id_to_name, valid_ids = _build_user_id_maps(users_f)
    output = _per_user_funnel(opps_f, events_f, id_to_name, valid_ids)
    return sort_funnel_by_numeric_desc(output)


def _funnel_all_projects(
    opps_f:   pd.DataFrame,
    events_f: pd.DataFrame,
    users_f:  pd.DataFrame,
) -> Dict[str, Any]:
    """Compute user-wise funnel broken down by project."""
    users_proj = explode_user_projects(users_f)
    output: Dict[str, Any] = {}

    for project, users_p in users_proj.groupby("Project", dropna=False):
        project_key = (
            project
            if (project and not (isinstance(project, float) and pd.isna(project)))
            else "UNASSIGNED_PROJECT"
        )
        users_p = users_p.copy()
        id_to_name, valid_ids = _build_user_id_maps(users_p)
        project_output = _per_user_funnel(opps_f, events_f, id_to_name, valid_ids)
        if project_output:
            output[project_key] = project_output

    return sort_funnel_by_numeric_desc(output)


# --------------------------------------------
# Intent Detection
# --------------------------------------------
def detect_projects_from_question(question: str) -> Optional[List[str]]:
    """Returns None (no filter), ["all"] (all projects), or list of canonical names."""
    q = question.lower().strip()
    found = {
        canonical
        for alias, canonical in sorted(PROJECT_ALIASES.items(), key=lambda x: -len(x[0]))
        if alias in q
    }
    if found:
        return list(found)
    if any(phrase in q for phrase in ALL_PROJECT_PHRASES):
        return ["all"]
    return None


def detect_active_users_filter(question: str) -> bool:
    """Returns True to filter active-only, False for no status filter."""
    q = question.lower()
    for phrase in sorted(ACTIVE_STATUS_MAP, key=len, reverse=True):
        if phrase in q:
            return ACTIVE_STATUS_MAP[phrase]
    return False


# --------------------------------------------
# DateResolver & Period Utilities
# --------------------------------------------
MONTH_MAP: Dict[str, int] = {
    **{m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun",
                                        "jul", "aug", "sep", "oct", "nov", "dec"])},
    **{m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may",
                                        "june", "july", "august", "september",
                                        "october", "november", "december"])},
    "sept": 9,
}
WORD_NUM: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
_MONTH_RE = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)"
)


@dataclass
class Period:
    label: str
    start: date
    end:   date

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label":      self.label,
            "start_date": self.start.isoformat(),
            "end_date":   self.end.isoformat(),
        }


def _today() -> datetime:
    return datetime.today()


def _current_fy() -> int:
    t = _today()
    return t.year if t.month >= 4 else t.year - 1


def _fy_quarter(month: int) -> int:
    if 4  <= month <= 6:  return 1
    if 7  <= month <= 9:  return 2
    if 10 <= month <= 12: return 3
    return 4


def _quarter_dates(q: int, fy: int) -> Tuple[date, date]:
    return {
        1: (date(fy,     4,  1), date(fy,     6, 30)),
        2: (date(fy,     7,  1), date(fy,     9, 30)),
        3: (date(fy,     10, 1), date(fy,     12, 31)),
        4: (date(fy + 1, 1,  1), date(fy + 1, 3,  31)),
    }[q]


_LAST_N_QUARTERS_RE = re.compile(
    r"\b(?:last|past|previous)\s+(\d+|" + "|".join(WORD_NUM) + r")\s+quarters?\b",
    re.I,
)


def _last_n_quarter_periods(question: str, now: Optional[datetime] = None) -> List[Period]:
    """
    Return the last N completed FY quarters, excluding the current quarter.

    FY quarters are Apr-Jun, Jul-Sep, Oct-Dec, Jan-Mar. Results are returned
    in chronological order so multi-period output is stable and readable.
    """
    m = _LAST_N_QUARTERS_RE.search(question.lower())
    if not m:
        return []

    raw = m.group(1)
    n = int(raw) if raw.isdigit() else WORD_NUM.get(raw, 0)
    if n <= 0:
        return []

    current = now or _today()
    fy = current.year if current.month >= 4 else current.year - 1
    curr_q = _fy_quarter(current.month)
    q = curr_q - 1 if curr_q > 1 else 4
    q_fy = fy if curr_q > 1 else fy - 1

    quarters: List[Tuple[int, int]] = []
    for _ in range(n):
        quarters.append((q, q_fy))
        q -= 1
        if q == 0:
            q = 4
            q_fy -= 1

    return [
        Period(f"Q{q_num} FY{fy_year}", *_quarter_dates(q_num, fy_year))
        for q_num, fy_year in reversed(quarters)
    ]


def _month_end(year: int, month: int) -> date:
    return date(year, month, monthrange(year, month)[1])


class AnalysisIntent(str, Enum):
    SINGLE  = "single_period"
    MOM     = "mom"
    QOQ     = "qoq"
    YOY     = "yoy"
    MULTI_M = "multi_month"
    MULTI_Q = "multi_quarter"
    MULTI_Y = "multi_year"
    UNKNOWN = "unknown"


_INTENT_KEYWORDS: List[Tuple[AnalysisIntent, List[str]]] = [
    (AnalysisIntent.YOY, ["yoy", "year on year", "year-on-year", "year over year",
                          "yearly comparison", "yoy performance"]),
    (AnalysisIntent.QOQ, ["qoq", "quarter on quarter", "quarter-on-quarter",
                          "quarter wise", "quarter-wise", "quarterly trend",
                          "quarterly comparison", "quarterwise"]),
    (AnalysisIntent.MOM, ["mom", "month on month", "month-on-month",
                          "monthly trend", "month wise", "month-wise",
                          "monthly comparison", "month over month"]),
]
_MULTI_Q_RE = re.compile(r"q[1-4].{0,20}(and|,).{0,20}q[1-4]", re.I)
_MULTI_M_RE = re.compile(_MONTH_RE + r".{0,30}(and|,).{0,30}" + _MONTH_RE, re.I)
_MULTI_Y_RE = re.compile(r"(20\d{2}).{0,20}(and|,).{0,20}20\d{2}", re.I)

_COMPARISON_KEYWORDS: Dict[AnalysisIntent, List[str]] = {
    AnalysisIntent.MOM: ["mom", "month on month", "month-on-month", "month over month"],
    AnalysisIntent.QOQ: ["qoq", "quarter on quarter", "quarter-on-quarter"],
    AnalysisIntent.YOY: ["yoy", "year on year", "year-on-year", "year over year"],
}


def detect_intent_from_keywords(question: str) -> AnalysisIntent:
    q = question.lower()
    for intent, kws in _INTENT_KEYWORDS:
        if any(kw in q for kw in kws):
            return intent
    if _MULTI_Q_RE.search(q): return AnalysisIntent.MULTI_Q
    if _MULTI_M_RE.search(q): return AnalysisIntent.MULTI_M
    if _MULTI_Y_RE.search(q): return AnalysisIntent.MULTI_Y
    return AnalysisIntent.UNKNOWN


def detect_comparison_intents(question: str) -> List[AnalysisIntent]:
    q = question.lower()
    found: List[AnalysisIntent] = []
    for intent, kws in _COMPARISON_KEYWORDS.items():
        if any(kw in q for kw in kws):
            found.append(intent)
    return found


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


class DateResolver:
    def resolve(self, question: str, llm_intent: Optional[Dict] = None) -> List[Period]:
        if llm_intent and llm_intent.get("analysis_type", "unknown") != "unknown":
            periods = self._from_llm(llm_intent)
            if periods:
                return periods
            logger.warning("LLM intent conversion failed; falling back to regex")
        return self._from_regex(question)

    def _from_llm(self, intent: Dict) -> List[Period]:
        quarter_periods = _last_n_quarter_periods(intent.get("raw_question", ""))
        if quarter_periods:
            return quarter_periods

        weekly = self._llm_relative_week_periods(intent.get("raw_question", ""))
        if weekly:
            return weekly
        out = []
        for p in intent.get("periods", []):
            try:
                start = date.fromisoformat(p["start_date"])
                end   = date.fromisoformat(p["end_date"])
                if end < start:
                    logger.warning("Discarding inverted LLM period %s: end before start", p)
                    continue
                out.append(Period(p["label"], start, end))
            except (KeyError, ValueError) as exc:
                logger.warning("Bad LLM period %s: %s", p, exc)
        return out

    @staticmethod
    def _llm_relative_week_periods(question: str) -> List[Period]:
        q               = question.lower().strip()
        today           = _today().date()
        this_week_start = today - timedelta(days=today.weekday())

        m = re.search(r"(?:last|past|previous)\s+(\d+|" + "|".join(WORD_NUM) + r")\s+weeks?\b", q)
        if m:
            raw = m.group(1)
            n   = int(raw) if raw.isdigit() else WORD_NUM[raw]
            start = this_week_start - timedelta(weeks=n)
            return [
                Period(
                    f"Week {idx + 1}",
                    start + timedelta(weeks=idx),
                    start + timedelta(weeks=idx, days=6),
                )
                for idx in range(n)
            ]
        if re.search(r"\b(?:last|past|previous)\s+week\b", q):
            start = this_week_start - timedelta(days=7)
            return [Period("Last Week", start, start + timedelta(days=6))]
        return []

    def _from_regex(self, question: str) -> List[Period]:
        q   = question.lower().strip()
        now = _today()
        fy  = _current_fy()

        for method in (
            lambda: self._date_pair(q),
            lambda: self._last_n_days(q, now),
            lambda: self._last_n_weeks(q, now),
            lambda: self._this_period(q, now, fy),
            lambda: self._last_n_months(q, now),
            lambda: _last_n_quarter_periods(q, now),
            lambda: self._last_n_years(q, now, fy),
            lambda: self._last_quarter(q, now, fy),
            lambda: self._single_month(q, fy),
            lambda: self._month_range(q, fy),
            lambda: self._single_quarter(q, fy),
            lambda: self._explicit_fy(q),
            lambda: self._year_range(q),
        ):
            result = method()
            if result:
                return result if isinstance(result, list) else [result]

        if re.search(r"\b(last|previous)\s+(financial\s+)?year\b", q):
            return [Period(f"FY{fy - 1}", date(fy - 1, 4, 1), date(fy, 3, 31))]

        yr = self._extract_year(q)
        if yr:
            return [Period(f"FY{yr}", date(yr, 4, 1), date(yr + 1, 3, 31))]

        return [Period(f"FY{fy}", date(fy, 4, 1), date(fy + 1, 3, 31))]

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        m = re.search(r"\b(20\d{2})\b", text)
        return int(m.group(1)) if m else None

    @staticmethod
    def _extract_month(text: str) -> Optional[int]:
        for key in sorted(MONTH_MAP, key=len, reverse=True):
            if re.search(rf"\b{key}\b", text):
                return MONTH_MAP[key]
        return None

    @staticmethod
    def _parse_dmy(s: str) -> Optional[date]:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
            try:
                return datetime.strptime(s.strip(), fmt).date()
            except ValueError:
                pass
        return None

    @staticmethod
    def _parse_natural(s: str) -> Optional[date]:
        s = re.sub(r"(st|nd|rd|th)", "", s).strip()
        m = re.search(r"(\d{1,2})\s+" + _MONTH_RE + r"(?:\s+(\d{2,4}))?", s)
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

    def _date_pair(self, q: str) -> Optional[Period]:
        RANGE = r"\s+(?:to|till|until|thru|through|–|-|—)\s+"
        m = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})" + RANGE + r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", q)
        if m:
            d1, d2 = self._parse_dmy(m.group(1)), self._parse_dmy(m.group(2))
            if d1 and d2:
                return Period(f"{d1} to {d2}", min(d1, d2), max(d1, d2))
        m = re.search(r"(\d{1,2}(?:st|nd|rd|th)?)" + RANGE + r"(\d{1,2}(?:st|nd|rd|th)?)\s+" + _MONTH_RE + r"(?:\s+(\d{4}))?", q)
        if m:
            raw_yr = f" {m.group(4)}" if m.group(4) else ""
            d1 = self._parse_natural(f"{m.group(1)} {m.group(3)}{raw_yr}")
            d2 = self._parse_natural(f"{m.group(2)} {m.group(3)}{raw_yr}")
            if d1 and d2:
                return Period(f"{d1} to {d2}", min(d1, d2), max(d1, d2))
        m = re.search(
            r"(\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+(?:\s+\d{2,4})?)" + RANGE +
            r"(\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+(?:\s+\d{2,4})?)",
            q,
        )
        if m:
            d1, d2 = self._parse_natural(m.group(1)), self._parse_natural(m.group(2))
            if d1 and d2:
                return Period(f"{d1} to {d2}", min(d1, d2), max(d1, d2))
        single = self._parse_natural(q)
        if single:
            return Period(str(single), single, single)
        return None

    @staticmethod
    def _last_n_days(q: str, now: datetime) -> Optional[Period]:
        m = re.search(r"(?:last|past)\s+(\d+|" + "|".join(WORD_NUM) + r")\s+days?", q)
        if m:
            raw = m.group(1)
            n   = int(raw) if raw.isdigit() else WORD_NUM[raw]
            return Period(f"Last {n} days", (now - timedelta(days=n - 1)).date(), now.date())
        if re.search(r"\b(?:last|past)\s+days?\b", q):
            return Period("Last day", now.date(), now.date())
        return None

    @staticmethod
    def _last_n_weeks(q: str, now: datetime) -> Optional[Period]:
        today           = now.date()
        this_week_start = today - timedelta(days=today.weekday())
        m = re.search(r"(?:last|past)\s+(\d+|" + "|".join(WORD_NUM) + r")\s+weeks?", q)
        if m:
            raw   = m.group(1)
            n     = int(raw) if raw.isdigit() else WORD_NUM[raw]
            end   = this_week_start - timedelta(days=1)
            start = this_week_start - timedelta(weeks=n)
            return Period(f"Last {n} weeks", start, end)
        if re.search(r"\b(?:last|previous|past)\s+week\b", q):
            end   = this_week_start - timedelta(days=1)
            start = this_week_start - timedelta(days=7)
            return Period("Last Week", start, end)
        return None

    @staticmethod
    def _this_period(q: str, now: datetime, fy: int) -> Optional[Period]:
        today = now.date()
        if re.search(r"\b(this|current)\s+month\b|\bmtd\b", q):
            return Period("This Month (MTD)", today.replace(day=1), today)
        if re.search(r"\b(this|current)\s+week\b", q):
            monday = today - timedelta(days=today.weekday())
            return Period("This Week", monday, today)
        if re.search(r"\b(this|current)\s+(fy|financial\s+year|year)\b|\bytd\b", q):
            return Period(f"FY{fy} YTD", date(fy, 4, 1), today)
        if re.search(r"\b(this|current)\s+quarter\b|\bqtd\b", q):
            q_num = _fy_quarter(today.month)
            s, e  = _quarter_dates(q_num, fy)
            return Period(f"Q{q_num} FY{fy} QTD", s, min(e, today))
        return None

    @staticmethod
    def _last_n_months(q: str, now: datetime) -> Optional[Period]:
        m = re.search(r"last\s+(\d+|" + "|".join(WORD_NUM) + r")\s+months?", q)
        if not m:
            return None
        raw   = m.group(1)
        n     = int(raw) if raw.isdigit() else WORD_NUM[raw]
        first = now.date().replace(day=1)
        end   = first - timedelta(days=1)
        start = end.replace(day=1)
        for _ in range(n - 1):
            start = (start - timedelta(days=1)).replace(day=1)
        return Period(f"Last {n} months", start, end)

    @staticmethod
    def _last_n_quarters(q: str, now: datetime, fy: int) -> Optional[Period]:
        periods = _last_n_quarter_periods(q, now)
        if not periods:
            return None
        return Period(f"Last {len(periods)} quarters", periods[0].start, periods[-1].end)

    @staticmethod
    def _last_n_years(q: str, now: datetime, fy: int) -> Optional[Period]:
        m = re.search(r"last\s+(\d+|" + "|".join(WORD_NUM) + r")\s+years?", q)
        if not m:
            return None
        raw      = m.group(1)
        n        = int(raw) if raw.isdigit() else WORD_NUM[raw]
        end_fy   = fy - 1
        start_fy = end_fy - n + 1
        return Period(f"Last {n} years", date(start_fy, 4, 1), date(end_fy + 1, 3, 31))

    @staticmethod
    def _last_quarter(q: str, now: datetime, fy: int) -> Optional[Period]:
        if not re.search(r"\blast\s+quarter\b|\bprevious\s+quarter\b", q):
            return None
        curr_q  = _fy_quarter(now.month)
        prev_q  = curr_q - 1 if curr_q > 1 else 4
        prev_fy = fy if curr_q > 1 else fy - 1
        s, e    = _quarter_dates(prev_q, prev_fy)
        return Period(f"Q{prev_q} FY{prev_fy}", s, e)

    def _single_month(self, q: str, fy: int) -> Optional[Period]:
        m_num = self._extract_month(q)
        if not m_num:
            return None
        yr = self._extract_year(q) or (fy if m_num >= 4 else fy + 1)
        return Period(datetime(yr, m_num, 1).strftime("%b %Y"), date(yr, m_num, 1), _month_end(yr, m_num))

    def _month_range(self, q: str, fy: int) -> Optional[Period]:
        sep = r"\s*(?:to|till|through|–|-|—|and)\s*"
        m   = re.compile(_MONTH_RE + sep + _MONTH_RE + r"(?:\s+(\d{4}))?", re.I).search(q)
        if not m:
            return None
        m1     = MONTH_MAP.get(m.group(1)[:3])
        m2     = MONTH_MAP.get(m.group(2)[:3])
        yr_raw = m.group(3)
        yr     = int(yr_raw) if yr_raw else fy
        y1     = yr if m1 >= 4 else yr + 1
        y2     = yr if m2 >= 4 else yr + 1
        if m2 < m1:
            y2 += 1
        return Period(
            f"{datetime(y1, m1, 1).strftime('%b %Y')} – {datetime(y2, m2, 1).strftime('%b %Y')}",
            date(y1, m1, 1), _month_end(y2, m2),
        )

    def _single_quarter(self, q: str, fy: int) -> Optional[Period]:
        m = re.search(r"\bq([1-4])\b", q, re.I)
        if not m:
            return None
        q_num = int(m.group(1))
        fy_m  = re.search(r"\bfy\s*(\d{2,4})\b", q, re.I)
        if fy_m:
            v  = int(fy_m.group(1))
            fy = v if v > 100 else 2000 + v
        if re.search(r"\b(last|previous)\s+(year|fy)\b", q):
            fy -= 1
        s, e = _quarter_dates(q_num, fy)
        return Period(f"Q{q_num} FY{fy}", s, e)

    @staticmethod
    def _explicit_fy(q: str) -> Optional[Period]:
        m = re.search(r"\bfy\s*(\d{2,4})\b", q, re.I) or re.search(r"financial\s+year\s+(\d{4})", q, re.I)
        if not m:
            return None
        v  = int(m.group(1))
        yr = v if v > 100 else 2000 + v
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
        lo, hi = min(y1, y2), max(y1, y2)
        return Period(f"FY{lo} to FY{hi}", date(lo, 4, 1), date(hi + 1, 3, 31))


# --------------------------------------------
# Multi-Period Generators
# --------------------------------------------
def _mom_periods(question: str) -> List[Period]:
    q     = question.lower()
    fy    = _current_fy()
    now   = _today()
    today = now.date()

    if re.search(r"\blast\s+quarter\b|\bprevious\s+quarter\b", q):
        curr_q = _fy_quarter(today.month)
        prev_q = curr_q - 1 if curr_q > 1 else 4
        pfy    = fy if curr_q > 1 else fy - 1
        s, e   = _quarter_dates(prev_q, pfy)
    elif re.search(r"\b(this|current)\s+quarter\b|\bqtd\b", q):
        curr_q = _fy_quarter(today.month)
        s, e   = _quarter_dates(curr_q, fy)
        e      = min(e, today)
    elif m := re.search(r"\bq([1-4])\b", q, re.I):
        q_num  = int(m.group(1))
        fy_m   = re.search(r"\bfy\s*(\d{2,4})\b", q, re.I)
        q_fy   = (int(fy_m.group(1)) if int(fy_m.group(1)) > 100 else 2000 + int(fy_m.group(1))) if fy_m else fy
        if re.search(r"\b(last|previous)\s+(year|fy)\b", q):
            q_fy -= 1
        s, e = _quarter_dates(q_num, q_fy)
    elif re.search(r"\b(last|previous)\s+(financial\s+)?(year|fy)\b", q):
        s, e = date(fy - 1, 4, 1), date(fy, 3, 31)
    elif re.search(r"\b(this|current)\s+(financial\s+)?(year|fy)\b|\bytd\b", q):
        s, e = date(fy, 4, 1), today
    else:
        base = _resolver.resolve(question)
        s, e = (base[0].start, base[0].end) if base else (date(fy, 4, 1), today)

    periods: List[Period] = []
    cur = s.replace(day=1)
    while cur <= e:
        me  = _month_end(cur.year, cur.month)
        end = min(me, e, today)
        lbl = cur.strftime("%b %Y")
        if cur.year == today.year and cur.month == today.month:
            lbl += " (MTD)"
        periods.append(Period(lbl, cur, end))
        cur = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
    return periods


def _qoq_periods(question: str) -> List[Period]:
    q   = question.lower()
    fy  = _current_fy()
    now = _today()

    years = sorted(set(int(y) for y in re.findall(r"\b(20\d{2})\b", q)))
    if len(years) >= 2 and (" to " in q or re.search(r"20\d{2}\s*[-–—]\s*20\d{2}", q)):
        return [
            Period(f"Q{qn} FY{yr}", *_quarter_dates(qn, yr))
            for yr in range(years[0], years[-1] + 1)
            for qn in range(1, 5)
        ]

    last_n_quarters = _last_n_quarter_periods(q, now)
    if last_n_quarters:
        return last_n_quarters

    if re.search(r"\blast\s+quarter\b|\bprevious\s+quarter\b", q):
        curr_q = _fy_quarter(now.month)
        pq     = curr_q - 1 if curr_q > 1 else 4
        pfy    = fy if curr_q > 1 else fy - 1
        return [Period(f"Q{pq} FY{pfy}", *_quarter_dates(pq, pfy))]

    fy_m = re.search(r"\bfy\s*(\d{2,4})\b", q, re.I)
    if fy_m:
        v  = int(fy_m.group(1))
        fy = v if v > 100 else 2000 + v
    if re.search(r"\b(last|previous)\s+(year|fy)\b", q):
        fy -= 1
    yr_m = re.search(r"\b(20\d{2})\b", q)
    if yr_m:
        fy = int(yr_m.group(1))

    return [Period(f"Q{i} FY{fy}", *_quarter_dates(i, fy)) for i in range(1, 5)]


def _yoy_periods(question: str) -> List[Period]:
    fy    = _current_fy()
    q     = question.lower()
    today = _today().date()

    if re.search(r"\b(last|previous)\s+(financial\s+)?(year|fy)\b", q):
        return [Period(f"FY{fy - 1}", date(fy - 1, 4, 1), date(fy, 3, 31))]

    ln = DateResolver._last_n_years(q, _today(), fy)
    if ln:
        start_fy = ln.start.year
        end_fy   = ln.end.year - 1
        return [Period(f"FY{y}", date(y, 4, 1), date(y + 1, 3, 31)) for y in range(start_fy, end_fy + 1)]

    years = sorted(set(int(y) for y in re.findall(r"\b(20\d{2})\b", q)))
    if years:
        is_range = " to " in q or re.search(r"20\d{2}\s*[-–—]\s*20\d{2}", q)
        if is_range and len(years) >= 2:
            years = list(range(years[0], years[-1] + 1))
        return [Period(f"FY{y}", date(y, 4, 1), date(y + 1, 3, 31)) for y in years]

    prior = [Period(f"FY{y}", date(y, 4, 1), date(y + 1, 3, 31)) for y in [fy - 3, fy - 2, fy - 1]]
    prior.append(Period(f"FY{fy} YTD", date(fy, 4, 1), today))
    return prior


def _multi_quarter_periods(question: str) -> List[Period]:
    q    = question.lower()
    fy   = _current_fy()
    fy_m = re.search(r"\bfy\s*(\d{2,4})\b", q, re.I)
    if fy_m:
        v  = int(fy_m.group(1))
        fy = v if v > 100 else 2000 + v
    if re.search(r"\b(last|previous)\s+(year|fy)\b", q):
        fy -= 1
    yr_m = re.search(r"\b(20\d{2})\b", q)
    if yr_m:
        fy = int(yr_m.group(1))
    q_nums = [int(x) for x in re.findall(r"\bq([1-4])\b", q, re.I)]
    if not q_nums:
        return []
    if " to " in q:
        q_nums = list(range(min(q_nums), max(q_nums) + 1))
    return [Period(f"Q{n} FY{fy}", *_quarter_dates(n, fy)) for n in sorted(set(q_nums))]


def _multi_month_periods(question: str) -> List[Period]:
    q  = question.lower()
    fy = _current_fy()
    if re.search(r"\b(last|previous)\s+(year|fy)\b", q):
        fy -= 1
    yr_m = re.search(r"\b(20\d{2})\b", q)
    if yr_m:
        fy = int(yr_m.group(1))
    periods: List[Period] = []
    for mn in re.findall(_MONTH_RE, q):
        m_num = MONTH_MAP.get(mn[:3])
        if not m_num:
            continue
        yr = fy if m_num >= 4 else fy + 1
        periods.append(Period(datetime(yr, m_num, 1).strftime("%b %Y"), date(yr, m_num, 1), _month_end(yr, m_num)))
    return periods


def _multi_year_periods(question: str) -> List[Period]:
    q     = question.lower()
    years = sorted(set(int(y) for y in re.findall(r"\b(20\d{2})\b", q)))
    if not years:
        return []
    is_range = " to " in q or re.search(r"20\d{2}\s*[-–—]\s*20\d{2}", q)
    if is_range:
        years = list(range(years[0], years[-1] + 1))
    return [Period(f"FY{y}", date(y, 4, 1), date(y + 1, 3, 31)) for y in years]


# --------------------------------------------
# FastAPI App
# --------------------------------------------
app = FastAPI(
    title="Sales User-Wise Funnel Analytics API",
    description="LLM-first + regex fallback. Supports 30+ date formats.",
    version="3.0.0",
)
_resolver = DateResolver()


def _run_period(
    period:            Period,
    users:             pd.DataFrame,
    active_users_only: bool,
    projects:          Optional[List[str]],
    rank_filter:       Optional[Tuple[str, int]] = None,
) -> Dict[str, Any]:
    """Fetch data for a single period and compute the user-wise sales funnel."""
    logger.info("Processing period %s: %s → %s", period.label, period.start, period.end)
    try:
        opportunities, events = fetch_data_for_period(period.start, period.end)
    except Exception as exc:
        logger.error("DB error for %s: %s", period.label, exc, exc_info=True)
        return {"label": period.label, "error": str(exc)}

    funnel = compute_user_wise_funnel(
        opportunities, events, users, period.start, period.end, active_users_only, projects
    )
    totals      = calculate_master_totals(funnel)
    funnel_list = sort_funnel_by_numeric_desc(funnel, return_as_list=True)

    if rank_filter:
        rank_type, n = rank_filter
        funnel_list  = apply_rank_filter(funnel_list, rank_type, n)
        rank_applied = {"rank_type": rank_type, "n": n, "metric": _RANK_SORT_METRIC}
    else:
        rank_applied = None

    result = {
        "label":  period.label,
        "period": f"{period.start.isoformat()} to {period.end.isoformat()}",
        "funnel": funnel_list,
        "totals": totals,
    }
    if rank_applied:
        result["rank_filter_applied"] = rank_applied
    return result


def _build_single_response(
    period:            Period,
    users:             pd.DataFrame,
    active_users_only: bool,
    projects:          Optional[List[str]],
    rank_filter:       Optional[Tuple[str, int]] = None,
) -> Dict[str, Any]:
    result = _run_period(period, users, active_users_only, projects, rank_filter)
    if "error" in result:
        return {"status": "error", "message": result["error"]}
    if not result.get("funnel"):
        return {
            "status":  "no_data",
            "message": "No users matched with opportunity/event owners in the specified date range",
            "filter":  result.get("period"),
        }
    response = {
        "status":                  "success",
        "analysis_type":           AnalysisIntent.SINGLE,
        "filter":                  result["period"],
        "total_matched_users":     len(result.get("funnel") or []),
        "totals":                  result["totals"],
        "sales_wise_user_metrics": result["funnel"],
    }
    if result.get("rank_filter_applied"):
        response["rank_filter_applied"] = result["rank_filter_applied"]
    return response


def _build_multi_response(
    analysis_type:     str,
    periods:           List[Period],
    users:             pd.DataFrame,
    active_users_only: bool,
    projects:          Optional[List[str]],
    rank_filter:       Optional[Tuple[str, int]] = None,
) -> Dict[str, Any]:
    results         = [_run_period(p, users, active_users_only, projects, rank_filter) for p in periods]
    all_funnels     = [r.get("funnel") or [] for r in results]
    flat_rows       = [row for funnel in all_funnels for row in (funnel if isinstance(funnel, list) else [])]
    combined_totals = calculate_master_totals(flat_rows)
    response = {
        "status":        "success",
        "analysis_type": analysis_type,
        "data":          results,
        "totals":        combined_totals,
    }
    if rank_filter:
        response["rank_filter_applied"] = {
            "rank_type": rank_filter[0],
            "n":         rank_filter[1],
            "metric":    _RANK_SORT_METRIC,
        }
    return response


def _build_comparison_sections(
    question:          str,
    intents:           List[AnalysisIntent],
    users:             pd.DataFrame,
    active_users_only: bool,
    projects:          Optional[List[str]],
    llm_intent:        Optional[Dict[str, Any]] = None,
    rank_filter:       Optional[Tuple[str, int]] = None,
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    seen: Set[AnalysisIntent]   = set()
    llm_analysis_type = (llm_intent or {}).get("analysis_type")
    llm_periods       = _resolver._from_llm(llm_intent) if llm_intent else []

    for intent in intents:
        if intent in seen:
            continue
        seen.add(intent)

        if intent == AnalysisIntent.MOM:
            periods = llm_periods if llm_analysis_type == AnalysisIntent.MOM and llm_periods else _mom_periods(question)
            comparisons["mom"] = _build_multi_response("month_on_month", periods, users, active_users_only, projects, rank_filter)
        elif intent == AnalysisIntent.QOQ:
            periods = llm_periods if llm_analysis_type in {AnalysisIntent.QOQ, AnalysisIntent.MULTI_Q} and llm_periods else _qoq_periods(question)
            comparisons["qoq"] = _build_multi_response("quarter_on_quarter", periods, users, active_users_only, projects, rank_filter)
        elif intent == AnalysisIntent.YOY:
            periods = llm_periods if llm_analysis_type in {AnalysisIntent.YOY, AnalysisIntent.MULTI_Y} and llm_periods else _yoy_periods(question)
            comparisons["yoy"] = _build_multi_response("year_on_year", periods, users, active_users_only, projects, rank_filter)

    return comparisons


@app.post("/funnel/salesuser/question")
async def sales_user_funnel_from_question(payload: dict = Body(...)) -> Dict[str, Any]:
    question = payload.get("question", "").strip()
    if not question:
        return {"status": "error", "message": "question is required"}

    logger.info("Request: %s", question)

    # Shared filters (independent of date logic)
    active_users_only = detect_active_users_filter(question)
    projects          = detect_projects_from_question(question)
    rank_filter       = extract_rank_filter(question)

    try:
        users = fetch_users_from_salesforce()
    except Exception as exc:
        logger.error("User fetch failed: %s", exc, exc_info=True)
        return {"status": "error", "message": f"User fetch failed: {exc}"}

    if users.empty:
        return {"status": "no_data", "message": "No users found from Salesforce API"}

    comparison_intents = detect_comparison_intents(question)
    last_n_quarters = _last_n_quarter_periods(question)
    if last_n_quarters and not comparison_intents:
        logger.info("Using deterministic last-N-quarters resolver: %d periods", len(last_n_quarters))
        return _build_multi_response("multi_quarter", last_n_quarters, users, active_users_only, projects, rank_filter)

    # Step 1: LLM intent extraction
    llm_intent    = llm_extract_intent(question)
    analysis_type = llm_intent.get("analysis_type", "unknown")

    logger.info("LLM intent: %s | Rank filter: %s", analysis_type, rank_filter)

    # Step 2: If LLM gave valid intent, use it
    if analysis_type != "unknown" and not comparison_intents:
        llm_periods = _resolver._from_llm(llm_intent)
        if llm_periods:
            if analysis_type == AnalysisIntent.SINGLE:
                return _build_single_response(llm_periods[0], users, active_users_only, projects, rank_filter)
            return _build_multi_response(analysis_type, llm_periods, users, active_users_only, projects, rank_filter)

    # Step 3: Keyword-based intent detection (fast path)
    kw_intent = detect_intent_from_keywords(question)
    if comparison_intents:
        response: Dict[str, Any] = {
            "status":        "success",
            "analysis_type": "comparison_bundle",
            "comparisons":   _build_comparison_sections(
                question, comparison_intents, users, active_users_only, projects, llm_intent, rank_filter
            ),
        }
        if has_explicit_period_context(question):
            periods = _resolver.resolve(question)
            if len(periods) == 1:
                response["primary"] = _build_single_response(periods[0], users, active_users_only, projects, rank_filter)
            elif periods:
                response["primary"] = _build_multi_response("multi_period", periods, users, active_users_only, projects, rank_filter)
        return response

    if kw_intent == AnalysisIntent.YOY:
        return _build_multi_response("year_on_year", _yoy_periods(question), users, active_users_only, projects, rank_filter)
    if kw_intent == AnalysisIntent.QOQ:
        return _build_multi_response("quarter_on_quarter", _qoq_periods(question), users, active_users_only, projects, rank_filter)
    if kw_intent == AnalysisIntent.MOM:
        return _build_multi_response("month_on_month", _mom_periods(question), users, active_users_only, projects, rank_filter)
    if kw_intent == AnalysisIntent.MULTI_Q:
        periods = _multi_quarter_periods(question)
        if periods:
            return _build_multi_response("multi_quarter", periods, users, active_users_only, projects, rank_filter)
    if kw_intent == AnalysisIntent.MULTI_M:
        periods = _multi_month_periods(question)
        if periods:
            return _build_multi_response("multi_month", periods, users, active_users_only, projects, rank_filter)
    if kw_intent == AnalysisIntent.MULTI_Y:
        periods = _multi_year_periods(question)
        if periods:
            return _build_multi_response("multi_year", periods, users, active_users_only, projects, rank_filter)

    # Step 4: Generic date resolution (regex fallback)
    periods = _resolver.resolve(question, llm_intent)
    if len(periods) == 1:
        return _build_single_response(periods[0], users, active_users_only, projects, rank_filter)
    return _build_multi_response("multi_period", periods, users, active_users_only, projects, rank_filter)


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "Sales User-Wise Funnel Analytics API v3", "llm": MODEL_ID}


@app.get("/supported-formats")
async def supported_formats():
    return {
        "single_date":    ["15 April 2024", "15/04/2024", "15-04-2024", "5th June", "Jun 2024", "April"],
        "date_range":     ["15 April to 30 June 2024", "15/04/2024 to 30/06/2024", "April to June", "April to June 2024"],
        "relative":       ["last day", "last 30 days", "last week", "last 2 weeks", "this week",
                           "last 3 months", "last month", "last quarter", "last year", "last FY",
                           "this month", "this quarter", "this year", "this FY", "MTD", "QTD", "YTD"],
        "quarter":        ["Q1", "Q2 FY24", "Q3 2024", "Q1 and Q3", "Q1 to Q3", "last quarter"],
        "financial_year": ["FY24", "FY2024", "FY 2024", "2023-24", "2022 to 2024", "financial year 2024"],
        "comparative":    ["MOM", "month on month", "month-wise", "QOQ", "quarter on quarter",
                           "quarterly", "YOY", "year on year", "year-over-year"],
        "project_filter": list(PROJECT_ALIASES.keys()) + ["all projects", "project wise"],
        "status_filter":  list(ACTIVE_STATUS_MAP.keys()),
    }
