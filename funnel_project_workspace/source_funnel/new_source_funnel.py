"""
Production-Ready Source-Wise Funnel Analytics API
=================================================
Architecture:
  - LLM (IBM Watsonx / OpenAI-compatible) handles intent + date extraction
  - Rule-based keyword + regex acts as fallback (never the primary path)
  - All date parsing goes through a unified DateResolver (from new_lead_funnel.py)
  - Source-wise funnel computation uses existing Presto queries and metrics
  - FastAPI endpoints: /funnel/source/question (source-wise) and /funnel/leadfunnel/question (aggregate)
"""

from __future__ import annotations

import json
import logging
import os
import re
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

# --------------------------------------------
# Logging & Environment
# --------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("funnel_source_tool.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("funnel_api")

# --------------------------------------------
# Configuration
# --------------------------------------------
CATALOG = os.getenv("CATALOG", "salesforcereport")
LEAD_SCHEMA = os.getenv("LEAD_SCHEMA", "lead_fy_year")
OPP_SCHEMA = os.getenv("OPP_SCHEMA", "opportunity_sf_report")
EVENT_SCHEMA = os.getenv("EVENT_SCHEMA", "event_sf_report")

LEAD_TABLE = os.getenv("LEAD_TABLE", "lead_fy_report")
OPP_TABLE = os.getenv("OPP_TABLE", "opportunity_report")
EVENT_TABLE = os.getenv("EVENT_TABLE", "event_report")

PRESTO_HOST = os.getenv("PRESTO_HOST")
PRESTO_PORT = int(os.getenv("PRESTO_PORT", "443"))
PRESTO_USER = os.getenv("PRESTO_USERNAME")
PRESTO_PASS = os.getenv("PRESTO_PASSWORD")

WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/llama-3-2-3b-instruct")

WORD_NUM: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

_LAST_N_QUARTERS_RE = re.compile(
    r"\b(?:last|past|previous)\s+(\d+|" + "|".join(WORD_NUM) + r")\s+quarters?\b",
    re.I,
)

# --------------------------------------------
# LLM Client (Watsonx)
# --------------------------------------------
_llm_model: Optional[ModelInference] = None

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
            params={"temperature": 0.4, "max_new_tokens": 1045},
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

    today_str = datetime.today().strftime("%Y-%m-%d")
    current_fy = datetime.today().year if datetime.today().month >= 4 else datetime.today().year - 1

    system_prompt = f"""

You are a funnel data analytics intelligent ai bot system.    
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
        parsed = _extract_json_object(raw)
        parsed["raw_question"] = question

        relative_months = DateResolver._llm_relative_month_periods(question)
        if relative_months:
            parsed["analysis_type"] = AnalysisIntent.MULTI_M.value
            parsed["periods"] = [period.to_dict() for period in relative_months]
        elif parsed.get("analysis_type") == AnalysisIntent.MOM.value:
            mom_periods = _mom_periods(question)
            if mom_periods:
                parsed["periods"] = [period.to_dict() for period in mom_periods]

        logger.info("LLM intent: %s", parsed.get("analysis_type"))
        return parsed
    except Exception as exc:
        logger.warning("LLM parse error (%s) - falling back to regex", exc)
        return {"analysis_type": "unknown"}


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
        df = pd.DataFrame(rows, columns=cols)
        logger.info("Fetched %d rows", len(df))
        return df
    except Exception as exc:
        logger.error("Presto error: %s", exc, exc_info=True)
        raise


def fetch_data_for_period(start_date: date, end_date: date) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch leads, opportunities, events for a given date range."""
    date_filter = f"""
        date_parse(replace(trim(created_date_c), '/', '-'), '%d-%m-%Y')
        BETWEEN date_parse('{start_date.strftime("%d-%m-%Y")}', '%d-%m-%Y')
            AND date_parse('{end_date.strftime("%d-%m-%Y")}', '%d-%m-%Y')
    """
    lead_sql = f"""
        SELECT lead_id_c, status, customer_feedback_c, created_date_c, lead_source_c, OwnerId, project_c, product_category_c
        FROM {CATALOG}.{LEAD_SCHEMA}.{LEAD_TABLE}
        WHERE {date_filter}
    """
    opp_sql = f"""
        SELECT opportunity_id_c, lead_id_c, sales_order_number_c, created_date_c, lead_source_c, project_c, project_category_c
        FROM {CATALOG}.{OPP_SCHEMA}.{OPP_TABLE}
        WHERE {date_filter}
    """
    event_sql = f"""
        SELECT OwnerId, Subject_c, Appointment_Status_c, created_date_c, project_c, product_category_c
        FROM {CATALOG}.{EVENT_SCHEMA}.{EVENT_TABLE}
        WHERE {date_filter}
    """
    leads = query_presto(CATALOG, LEAD_SCHEMA, lead_sql)
    opps = query_presto(CATALOG, OPP_SCHEMA, opp_sql)
    events = query_presto(CATALOG, EVENT_SCHEMA, event_sql)
    return leads, opps, events


# --------------------------------------------
# Funnel Computation (Source-Wise)
# --------------------------------------------
NON_ADDITIVE_MARKERS = ["%", ":"]

def _is_additive_key(key: str) -> bool:
    return not any(marker in key for marker in NON_ADDITIVE_MARKERS)

def _normalize_to_rows(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
    return []

def calculate_master_totals(data: Any) -> Dict[str, Union[int, float]]:
    rows = _normalize_to_rows(data)
    totals: Dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if not _is_additive_key(key):
                continue
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    final = {}
    for k, v in totals.items():
        final[k] = int(v) if float(v).is_integer() else round(v, 2)
    return final

def _get_empty_metrics() -> Dict[str, Any]:
    return {
        "Total Leads": 0,
        "Valid Leads": 0,
        "Junk Leads": 0,
        "SOL Leads (Interested)": 0,
        "Meeting Booked": 0,
        "Meeting Done": 0,
        "Sales Done": 0,
        "Junk %": 0,
        "TL:VL": 0,
        "VL:SOL": 0,
        "SOL:MB": 0,
        "MB:MD": 0,
        "MD:SD": 0,
        "TL:SD": 0,
        "VL:SD": 0,
        "SOL:SD": 0,
        "MB:SD": 0
    }

def _compute_funnel_metrics_exact(leads: pd.DataFrame, opps: pd.DataFrame, events: pd.DataFrame) -> Dict[str, Any]:
    total_leads = len(leads)
    if total_leads == 0 and len(opps) == 0:
        return _get_empty_metrics()

    cf_series = leads.get("customer_feedback_c", pd.Series([""] * total_leads))
    cf = cf_series.fillna("").astype(str).str.strip().str.lower()
    junk_leads = (cf == "junk").sum()
    sol_leads = (cf == "interested").sum()
    valid_leads = ((cf != "junk")).sum()

    sales_col = opps.get("sales_order_number_c", pd.Series([""] * len(opps)))
    sales_str = sales_col.fillna("").astype(str).str.strip().str.lower()
    sales_done_count = int(((sales_str != "") & (sales_str != "nan")).sum())

    subj = events.get("Subject_c", pd.Series([""] * len(events))).fillna("").astype(str).str.strip().str.lower()
    status = events.get("Appointment_Status_c", pd.Series([""] * len(events))).fillna("").astype(str).str.strip().str.lower()
    meeting_booked = (subj == "personal appointment booked").sum()
    meeting_done = ((subj == "personal appointment booked") & (status == "completed")).sum()

    junk_percent = round((junk_leads / total_leads) * 100, 2) if total_leads else 0
    def safe_div(n, d): return round(n / d, 2) if d else 0

    return {
        "Total Leads": int(total_leads),
        "Valid Leads": int(valid_leads),
        "Junk Leads": int(junk_leads),
        "SOL Leads (Interested)": int(sol_leads),
        "Meeting Booked": int(meeting_booked),
        "Meeting Done": int(meeting_done),
        "Sales Done": int(sales_done_count),
        "Junk %": junk_percent,
        "TL:VL": safe_div(total_leads, valid_leads),
        "VL:SOL": safe_div(valid_leads, sol_leads),
        "SOL:MB": safe_div(sol_leads, meeting_booked),
        "MB:MD": safe_div(meeting_booked, meeting_done),
        "MD:SD": safe_div(meeting_done, sales_done_count),
        "TL:SD": safe_div(total_leads, sales_done_count),
        "VL:SD": safe_div(valid_leads, sales_done_count),
        "SOL:SD": safe_div(sol_leads, sales_done_count),
        "MB:SD": safe_div(meeting_booked, sales_done_count),
    }

# Project, Product, Source aliases (from original main.py)
PROJECT_ALIASES = {
    "wave city": "wave city", "wavecity": "wave city", "wave_city": "wave city",
    "wmcc sec 32": "wmcc sec 32", "wmcc": "wmcc sec 32", "wmcc sector 32": "wmcc sec 32",
    "wave estate": "wave estate", "estate": "wave estate",
}
PRODUCT_ALIASES = {
    "amore": "AMORE", "armonia": "ARMONIA", "villa": "VILLA", "comm booth": "COMM BOOTH",
    "dream homes": "DREAM HOMES", "eden": "EDEN", "eligo": "ELIGO", "ews": "EWS",
    "ews 410": "EWS_001_(410)", "executive floors": "EXECUTIVE FLOORS",
    "golf range": "Golf Range", "harmony greens": "HARMONY GREENS", "hssc": "HSSC",
    "institutional": "INSTITUTIONAL", "lig": "LIG", "lig 310": "LIG_001_(310)",
    "livork": "LIVORK", "mayfair park": "Mayfair Park", "new plots": "NEW PLOTS",
    "old plots": "OLD PLOTS", "plot res if": "PLOT-RES-IF", "plots comm": "PLOTS-COMM",
    "plots res": "PLOTS-RES", "prime floors": "PRIME FLOORS", "swamanorath": "SWAMANORATH",
    "vasilia": "VASILIA", "veridia": "VERIDIA", "veridia 3": "VERIDIA-3",
    "wave floor": "WAVE FLOOR", "wave galleria": "WAVE GALLERIA", "wave garden": "WAVE GARDEN",
}
SOURCE_ALIASES = {
    "bulk sale": "Bulk Sale", "channel partner": "Channel Partner", "digital": "Digital",
    "direct": "Direct", "direct walkin": "Direct Walkin", "electronic media": "Electronic Media",
    "events / exhibitions": "Events / Exhibitions", "existing customer": "Existing customer",
    "lead reassigned": "Lead Reassigned", "outbound campaign": "Outbound Campaign",
    "outdoor": "Outdoor", "print media": "Print Media", "reference sale": "Reference Sale",
    "referral": "Referral", "sms campaign": "SMS Campaign", "transfered unit": "Transfered Unit",
    "shifting": "Shifting", "word of mouth": "Word of mouth",
}
GENERIC_PRODUCT_TRIGGERS = {"product", "products", "category", "categories"}
GENERIC_PROJECT_TRIGGERS = {"project", "projects"}
SOURCE_TRIGGER_WORDS = {"source", "sources", "channel", "channels", "from"}

def compute_source_wise_funnel(
    leads: pd.DataFrame,
    opps: pd.DataFrame,
    events: pd.DataFrame,
    question: Optional[str] = None
) -> Dict[str, Any]:
    """Compute source-wise funnel with intelligent project/product/source filtering."""
    leads = leads.copy()
    opps = opps.copy()
    events = events.copy()

    group_by_col = "lead_source_c"
    filter_project: List[str] = []
    filter_product: List[str] = []
    filter_source: List[str] = []

    if question:
        q_lower = question.lower().strip()
        # Source detection
        if any(word in q_lower for word in SOURCE_TRIGGER_WORDS):
            matched = {canonical for alias, canonical in SOURCE_ALIASES.items() if alias in q_lower}
            if matched:
                filter_source = list(matched)
        # Project detection
        matched_proj = {canonical for alias, canonical in PROJECT_ALIASES.items() if alias in q_lower}
        if matched_proj:
            filter_project = list(matched_proj)
        # Product detection
        matched_prod = {canonical for alias, canonical in PRODUCT_ALIASES.items() if alias in q_lower}
        if matched_prod:
            filter_product = list(matched_prod)
        # Generic grouping fallback
        if not any([filter_source, filter_project, filter_product]):
            if any(t in q_lower for t in GENERIC_PRODUCT_TRIGGERS):
                group_by_col = "product_category_c"
            elif any(t in q_lower for t in GENERIC_PROJECT_TRIGGERS):
                group_by_col = "project_c"

    # Apply filters
    def apply_filter(df: pd.DataFrame, col: str, values: List[str], normalize: str = "lower") -> pd.DataFrame:
        if not values or col not in df.columns:
            return df
        norm_vals = [v.lower() if normalize == "lower" else v.upper() for v in values]
        temp = df[col].fillna("").astype(str).str.strip()
        temp = temp.str.lower() if normalize == "lower" else temp.str.upper()
        return df[temp.isin(norm_vals)].copy()

    for name, df, col, norm in [
        ("leads", leads, "project_c", "lower"),
        ("opps", opps, "project_c", "lower"),
        ("events", events, "project_c", "lower"),
    ]:
        df = apply_filter(df, col, filter_project, norm)
        if name == "leads": leads = df
        elif name == "opps": opps = df
        else: events = df

    for name, df, col, norm in [
        ("leads", leads, "product_category_c", "upper"),
        ("opps", opps, "project_category_c", "upper"),
        ("events", events, "product_category_c", "upper"),
    ]:
        df = apply_filter(df, col, filter_product, norm)
        if name == "leads": leads = df
        elif name == "opps": opps = df
        else: events = df

    for name, df in [("leads", leads), ("opps", opps)]:
        df = apply_filter(df, "lead_source_c", filter_source, "lower")
        if name == "leads": leads = df
        else: opps = df

    if leads.empty and opps.empty:
        return {
            "analysis": "Filtered funnel",
            "filters_applied": {k: v for k, v in {"project": filter_project, "product": filter_product, "source": filter_source}.items() if v},
            "message": "No data found",
            "sources": {"Overall": _get_empty_metrics()}
        }

    # Grouping
    if group_by_col == "product_category_c":
        for name, df, col in [("leads", leads, "product_category_c"), ("opps", opps, "project_category_c"), ("events", events, "product_category_c")]:
            if col in df.columns:
                df["__group__"] = df[col].fillna("").astype(str).str.strip()
            else:
                df["__group__"] = ""
        group_by_col = "__group__"

    for df in (leads, opps):
        if group_by_col not in df.columns:
            df[group_by_col] = ""
    for df in (leads, opps):
        df["__norm__"] = df[group_by_col].astype(str).str.strip().str.lower()

    display_map = {}
    for df in (leads, opps):
        mask = df["__norm__"] != ""
        if mask.any():
            pairs = df.loc[mask, [group_by_col, "__norm__"]].drop_duplicates("__norm__")
            for norm, orig in zip(pairs["__norm__"], pairs[group_by_col]):
                if norm not in display_map:
                    display_map[norm] = str(orig).strip().title()

    for df in (leads, events):
        if "OwnerId" not in df.columns:
            df["OwnerId"] = ""
        df["__owner__"] = df["OwnerId"].fillna("").astype(str).str.strip()

    all_groups = pd.concat([leads["__norm__"], opps["__norm__"]], ignore_index=True)
    unique_groups = [g for g in all_groups.unique() if g]

    output = {}
    for g in unique_groups:
        name = display_map.get(g, g.title())
        l_g = leads[leads["__norm__"] == g].copy()
        o_g = opps[opps["__norm__"] == g].copy()
        owners = l_g["__owner__"].unique()
        e_g = events[events["__owner__"].isin(owners)].copy()
        output[name] = _compute_funnel_metrics_exact(l_g, o_g, e_g)

    for df in (leads, opps, events):
        df.drop(columns=[c for c in df.columns if c.startswith("__")], inplace=True, errors="ignore")

    result: Dict[str, Any] = {"sources": output}
    filters = {k: v for k, v in {"project": filter_project, "product": filter_product, "source": filter_source}.items() if v}
    if filters:
        result["filters_applied"] = filters
        result["analysis"] = "Filtered funnel"
    elif group_by_col == "__group__":
        result["analysis"] = "Product-wise funnel"
    elif group_by_col == "project_c":
        result["analysis"] = "Project-wise funnel"
    else:
        result["analysis"] = "Source-wise funnel"

    return result

def sort_funnel_by_numeric_desc(data: Any, return_as_list: bool = False) -> Any:
    if not isinstance(data, dict) or not data:
        return data
    first_value = next(iter(data.values()))
    if isinstance(first_value, dict):
        first_inner = next(iter(first_value.values()), None) if first_value else None
        if isinstance(first_inner, dict):
            sorted_data = {}
            for key in sorted(data.keys()):
                sorted_data[key] = sort_funnel_by_numeric_desc(data[key], return_as_list)
            return sorted_data
        else:
            def sort_key(item):
                _, metrics = item
                return -(metrics.get("Total Leads") or next(
                    (v for k, v in metrics.items() if isinstance(v, (int, float)) and "%" not in k and ":" not in k),
                    0
                ))
            sorted_items = sorted(data.items(), key=sort_key)
            if return_as_list:
                return [{"name": name, **metrics} for name, metrics in sorted_items]
            return dict(sorted_items)
    return data


def extract_rank_filter(question: str) -> Optional[Tuple[str, int]]:
    """
    Extract Top N or Bottom N ranking filter from question.
    Returns: ("top", N) or ("bottom", N), or None if no ranking filter found.
    
    Examples:
        "top 5 sources" → ("top", 5)
        "bottom 3 leads" → ("bottom", 3)
        "show me top 10" → ("top", 10)
    """
    q = question.lower().strip()
    
    # Pattern for "top N" or "bottom N"
    pattern = r"\b(top|bottom)\s+(\d+|" + "|".join(WORD_NUM.keys()) + r")\b"
    match = re.search(pattern, q)
    
    if not match:
        return None
    
    rank_type = match.group(1)  # "top" or "bottom"
    n_str = match.group(2)
    
    # Convert word number to digit (e.g., "five" → 5)
    if n_str.isdigit():
        n = int(n_str)
    else:
        n = WORD_NUM.get(n_str, None)
        if n is None:
            return None
    
    return (rank_type, n)


def apply_rank_filter(
    data: List[Dict[str, Any]],
    rank_type: str,
    n: int,
    sort_metric: str = "Total Leads"
) -> List[Dict[str, Any]]:
    if not isinstance(data, list) or not data or n <= 0:
        return data

    # ✅ Sort once, here, by the declared sort_metric
    def get_metric_value(item):
        if isinstance(item, dict):
            value = item.get(sort_metric, 0)
            return value if isinstance(value, (int, float)) else 0
        return 0

    sorted_data = sorted(data, key=get_metric_value, reverse=True)

    if rank_type.lower() == "top":
        return sorted_data[:n]
    elif rank_type.lower() == "bottom":
        # ✅ Take last N from descending list → gives true bottom N, in descending order
        return list(reversed(sorted_data[-n:])) if n < len(sorted_data) else list(reversed(sorted_data))

    return data

# --------------------------------------------
# DateResolver & Multi-Period Generators (from new_lead_funnel.py)
# --------------------------------------------
@dataclass
class Period:
    label: str
    start: date
    end: date
    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "start_date": self.start.isoformat(), "end_date": self.end.isoformat()}

def _today() -> datetime:
    return datetime.today()

def _current_fy() -> int:
    t = _today()
    return t.year if t.month >= 4 else t.year - 1

def _fy_quarter(month: int) -> int:
    for quarter, rng in enumerate((range(4, 7), range(7, 10), range(10, 13)), start=1):
        if month in rng:
            return quarter
    return 4

def _quarter_dates(q: int, fy: int) -> Tuple[date, date]:
    mapping = {
        1: (date(fy, 4, 1), date(fy, 6, 30)),
        2: (date(fy, 7, 1), date(fy, 9, 30)),
        3: (date(fy, 10, 1), date(fy, 12, 31)),
        4: (date(fy + 1, 1, 1), date(fy + 1, 3, 31)),
    }
    return mapping[q]

def _month_end(year: int, month: int) -> date:
    return date(year, month, monthrange(year, month)[1])

MONTH_MAP: Dict[str, int] = {
    **{m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])},
    **{m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])},
    "sept": 9,
}
WORD_NUM: Dict[str, int] = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_MONTH_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

class AnalysisIntent(str, Enum):
    SINGLE = "single_period"
    MOM = "mom"
    QOQ = "qoq"
    YOY = "yoy"
    MULTI_M = "multi_month"
    MULTI_Q = "multi_quarter"
    MULTI_Y = "multi_year"
    UNKNOWN = "unknown"

_INTENT_KEYWORDS: List[Tuple[AnalysisIntent, List[str]]] = [
    (AnalysisIntent.YOY, ["yoy", "year on year", "year-on-year", "year over year", "yearly comparison", "yoy performance"]),
    (AnalysisIntent.QOQ, ["qoq", "quarter on quarter", "quarter-on-quarter", "quarter wise", "quarter-wise", "quarterly trend", "quarterly comparison", "quarterwise"]),
    (AnalysisIntent.MOM, ["mom", "month on month", "month-on-month", "monthly trend", "month wise", "month-wise", "monthly comparison", "month over month"]),
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
        raw_q = intent.get("raw_question", "")

        # --- deterministic overrides: never trust LLM for relative-date phrases ---
        quarter_periods = _last_n_quarter_periods(raw_q)
        if quarter_periods:
            return quarter_periods

        # "last quarter" / "previous quarter" — compute from today, not from LLM
        if re.search(r"\blast\s+quarter\b|\bprevious\s+quarter\b", raw_q, re.I):
            now = _today()
            fy = _current_fy()
            p = self._last_quarter(raw_q.lower(), now, fy)
            if p:
                return [p]

        weekly_periods = self._llm_relative_week_periods(raw_q)
        if weekly_periods:
            return weekly_periods

        monthly_periods = self._llm_relative_month_periods(raw_q)
        if monthly_periods:
            return monthly_periods

        out = []
        for p in intent.get("periods", []):
            try:
                out.append(Period(p["label"], date.fromisoformat(p["start_date"]), date.fromisoformat(p["end_date"])))
            except (KeyError, ValueError) as exc:
                logger.warning("Bad LLM period %s: %s", p, exc)
        return out

    @staticmethod
    def _llm_relative_week_periods(question: str) -> List[Period]:
        q = question.lower().strip()
        if not q:
            return []

        today = _today().date()
        this_week_start = today - timedelta(days=today.weekday())

        m = re.search(r"(?:last|past|previous)\s+(\d+|" + "|".join(WORD_NUM) + r")\s+weeks?\b", q)
        if m:
            n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUM[m.group(1)]
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
            end = start + timedelta(days=6)
            return [Period("Last Week", start, end)]

        return []

    @staticmethod
    def _llm_relative_month_periods(question: str) -> List[Period]:
        q = question.lower().strip()
        m = re.search(r"(?:last|past|previous)\s+(\d+|" + "|".join(WORD_NUM) + r")\s+months?\b", q)
        if not m:
            return []

        n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUM[m.group(1)]
        first_this = _today().date().replace(day=1)
        periods: List[Period] = []
        current = first_this - timedelta(days=1)
        for _ in range(n):
            start = current.replace(day=1)
            periods.append(Period(current.strftime("%b %Y"), start, current))
            current = start - timedelta(days=1)
        return list(reversed(periods))

    def _from_regex(self, question: str) -> List[Period]:
        q = question.lower().strip()
        now = _today()
        fy = _current_fy()
        pair = self._parse_date_pair(q)
        if pair:
            s, e = pair
            return [Period(f"{s} to {e}", s, e)]
        p = self._last_n_days(q, now)
        if p:
            return [p]
        p = self._last_n_weeks(q, now)
        if p:
            return [p]
        p = self._this_period(q, now, fy)
        if p:
            return [p]
        p = _last_n_quarter_periods(q, now)
        if p:
            return p
        p = self._last_n_months(q, now)
        if p:
            return [p]
        p = self._last_n_quarters(q, now, fy)
        if p:
            return [p]
        
        p = self.last_n_years(q, now, fy)
        if p:
            return [p]
        
        p = self._last_quarter(q, now, fy)
        if p:
            return [p]
        p = self._single_month(q, fy)
        if p:
            return [p]
        p = self._month_range(q, fy)
        if p:
            return [p]
        p = self._single_quarter(q, fy)
        if p:
            return [p]
        p = self._explicit_fy(q)
        if p:
            return [p]
        p = self._year_range(q)
        if p:
            return [p]
        yr = self._extract_year(q)
        if yr:
            return [Period(f"FY{yr}", date(yr, 4, 1), date(yr + 1, 3, 31))]
        if re.search(r"\b(last|previous)\s+(financial\s+)?year\b", q):
            return [Period(f"FY{fy - 1}", date(fy - 1, 4, 1), date(fy, 3, 31))]
        return [Period(f"FY{fy}", date(fy, 4, 1), date(fy + 1, 3, 31))]

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
        slash = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\s+to\s+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", q)
        if slash:
            d1 = DateResolver._parse_dmy(slash.group(1))
            d2 = DateResolver._parse_dmy(slash.group(2))
            if d1 and d2:
                return min(d1, d2), max(d1, d2)
        nl = re.search(r"(\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+(?:\s+\d{2,4})?)\s+to\s+(\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+(?:\s+\d{2,4})?)", q)
        if nl:
            d1 = DateResolver._parse_natural(nl.group(1))
            d2 = DateResolver._parse_natural(nl.group(2))
            if d1 and d2:
                return min(d1, d2), max(d1, d2)
        sm = re.search(r"(\d{1,2})\s+to\s+(\d{1,2})\s+" + _MONTH_RE + r"(?:\s+(\d{4}))?", q)
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
        single = DateResolver._parse_natural(q)
        if single:
            return single, single
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
        day = int(m.group(1))
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
        m = re.search(r"(?:last|past)\s+(\d+|" + "|".join(WORD_NUM) + r")\s+days?", q)
        if m:
            n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUM[m.group(1)]
            s = (now - timedelta(days=n - 1)).date()
            return Period(f"Last {n} days", s, now.date())
        if re.search(r"\b(?:last|past)\s+days?\b", q):
            today = now.date()
            return Period("Last day", today, today)
        return None
    @staticmethod
    def _last_n_weeks(q: str, now: datetime) -> Optional[Period]:
        today = now.date()
        this_week_start = today - timedelta(days=today.weekday())

        m = re.search(r"(?:last|past)\s+(\d+|" + "|".join(WORD_NUM) + r")\s+weeks?", q)
        if m:
            n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUM[m.group(1)]
            end = this_week_start - timedelta(days=1)
            start = this_week_start - timedelta(weeks=n)
            return Period(f"Last {n} weeks", start, end)

        if re.search(r"\b(?:last|previous|past)\s+week\b", q):
            end = this_week_start - timedelta(days=1)
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
            s, e = _quarter_dates(q_num, fy)
            return Period(f"Q{q_num} FY{fy} QTD", s, min(e, today))
        return None
    @staticmethod
    def _last_n_months(q: str, now: datetime) -> Optional[Period]:
        m = re.search(r"last\s+(\d+|" + "|".join(WORD_NUM) + r")\s+months?", q)
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
        periods = _last_n_quarter_periods(q, now)
        if not periods:
            return None
        return Period(f"Last {len(periods)} quarters", periods[0].start, periods[-1].end)
    
    @staticmethod
    def last_n_years(q: str, now: datetime, fy: int) -> Optional[Period]:
        m = re.search(r"last\s+(\d+|" + "|".join(WORD_NUM) + r")\s+years?", q)
        if not m:
            return None
        n = int(m.group(1)) if m.group(1).isdigit() else WORD_NUM[m.group(1)]
        end_fy = fy - 1 if re.search(r"\b(last|previous)\s+(financial\s+)?year\b", q) else fy
        start_fy = end_fy - n + 1
        return Period(f"Last {n} years", date(start_fy, 4, 1), date(end_fy + 1, 3, 31))

    @staticmethod
    def _last_quarter(q: str, now: datetime, fy: int) -> Optional[Period]:
        if not re.search(r"\blast\s+quarter\b|\bprevious\s+quarter\b", q):
            return None
        curr_q = _fy_quarter(now.month)
        prev_q = curr_q - 1 if curr_q > 1 else 4
        prev_fy = fy if curr_q > 1 else fy - 1
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
        return Period(datetime(yr, m_num, 1).strftime("%b %Y"), date(yr, m_num, 1), _month_end(yr, m_num))
    @staticmethod
    def _month_range(q: str, fy: int) -> Optional[Period]:
        sep = r"\s*(?:to|till|through|–|-|—|and)\s*"
        pat = re.compile(_MONTH_RE + sep + _MONTH_RE + r"(?:\s+(\d{4}))?", re.I)
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
        return Period(f"{datetime(y1, m1, 1).strftime('%b %Y')} – {datetime(y2, m2, 1).strftime('%b %Y')}", date(y1, m1, 1), _month_end(y2, m2))
    @staticmethod
    def _single_quarter(q: str, fy: int) -> Optional[Period]:
        m = re.search(r"\bq([1-4])\b", q, re.I)
        if not m:
            return None
        q_num = int(m.group(1))
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
        m = re.search(r"\bfy\s*(\d{2,4})\b", q, re.I)
        if not m:
            if "financial year" not in q:
                return None
            m = re.search(r"financial\s+year\s+(\d{4})", q, re.I)
            if not m:
                return None
        v = int(m.group(1))
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
        return Period(f"FY{min(y1, y2)} to FY{max(y1, y2)}", date(min(y1, y2), 4, 1), date(max(y1, y2) + 1, 3, 31))

def _normalize(text: str) -> str:
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

    # "last N quarters" — must be checked BEFORE bare "last quarter"
    last_n_q_periods = _last_n_quarter_periods(question)
    if last_n_q_periods:
        # Expand all N quarters into one continuous date span for MOM
        s, e = last_n_q_periods[0].start, last_n_q_periods[-1].end

    elif has_last and has_quarter and quarter_num is None:
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

    while cur <= min(e, today):
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

    years = sorted(set(int(y) for y in re.findall(r"\b(20\d{2})\b", q)))
    is_year_range = len(years) >= 2 and (" to " in q or re.search(r"20\d{2}\s*[-â€“â€”]\s*20\d{2}", q))
    if is_year_range:
        periods: List[Period] = []
        for year in range(years[0], years[-1] + 1):
            for quarter in range(1, 5):
                periods.append(Period(f"Q{quarter} FY{year}", *_quarter_dates(quarter, year)))
        return periods

    if (last_n := DateResolver._last_n_quarters(q, now, fy)):
        periods: List[Period] = []
        cur = last_n.start
        while cur <= last_n.end:
            q_num = _fy_quarter(cur.month)
            q_fy = cur.year if cur.month >= 4 else cur.year - 1
            periods.append(Period(f"Q{q_num} FY{q_fy}", *_quarter_dates(q_num, q_fy)))

            next_month = cur.month + 3
            next_year = cur.year
            if next_month > 12:
                next_month -= 12
                next_year += 1
            cur = date(next_year, next_month, 1)
        return periods

    if re.search(r"\blast\s+quarter\b|\bprevious\s+quarter\b", q):
        curr_q = _fy_quarter(now.month)
        prev_q = curr_q - 1 if curr_q > 1 else 4
        prev_fy = fy if curr_q > 1 else fy - 1
        return [Period(f"Q{prev_q} FY{prev_fy}", *_quarter_dates(prev_q, prev_fy))]

    fy_m = re.search(r"\bfy\s*(\d{2,4})\b", q, re.I)
    if fy_m:
        v = int(fy_m.group(1)); fy = v if v > 100 else 2000 + v
    if re.search(r"\b(last|previous)\s+(year|fy)\b", q):
        fy -= 1
    yr_m = re.search(r"\b(20\d{2})\b", q)
    if yr_m:
        fy = int(yr_m.group(1))
    return [
        Period(f"Q{i} FY{fy}", *_quarter_dates(i, fy))
        for i in range(1, 5)
    ]
def _yoy_periods(question: str) -> List[Period]:
    fy = _current_fy()
    q = question.lower()
    today = _today().date()

    if re.search(r"\b(last|previous)\s+(financial\s+)?(year|fy)\b|\blast\s+years\b", q):
        return [Period(f"FY{fy - 1}", date(fy - 1, 4, 1), date(fy, 3, 31))]

    if (last_n := DateResolver.last_n_years(q, _today(), fy)):
        start_fy = last_n.start.year
        end_fy = last_n.end.year - 1
        return [
            Period(f"FY{year}", date(year, 4, 1), date(year + 1, 3, 31))
            for year in range(start_fy, end_fy + 1)
        ]

    years = _multi_year_periods(question)
    if years:
        return years

    yr_m = re.search(r"\b(20\d{2})\b", q)
    if yr_m:
        fy = int(yr_m.group(1))
        return [Period(f"FY{fy}", date(fy, 4, 1), date(fy + 1, 3, 31))]

    periods = [
        Period(f"FY{past_fy}", date(past_fy, 4, 1), date(past_fy + 1, 3, 31))
        for past_fy in [fy - 3, fy - 2, fy - 1]
    ]
    periods.append(Period(f"FY{fy} YTD", date(fy, 4, 1), today))
    return periods


def _multi_quarter_periods(question: str) -> List[Period]:
    q = question.lower()
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
    q = question.lower()
    fy = _current_fy()
    if re.search(r"\b(last|previous)\s+(year|fy)\b", q): fy -= 1
    yr_m = re.search(r"\b(20\d{2})\b", q)
    if yr_m: fy = int(yr_m.group(1))
    month_names = re.findall(_MONTH_RE, q)
    periods = []
    for mn in month_names:
        m_num = MONTH_MAP.get(mn[:3])
        if not m_num: continue
        yr = fy if m_num >= 4 else fy + 1
        s = date(yr, m_num, 1)
        e = _month_end(yr, m_num)
        periods.append(Period(datetime(yr, m_num, 1).strftime("%b %Y"), s, e))
    return periods

def _multi_year_periods(question: str) -> List[Period]:
    q = question.lower()
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
app = FastAPI(title="Lead Funnel Analytics API", description="Source-wise funnel with LLM intent extraction", version="3.0.0")
_resolver = DateResolver()

def _run_period(period: Period, question: Optional[str] = None) -> Dict[str, Any]:
    try:
        leads, opps, events = fetch_data_for_period(period.start, period.end)
    except Exception as exc:
        logger.error("DB error for %s: %s", period.label, exc)
        return {"label": period.label, "error": str(exc)}
    funnel = compute_source_wise_funnel(leads, opps, events, question=question)
    # Also compute totals from the sources dict
    sources_data = funnel.get("sources", {})
    totals = calculate_master_totals(sources_data)
    
    # Apply rank filter (top/bottom N) if specified in question
    # ✅ Pass sources dict only
    funnel_list = sort_funnel_by_numeric_desc(funnel.get("sources", {}), return_as_list=True)
    sources_list = [{"name": name, **metrics} for name, metrics in funnel.get("sources", {}).items()]

    if question:
        rank_filter = extract_rank_filter(question)
        if rank_filter:
            rank_type, n = rank_filter
            funnel_list = apply_rank_filter(sources_list, rank_type, n)
        else:
            funnel_list = sort_funnel_by_numeric_desc(funnel.get("sources", {}), return_as_list=True)
    else:
        funnel_list = sort_funnel_by_numeric_desc(funnel.get("sources", {}), return_as_list=True)    
    return {
        "label": period.label,
        "period": f"{period.start.isoformat()} to {period.end.isoformat()}",
        "funnel": funnel_list,
        "funnel_dict": funnel,
        "totals": totals,
    }

def _build_single_response(period: Period, question: Optional[str] = None) -> Dict[str, Any]:
    result = _run_period(period, question)
    response = {
        "status": "success",
        "analysis_type": AnalysisIntent.SINGLE,
        "filter": result.get("period"),
        "source_wise_metrics": result.get("funnel"),
        "totals": result.get("totals"),
    }
    
    # Add rank filter info if applied
    if question:
        rank_filter = extract_rank_filter(question)
        if rank_filter:
            rank_type, n = rank_filter
            response["rank_filter_applied"] = {
                "type": rank_type,
                "n": n,
                "sort_metric": "Total Leads"
            }
    
    return response

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
            comparisons["mom"] = _build_multi_response("month_on_month", periods, question)
        elif intent == AnalysisIntent.QOQ:
            periods = llm_periods if llm_analysis_type in {AnalysisIntent.QOQ, AnalysisIntent.MULTI_Q} and llm_periods else _qoq_periods(question)
            comparisons["qoq"] = _build_multi_response("quarter_on_quarter", periods, question)
        elif intent == AnalysisIntent.YOY:
            periods = llm_periods if llm_analysis_type in {AnalysisIntent.YOY, AnalysisIntent.MULTI_Y} and llm_periods else _yoy_periods(question)
            comparisons["yoy"] = _build_multi_response("year_on_year", periods, question)

    return comparisons


def _build_multi_response(analysis_type: str, periods: List[Period], question: Optional[str] = None) -> Dict[str, Any]:
    results = [_run_period(p, question) for p in periods]
    # Aggregated totals across all periods
    all_funnels = [r.get("funnel_dict", {}).get("sources", {}) for r in results]
    combined = {}
    for funnel_dict in all_funnels:
        for source, metrics in funnel_dict.items():
            if source not in combined:
                combined[source] = {}
            for k, v in metrics.items():
                if _is_additive_key(k) and isinstance(v, (int, float)):
                    combined[source][k] = combined[source].get(k, 0) + v
    # Recompute ratios for combined totals? Keep simple sums.
    total_combined = calculate_master_totals(combined)
    
    response = {
        "status": "success",
        "analysis_type": analysis_type,
        "data": results,
        "totals": total_combined,
    }
    
    # Add rank filter info if applied
    if question:
        rank_filter = extract_rank_filter(question)
        if rank_filter:
            rank_type, n = rank_filter
            response["rank_filter_applied"] = {
                "type": rank_type,
                "n": n,
                "sort_metric": "Total Leads"
            }
    
    return response

@app.post("/funnel/source/question")
async def source_funnel_from_question(payload: dict = Body(...)) -> Dict[str, Any]:
    question = payload.get("question", "").strip()
    if not question:
        return {"status": "error", "message": "question is required"}
    logger.info("Request: %s", question)
    comparison_intents = detect_comparison_intents(question)

    # Step 1: LLM intent extraction
    llm_intent = llm_extract_intent(question)
    analysis_type = llm_intent.get("analysis_type", "unknown")

    print(f"LLM intent: {llm_intent}")

    # Step 2: If LLM gave valid intent, use it
    if analysis_type != "unknown" and not comparison_intents:
        llm_periods = _resolver._from_llm(llm_intent)
        if llm_periods:
            if analysis_type == AnalysisIntent.SINGLE:
                return _build_single_response(llm_periods[0], question)
            return _build_multi_response(analysis_type, llm_periods, question)

    # Step 3: Keyword-based intent detection (fast path)
    kw_intent = detect_intent_from_keywords(question)
    last_n_quarters = _last_n_quarter_periods(question)
    if last_n_quarters and not comparison_intents:
        logger.info("Using deterministic last-N-quarters resolver: %d periods", len(last_n_quarters))
        return _build_multi_response(analysis_type, last_n_quarters, question)

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
    
    print(f"Keyword intent: {kw_intent}")
    if kw_intent == AnalysisIntent.YOY:
        periods = _yoy_periods()
        return _build_multi_response("year_on_year", periods, question)
    if kw_intent == AnalysisIntent.QOQ:
        periods = _qoq_periods(question)
        return _build_multi_response("quarter_on_quarter", periods, question)
    if kw_intent == AnalysisIntent.MOM:
        periods = _mom_periods(question)
        return _build_multi_response("month_on_month", periods, question)
    if kw_intent == AnalysisIntent.MULTI_Q:
        periods = _multi_quarter_periods(question)
        if periods:
            return _build_multi_response("multi_quarter", periods, question)
    if kw_intent == AnalysisIntent.MULTI_M:
        periods = _multi_month_periods(question)
        if periods:
            return _build_multi_response("multi_month", periods, question)
    if kw_intent == AnalysisIntent.MULTI_Y:
        periods = _multi_year_periods(question)
        if periods:
            return _build_multi_response("multi_year", periods, question)

    # Step 4: Generic date resolution (regex fallback)
    periods = _resolver.resolve(question, llm_intent)
    if len(periods) == 1:
        return _build_single_response(periods[0], question)
    return _build_multi_response("multi_period", periods, question)

@app.post("/funnel/leadfunnel/question")
async def lead_funnel_aggregate(payload: dict = Body(...)):
    """Aggregate funnel (no source breakdown) - compatibility with new_lead_funnel.py"""
    # This endpoint returns aggregated funnel without source breakdown.
    question = payload.get("question", "").strip()
    if not question:
        return {"status": "error", "message": "question is required"}
    # Reuse same logic but flatten sources
    result = await source_funnel_from_question(payload)
    if result.get("status") == "success":
        # Convert source_wise_metrics to simple funnel by summing all sources
        if "source_wise_metrics" in result:
            aggregated = calculate_master_totals(result["source_wise_metrics"])
            result["lead_funnel"] = aggregated
            del result["source_wise_metrics"]
        elif "data" in result:
            # multi-period case
            for item in result["data"]:
                source_map = item.get("funnel_dict", {}).get("sources", {})
                if source_map:
                    item["funnel"] = calculate_master_totals(source_map)
                item.pop("funnel_dict", None)
    return result

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "Source-Wise Funnel Analytics API v3", "llm": MODEL_ID}

@app.get("/supported-formats")
async def supported_formats():
    return {
        "single_date": ["15 April 2024", "15/04/2024", "15-04-2024", "5th June", "Jun 2024", "April"],
        "date_range": ["15 April to 30 June 2024", "15/04/2024 to 30/06/2024", "April to June", "April to June 2024", "April 2022 to June 2025"],
        "relative": ["last day", "last days", "last 30 days", "last week", "last 2 weeks", "this week", "last 3 months", "last month", "last quarter", "last year", "last FY", "this month", "this quarter", "this year", "this FY", "MTD", "QTD", "YTD"],
        "quarter": ["Q1", "Q2 FY24", "Q3 2024", "Q1 and Q3", "Q1 to Q3", "last quarter"],
        "financial_year": ["FY24", "FY2024", "FY 2024", "2023-24", "2022 to 2024", "financial year 2024"],
        "comparative": ["MOM", "month on month", "month-wise", "QOQ", "quarter on quarter", "quarterly", "YOY", "year on year", "year-over-year"],
        "ranking": ["top 5", "top 10", "bottom 3", "top five sources", "bottom 10 leads", "top 2 projects", "show me top 5", "give me bottom 3"],
    }
