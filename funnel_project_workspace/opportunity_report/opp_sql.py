

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from prestodb.auth import BasicAuthentication
import prestodb
import re
import os
import json
import logging
from datetime import datetime, date, timedelta
from calendar import monthrange
from enum import Enum
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple, Union

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENVIRONMENT / CONNECTION CONFIG
# ============================================================================
PRESTO_HOST     = os.getenv("PRESTO_hostname")
PRESTO_PORT     = int(os.getenv("PRESTO_portnumber", "30984"))
PRESTO_USER     = os.getenv("PRESTO_username")
PRESTO_PASSWORD = os.getenv("PRESTO_password")
PRESTO_CATALOG  = "salesforcereport"
PRESTO_SCHEMA   = "opportunity_sf_report"
PRESTO_TABLE    = "opportunity_report"

WATSONX_API_KEY    = os.getenv("API_KEY")
WATSONX_PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/llama-3-3-70b-instruct")

credentials = Credentials(url=WATSONX_URL, api_key=PRESTO_PASSWORD)
llm_model = ModelInference(
    model_id=MODEL_ID,
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID,
    params={"max_new_tokens": 500, "temperature": 0.1}
)

CITY_LIST = [
    "Noida",
    "Greater Noida",
    "Delhi",
    "Gurgaon",
    "Ghaziabad",
    "Faridabad",
    "Mumbai",
    "Pune",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Lucknow",
    "Kanpur",
    "Jaipur",
    "Ahmedabad",
    "Surat",
    "Indore",
    "Nagpur",
    "Patna"
]


# ============================================================================
# QUERY TYPE ENUM
# ============================================================================
class QueryType(str, Enum):
    SINGLE_DATE            = "single_date"
    DATE_RANGE             = "date_range"
    LAST_N_DAYS            = "last_n_days"
    LAST_N_WEEKS           = "last_n_weeks"
    THIS_WEEK              = "this_week"
    LAST_N_MONTHS          = "last_n_months"
    LAST_N_YEARS           = "last_n_years"
    LAST_N_QUARTERS        = "last_n_quarters"
    THIS_MONTH             = "this_month"
    THIS_QUARTER           = "this_quarter"
    THIS_YEAR              = "this_year"
    LAST_MONTH             = "last_month"
    LAST_QUARTER           = "last_quarter"
    LAST_YEAR              = "last_year"
    QUARTER_WISE           = "quarter_wise"
    MONTH_WISE             = "month_wise"
    YEAR_WISE              = "year_wise"
    SPECIFIC_YEAR          = "specific_year"
    SPECIFIC_DATE          = "specific_date"
    CURRENT_FY             = "current_fy"
    MONTH_RANGE            = "month_range"
    MONTH_RANGE_MONTH_WISE = "month_range_month_wise"
    MULTI_DATE_RANGE       = "MULTI_DATE_RANGE"
    MONTH_MULTI_MONTH_WISE = "month_multi_month_wise"
    MULTI_MONTH            = "multi_month"


class AggregationType(str, Enum):
    COUNT = "count"
    SUM   = "sum"
    AVG   = "avg"
    MIN   = "min"
    MAX   = "max"


# ============================================================================
# API MODELS
# ============================================================================
class OppSQLRequest(BaseModel):
    question:  str  = Field(..., description="Natural language query about opportunities/sales")
    catalog:   str  = Field(default=PRESTO_CATALOG)
    db_schema: str  = Field(default=PRESTO_SCHEMA)
    table:     str  = Field(default=PRESTO_TABLE)

class DateRange(BaseModel):
    start_date: str
    end_date:   Optional[str] = None
    label:      Optional[str] = None

class OppSQLResponse(BaseModel):
    status:             str
    query_type:         str
    sql:                str
    schema_metadata:    list | None = Field(default=None, alias="schema")
    data:               list | None = None
    execution:          dict | None = None
    date_ranges:        List[DateRange]
    is_valid:           bool
    validation_message: Optional[str]           = None
    metadata:           Optional[Dict[str, Any]] = None
    intent_summary:     Optional[Dict[str, Any]] = None
    totals:             Optional[Dict[str, Any]] = None

    class Config:
        validate_by_name = True


# ============================================================================
# DATE PARSER  — YYYYMMDD integer format, Apr–Mar FY
# ============================================================================
class DateParser:
    MONTH_MAP = {
        "jan": 1,  "january": 1,   "feb": 2,  "february": 2,
        "mar": 3,  "march": 3,     "apr": 4,  "april": 4,
        "may": 5,  "jun": 6,       "june": 6,
        "jul": 7,  "july": 7,      "aug": 8,  "august": 8,
        "sep": 9,  "sept": 9,      "september": 9,
        "oct": 10, "october": 10,  "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    WORD_TO_NUM = {
        "one": 1,  "two": 2,   "three": 3, "four": 4,
        "five": 5, "six": 6,   "seven": 7, "eight": 8,
        "nine": 9, "ten": 10,  "eleven": 11, "twelve": 12,
    }

    @staticmethod
    def get_current_fy(today=None) -> int:
        if not today:
            today = datetime.today()
        return today.year if today.month >= 4 else today.year - 1

    @staticmethod
    def get_fy_quarter(month: int) -> int:
        if 4 <= month <= 6:   return 1
        if 7 <= month <= 9:   return 2
        if 10 <= month <= 12: return 3
        return 4

    @staticmethod
    def date_to_yyyymmdd(dt: date) -> str:
        return dt.strftime("%Y%m%d")

    @staticmethod
    def get_fy_start_end(fy_year: int) -> Tuple[str, str]:
        return f"{fy_year}0401", f"{fy_year + 1}0331"

    @staticmethod
    def today_yyyymmdd() -> str:
        return datetime.today().strftime("%Y%m%d")

    @staticmethod
    def extract_date_tokens(text: str):
        print(text, '========================')
        return re.findall(
            r'\d{4}-\d{2}-\d{2}'
            r'|\d{1,2}[/-]\d{1,2}[/-]\d{4}'
            # 15 sep | 1st september | 11th mar 2024
            r'|\b\d{1,2}(?:st|nd|rd|th)?\s+'
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
            r'|january|february|march|april|june|july|august|september|october|november|december)'
            r'(?:,?\s*\d{4})?\b'
            # sep 15 | september 1st | mar 3rd, 2025
            r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
            r'|january|february|march|april|june|july|august|september|october|november|december)'
            r'\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b'
            # Standalone month names
            r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
            r'|january|february|march|april|june|july|august|september|october|november|december)\b',
            text,
            re.IGNORECASE
        )

    @staticmethod
    def parse_flexible_date(text: str, default_year=None):
        if not text:
            return None
        text = text.lower().strip()
        today = date.today()
        current_fy = DateParser.get_current_fy(today)

        def resolve_year(mth):
            if default_year:
                return default_year
            return current_fy if mth >= 4 else current_fy + 1

        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            pass
        try:
            return datetime.strptime(text.replace("-", "/"), "%d/%m/%Y").date()
        except ValueError:
            pass

        # m = re.search(r'(\d{1,2})\s+([a-z]{3,9})(?:\s+(\d{4}))?', text)
        m = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})(?:\s+(\d{4}))?', text)
        if m:
            try:
                d   = int(m.group(1))
                mth = DateParser.MONTH_MAP.get(m.group(2)) or DateParser.MONTH_MAP.get(m.group(2)[:3])
                y   = int(m.group(3)) if m.group(3) else resolve_year(mth)
                if mth:
                    return date(y, mth, d)
            except Exception:
                pass

        m = re.search(r'([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?', text)
        if m:
            try:
                mth = DateParser.MONTH_MAP.get(m.group(1)) or DateParser.MONTH_MAP.get(m.group(1)[:3])
                d   = int(m.group(2))
                y   = int(m.group(3)) if m.group(3) else resolve_year(mth)
                if mth:
                    return date(y, mth, d)
            except Exception:
                pass
        return None

    @staticmethod
    def parse_from_date(query: str, today: date = None) -> Optional[Dict[str, Any]]:
        if not today:
            today = date.today()
        q = query.lower()
        pattern = re.search(
            r'(from|after|since)\s+'
            r'(?:(\d{1,2})\s+)?'
            r'([a-z]{3,9}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            r'(?:\s+(\d{4}))?',
            q
        )
        if not pattern:
            return None

        keyword, day, month_part, year = pattern.groups()
        original_day = day
        current_fy   = DateParser.get_current_fy(today)

        if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', month_part):
            clean = month_part.replace("/", "-")
            fmt   = "%d-%m-%Y" if len(clean.split("-")[-1]) == 4 else "%d-%m-%y"
            dt    = datetime.strptime(clean, fmt).date()
        else:
            month = DateParser.MONTH_MAP.get(month_part[:3])
            if not month:
                return None
            if year:
                found_year = int(year)
            else:
                found_year = current_fy if month >= 4 else current_fy + 1
            day_int = int(day) if day else 1
            dt = date(found_year, month, day_int)

        if keyword == "after":
            is_numeric = re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', month_part)
            print(original_day,is_numeric,month_part)
            if original_day is None and not is_numeric:
                last_day = monthrange(dt.year, dt.month)[1]
                dt = date(dt.year, dt.month, last_day) + timedelta(days=1)
            else:
                dt = dt + timedelta(days=1)

        return {
            "type":       QueryType.DATE_RANGE,
            "start_date": DateParser.date_to_yyyymmdd(dt),
            "end_date":   None,
            "label":      f"From {dt.strftime('%d %b %Y')}",
        }
        
    @staticmethod
    def parse_fy_till_date(q: str):
        if not q or not isinstance(q, str):
            return None

        q = q.lower().strip()

        # Split on till/upto
        parts = re.split(r'\b(?:till|upto|up to)\b', q, maxsplit=1)
        if len(parts) != 2:
            return None

        left_part, right_part = parts[0].strip(), parts[1].strip()

        # 🚫 IMPORTANT: if left side already has a date → normal range
        if DateParser.extract_date_tokens(left_part):
            return None

        # Parse end date
        end_date = DateParser.parse_flexible_date(right_part)
        if not end_date:
            return None

        # Financial Year logic (Apr–Mar)
        fy_start_year = end_date.year if end_date.month >= 4 else end_date.year - 1
        start_date = date(fy_start_year, 4, 1)

        if start_date > end_date:
            return None

        return {
            "type": QueryType.DATE_RANGE,
            "start_date": DateParser.date_to_yyyymmdd(start_date),
            "end_date": DateParser.date_to_yyyymmdd(end_date),
            "label": f"FY {fy_start_year} till {end_date.strftime('%d %b %Y')}"
        }

    @staticmethod
    def parse_month_only(text, year, is_start=True):
        m = re.search(r'\b([a-z]{3,9})\b', text)
        if not m:
            return None
        mth_name = m.group(1)
        mth = DateParser.MONTH_MAP.get(mth_name) or DateParser.MONTH_MAP.get(mth_name[:3])
        if not mth:
            return None
        if is_start:
            return date(year, mth, 1)
        else:
            last_day = monthrange(year, mth)[1]
            return date(year, mth, last_day)

    @staticmethod
    def parse_specific_date_or_range(q: str):
        if not q or not isinstance(q, str):
            return None

        q = q.lower().strip()
        today = date.today()

        # 🚫 Block month-year only (e.g. "sep 2024")
        if re.search(r'\b([a-z]+)\s+\d{4}\b', q):
            if not re.search(r'\b([1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\b', q):
                return None
        
        # ==================================================
        # 0️⃣ FY till-date (NEW FUNCTION CALL)
        # ==================================================
        fy_till = DateParser.parse_fy_till_date(q)
        if fy_till:
            return fy_till

        print("checking date range function---------------")

        # ==================================================
        # 1️⃣ DATE RANGE (HIGHEST PRIORITY)
        # ==================================================
        # range_match = re.search(
        #     r'(.+?)\s+(and|to|until|till|through|–|-)\s+(.+)',
        #     q
        # )
        range_match = re.search(
            r'(.+?)\s+(to|until|till|through|–|-)\s+(.+)',
            q
        )

        between_match = re.search(
            r'(.+?)\s+between\s+(.+?)\s+and\s+(.+)',
            q,
            re.IGNORECASE
        )
        if range_match or between_match:
            raw_start = range_match.group(1).strip() if range_match else between_match.group(2).strip()
            raw_end = range_match.group(3).strip() if range_match else between_match.group(3).strip()

            start_tokens = DateParser.extract_date_tokens(raw_start)
            end_tokens = DateParser.extract_date_tokens(raw_end)

            print(start_tokens,end_tokens)
            start_text = start_tokens[0] if start_tokens else raw_start
            end_text = end_tokens[0] if end_tokens else raw_end
            print(start_text,end_text)
            # Try parse end first (for propagation)
            end_date = DateParser.parse_flexible_date(end_text)
            # 🔥 HANDLE end month-only
            if not end_date:
                end_date = DateParser.parse_month_only(end_text, today.year, is_start=False)

            if not end_date:
                return None

            # Try start
            start_date = DateParser.parse_flexible_date(start_text, default_year=end_date.year)
            # 🔥 HANDLE start month-only
            if not start_date:
                start_date = DateParser.parse_month_only(start_text, end_date.year, is_start=True)

            print(start_date,end_date,'====================')
            if start_date and start_date > end_date:
                start_date = DateParser.parse_flexible_date(start_text, default_year=end_date.year - 1) \
                      or DateParser.parse_month_only(start_text, end_date.year - 1, is_start=True) 

            # 🔥 PROPAGATE month/year if start is day-only
            if not start_date:
                day_match = re.search(r'\b([1-9]|[12][0-9]|3[01])\b', start_text)
                if day_match:
                    start_date = date(
                        end_date.year,
                        end_date.month,
                        int(day_match.group(1))
                    )

            if not start_date or start_date > end_date:
                return None

            return {
                "type": QueryType.DATE_RANGE,
                "start_date": DateParser.date_to_yyyymmdd(start_date),
                "end_date": DateParser.date_to_yyyymmdd(end_date),
                "label": f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
            }

        print("check spacific date range function")
        # ==================================================
        # 2️⃣ SPECIFIC DATE
        # ==================================================

        if any(k in q for k in ["after", "since"]):
            return None
        tokens = DateParser.extract_date_tokens(q)
        if not tokens:
            return None
        parsed_date = DateParser.parse_flexible_date(tokens[0])
        if not parsed_date:
            return None
        print(parsed_date,"parsed_date------------------")
        return {
            "type": QueryType.SPECIFIC_DATE,
            "date": DateParser.date_to_yyyymmdd(parsed_date),
            "label": parsed_date.strftime("%d %B %Y")
        }

    @staticmethod
    def parse_fy_till_month(query: str) -> Optional[Dict[str, Any]]:
        if not query or not isinstance(query, str):
            return None
        q = query.lower().strip()
        parts = re.split(r'\b(?:till|upto|up to)\b', q, maxsplit=1)
        if len(parts) != 2:
            return None
        left_part, right_part = parts[0].strip(), parts[1].strip()
        if DateParser.extract_date_tokens(left_part):
            return None
        if re.search(
            r'\b([1-9]|[12][0-9]|3[01])(st|nd|rd|th)?\s+(' +
            '|'.join(DateParser.MONTH_MAP.keys()) + r')\b',
            right_part, re.IGNORECASE
        ):
            return None
        month_match = re.search(
            r'\b(' + '|'.join(DateParser.MONTH_MAP.keys()) + r')\b',
            right_part, re.IGNORECASE
        )
        if not month_match:
            return None
        month_num  = DateParser.MONTH_MAP[month_match.group(1).lower()]
        year_match = re.search(r'\b(20\d{2})\b', right_part)
        today      = date.today()
        current_fy = DateParser.get_current_fy(today)
        year = int(year_match.group(1)) if year_match else (
            current_fy if month_num >= 4 else current_fy + 1
        )
        fy_start_year = year if month_num >= 4 else year - 1
        start_date = date(fy_start_year, 4, 1)
        last_day   = monthrange(year, month_num)[1]
        end_date   = date(year, month_num, last_day)
        if start_date > end_date:
            return None
        return {
            "type":       QueryType.DATE_RANGE,
            "start_date": DateParser.date_to_yyyymmdd(start_date),
            "end_date":   DateParser.date_to_yyyymmdd(end_date),
            "label":      f"FY {fy_start_year} till {end_date.strftime('%b %Y')}",
        }

    @staticmethod
    def parse_quarter_intent(q: str):
        q = q.lower()
        today      = datetime.today()
        current_fy = DateParser.get_current_fy(today)
        target_fy  = current_fy

        # ✅ Detect flags
        is_mom = bool(re.search(r"\b(mom|month\s*on\s*month|monthly|month wise)\b", q))
        is_qoq = bool(re.search(r"\b(qoq|quarter\s*on\s*quarter|quaterly|quarter wise)\b", q))
        is_yoy = bool(re.search(r"\b(yoy|year\s*on\s*year|yearly|year wise)\b", q))

        # ✅ Year override
        year_match = extract_fy(q)
        if year_match:
            target_fy = year_match

        target_fy = detect_fy(q,current_fy)

        # if re.search(r"\b(last|previous)\s+(year|fy)\b", q):
        #     target_fy = current_fy - 1

        # ✅ Quarter extraction
        q_pattern = r"(?:q(?:uarter|tr)?\s*([1-4]))"

        range_match = re.search(
            rf"\b{q_pattern}\s*(?:to|till|until|–|-)\s*{q_pattern}\b", q
        )

        quarters = []

        if range_match:
            start_q = int(range_match.group(1))
            end_q   = int(range_match.group(2))
            quarters = [f"q{i}" for i in range(start_q, end_q + 1)]
        else:
            found_qs = re.findall(rf"\b{q_pattern}\b", q)
            quarters = [f"q{num}" for num in found_qs]

        if not quarters:
            return None

        quarters = sorted(set(quarters))

        # ✅ Quarter → Date function
        def get_q_dates(fy, q_num):
            if q_num == 1:
                return datetime(fy,4,1), datetime(fy,6,30)
            elif q_num == 2:
                return datetime(fy,7,1), datetime(fy,9,30)
            elif q_num == 3:
                return datetime(fy,10,1), datetime(fy,12,31)
            else:
                return datetime(fy+1,1,1), datetime(fy+1,3,31)

        # =========================
        # ✅ YoY (VERY POWERFUL 🔥)
        # =========================
        if is_yoy:
            start_fy = 2020   # or dynamic
            end_fy   = current_fy

            periods = []

            for fy in range(start_fy, end_fy + 1):
                for q_txt in quarters:
                    q_num = int(q_txt[1])
                    s, e = get_q_dates(fy, q_num)

                    periods.append({
                        "label": f"Q{q_num} FY{fy}",
                        "start_date": DateParser.date_to_yyyymmdd(s.date()),
                        "end_date": DateParser.date_to_yyyymmdd(e.date())
                    })

            return {
                "type": QueryType.QUARTER_WISE,
                "quarters": periods,
                "label": "Quarter YoY"
            }

        # =========================
        # ✅ QoQ
        # =========================
        if is_qoq:
            quarters_out = []

            for q_txt in quarters:
                q_num = int(q_txt[1])
                s, e = get_q_dates(target_fy, q_num)

                quarters_out.append({
                    "quarter": f"Q{q_num} FY{target_fy}",
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date": DateParser.date_to_yyyymmdd(e.date())
                })

            return {
                "type": QueryType.QUARTER_WISE,
                "quarters": quarters_out,
                "label": "QoQ"
            }

        # =========================
        # ✅ MoM inside quarters 🔥
        # =========================
        if is_mom:
            periods = []

            for q_txt in quarters:
                q_num = int(q_txt[1])
                s, e = get_q_dates(target_fy, q_num)

                m, y = s.month, s.year
                while (y < e.year) or (y == e.year and m <= e.month):
                    _, ld = monthrange(y, m)

                    sm = datetime(y, m, 1)
                    em = datetime(y, m, ld)

                    periods.append({
                        "label": sm.strftime("%b %Y"),
                        "start_date": DateParser.date_to_yyyymmdd(sm.date()),
                        "end_date": DateParser.date_to_yyyymmdd(em.date())
                    })

                    m += 1
                    if m == 13:
                        m = 1
                        y += 1

            return {
                "type": QueryType.MONTH_WISE,
                "periods": periods,
                "label": "Quarter MoM"
            }

        # =========================
        # ✅ Default (range / multi)
        # =========================
        quarter_ranges = []

        for q_txt in quarters:
            q_num = int(q_txt[1])
            start, end = get_q_dates(target_fy, q_num)

            quarter_ranges.append({
                "start_date": DateParser.date_to_yyyymmdd(start.date()),
                "end_date": DateParser.date_to_yyyymmdd(end.date()),
                "label": f"Q{q_num} FY{target_fy}",
            })

        if len(quarter_ranges) == 1:
            return {"type": QueryType.DATE_RANGE, **quarter_ranges[0]}

        return {
            "type": QueryType.MULTI_DATE_RANGE,
            "ranges": quarter_ranges,
            "label": " + ".join(r["label"] for r in quarter_ranges),
        }


# ============================================================================
# OPPORTUNITY SCHEMA / COLUMN METADATA
# ============================================================================
class OppColumnMetadata:
    COLUMNS = {
        "lead_id_c":                  {"type": "VARCHAR", "description": "Opportunity unique ID — use COUNT(DISTINCT lead_id_c) for opportunity count"},
        "opportunity_id_c":           {"type": "VARCHAR", "description": "Salesforce opportunity ID"},
        "owner_name_c":               {"type": "VARCHAR", "description": "Name of the opportunity owner / sales person"},
        "created_by_c":               {"type": "VARCHAR", "description": "System/record creation metadata — NOT the owner"},
        "created_date_c":             {"type": "INTEGER", "description": "Opportunity creation date in YYYYMMDD integer format"},
        "last_modified_date_c":       {"type": "INTEGER", "description": "Last modified date in YYYYMMDD integer format"},
        "sales_order_date_c":         {"type": "INTEGER", "description": "Sales order date in YYYYMMDD integer format"},
        "lead_source_c":              {"type": "VARCHAR", "description": "High-level lead/opportunity source"},
        "lead_source_sub_category_c": {"type": "VARCHAR", "description": "Detailed lead sub-source channel"},
        "project_c":                  {"type": "VARCHAR", "description": "Project name (Wave City, WMCC, Wave Estate, etc.)"},
        "project_category_c":         {"type": "VARCHAR", "description": "Product/unit category (veridia, dream homes, eden, plots, etc.)"},
        "property_size_c":            {"type": "VARCHAR", "description": "Property size type (1BHK, 2BHK, 3BHK, plots, skyvillas, etc.)"},
        "property_type_c":            {"type": "VARCHAR", "description": "Property type (Residential, Commercial)"},
        "range_budget_c":             {"type": "VARCHAR", "description": "Customer budget range (free text e.g. 3cr-4cr, 50 Lacs)"},
        "sales_order_number_c":       {"type": "VARCHAR", "description": "Sales order number — NOT NULL means sale done"},
        "sap_customer_code_c":        {"type": "VARCHAR", "description": "SAP customer code"},
        "sales_open_reason_c":        {"type": "VARCHAR", "description": "Reason opportunity is still open"},
        "disqualification_reason_c":  {"type": "VARCHAR", "description": "Reason for disqualification"},
    }

    # Date columns stored as YYYYMMDD INTEGER
    DATE_COLUMNS = ["created_date_c", "last_modified_date_c", "sales_order_date_c"]

    # Primary count columns
    COUNT_COLUMN       = "lead_id_c"          # opportunity count
    SALES_COUNT_COLUMN = "sales_order_number_c"  # sale done count

    # Dimension columns eligible for GROUP BY
    DIMENSION_COLUMNS = [
        "lead_source_c", "lead_source_sub_category_c", "project_c",
        "project_category_c", "property_size_c", "property_type_c",
         "owner_name_c", "sap_customer_code_c",
        "sales_open_reason_c", "disqualification_reason_c",
        "range_budget_c",
    ]

    # ── Known valid values for normalisation ──────────────────────────────────
    VALID_VALUES = {
        "lead_source_c": [
            "Bulk Sale", "Channel Partner", "Digital", "Direct Walkin",
            "Electronic Media", "Events / Exhibitions", "Existing Customer",
            "Lead Reassigned", "Outbound Campaign", "Outdoor", "Print Media",
            "Reference Sale", "Referral", "Referral Sale", "SMS Campaign",
            "Transfered", "Unit Shifting", "Word of Mouth",
        ],
        "lead_source_sub_category_c": [
            "99 Acres", "Adgebra", "Bulk Sale", "Channel Partner", "ChatGPT.com",
            "Cinema Slides", "Delhi Times", "Digital Posters", "Direct",
            "Direct Email", "Direct Walkin", "Existing Customer",
            "External Referral", "Facebook", "Google", "Google_OD",
            "Housing.com", "HT.com", "Inshorts", "Instagram",
            "Internal Referral", "Internal-Video-Eden",
            "Lead Reassigned - CDL", "Lead Reassigned - SDL",
            "Live Chat", "Magic Bricks", "MagicBricks", "Mygate",
            "NoBroker", "Organic", "Outbound Campaign", "Outdoor",
            "Quora", "RCS", "Reference Sale", "Referral Sale",
            "SFMC", "Social Media-FB Page", "Spotify", "Taboola",
            "TOI.com", "Transfered", "Unit Shift", "Word of Mouth", "YouTube",
        ],
        "property_type_c": ["Residential", "Commercial","plots"],
        "property_size_c": [
            "1BHK", "2BHK", "3BHK", "4BHK", "5BHK",
            "Penthouse", "Plots", "Skyvillas",
            "Commercial Office Space", "Commercial Space",
        ],
    }

    # ── Known product/category values ─────────────────────────────────────────
    PRODUCT_CATEGORIES = [
        "veridia", "dream homes", "eligo", "wave floor", "old plots",
        "executive floors", "plots-res", "wave garden", "eden", "new plots",
        "wave galleria", "wrc old plot", "swamanorath", "amore", "livork",
        "wave floor 99", "ews_p2", "ews", "prime floors", "wrc plots",
        "mayfair park", "silver", "wave floor 85", "ews_001_(410)", "hssc",
        "metro mart", "lig_001_(310)", "wrc floors", "wbt 1",
        "wave garden gh2-ph-2", "lig", "lig_p2", "armonia villa", "trucia",
        "gold", "elegantia", "plots-res-if", "irenia", "harmony greens",
        "veridia-4", "edenia", "vasilia", "plots-comm", "dream bazaar",
        "veridia-5", "sco", "retail", "wave residency", "veridia-3",
        "wbt a", "eminence", "comm booth", "veridia-7", "courtyard",
        "wave business square", "institutional", "veridia tower 7",
        "wrc fsi", "fsi", "hubb", "group housing 1", "villas",
        "plot-res-if", "veridia-6", "villa", "commercial plots",
        "aranyam valley", "wrc institutional", "institutional_we",
        "dream homes_we", "golf range", "waved garden",
    ]

    # ── Known project values ───────────────────────────────────────────────────
    PROJECT_VALUES = [
        "wave city", "wmcc", "wmcc sec 32", "wmcc sector 32",
        "wave estate", "wave amore", "wave executive floors",
        "wave city phase 1",
    ]

    # ── KEYWORD MAPPING  ──────────────────────────────────────────────────────
    # Maps user keyword → {column, value}   (filter)
    #                   → {column}          (group-by only)
    #                   → {aggregation, column}  (metric)
    KEYWORD_MAPPING = {
        # ── Sale done ────────────────────────────────────────────────────────
        "sale done":        {"column": "sales_order_number_c", "value": "__sale_done__"},
        "sales done":       {"column": "sales_order_number_c", "value": "__sale_done__"},
        "booking done":     {"column": "sales_order_number_c", "value": "__sale_done__"},
        "booked":           {"column": "sales_order_number_c", "value": "__sale_done__"},

        # ── Property type ─────────────────────────────────────────────────────
        "residential":      {"column": "property_type_c", "value": "Residential"},
        "commercial":       {"column": "property_type_c", "value": "Commercial"},

        # ── Property size ─────────────────────────────────────────────────────
        "1bhk":             {"column": "property_size_c", "value": "1BHK"},
        "1 bhk":            {"column": "property_size_c", "value": "1BHK"},
        "2bhk":             {"column": "property_size_c", "value": "2BHK"},
        "2 bhk":            {"column": "property_size_c", "value": "2BHK"},
        "3bhk":             {"column": "property_size_c", "value": "3BHK"},
        "3 bhk":            {"column": "property_size_c", "value": "3BHK"},
        "4bhk":             {"column": "property_size_c", "value": "4BHK"},
        "4 bhk":            {"column": "property_size_c", "value": "4BHK"},
        "5bhk":             {"column": "property_size_c", "value": "5BHK"},
        "5 bhk":            {"column": "property_size_c", "value": "5BHK"},
        "penthouse":        {"column": "property_size_c", "value": "Penthouse"},
        "skyvillas":        {"column": "property_size_c", "value": "Skyvillas"},
        "sky villa":        {"column": "property_size_c", "value": "Skyvillas"},
        "plots":            {"column": "property_size_c", "value": "Plots"},

        # ── Lead source (main) ────────────────────────────────────────────────
        "digital":            {"column": "lead_source_c", "value": "Digital"},
        "direct walkin":      {"column": "lead_source_c", "value": "Direct Walkin"},
        "channel partner":    {"column": "lead_source_c", "value": "Channel Partner"},
        "bulk sale":          {"column": "lead_source_c", "value": "Bulk Sale"},
        "outbound campaign":  {"column": "lead_source_c", "value": "Outbound Campaign"},
        "outdoor":            {"column": "lead_source_c", "value": "Outdoor"},
        "print media":        {"column": "lead_source_c", "value": "Print Media"},
        "electronic media":   {"column": "lead_source_c", "value": "Electronic Media"},
        "existing customer":  {"column": "lead_source_c", "value": "Existing Customer"},
        "lead reassigned":    {"column": "lead_source_c", "value": "Lead Reassigned"},
        "reference sale":     {"column": "lead_source_c", "value": "Reference Sale"},
        "referral sale":      {"column": "lead_source_c", "value": "Referral Sale"},
        "word of mouth":      {"column": "lead_source_c", "value": "Word of Mouth"},
        "unit shifting":      {"column": "lead_source_c", "value": "Unit Shifting"},
        "transfered":         {"column": "lead_source_c", "value": "Transfered"},
        "sms campaign":       {"column": "lead_source_c", "value": "SMS Campaign"},
        "events":             {"column": "lead_source_c", "value": "Events / Exhibitions"},
        "exhibitions":        {"column": "lead_source_c", "value": "Events / Exhibitions"},

        # ── Lead sub-source ───────────────────────────────────────────────────
        "99 acres":           {"column": "lead_source_sub_category_c", "value": "99 Acres"},
        "magicbricks":        {"column": "lead_source_sub_category_c", "value": "MagicBricks"},
        "magic bricks":       {"column": "lead_source_sub_category_c", "value": "Magic Bricks"},
        "housing.com":        {"column": "lead_source_sub_category_c", "value": "Housing.com"},
        "google":             {"column": "lead_source_sub_category_c", "value": "Google"},
        "facebook":           {"column": "lead_source_sub_category_c", "value": "Facebook"},
        "instagram":          {"column": "lead_source_sub_category_c", "value": "Instagram"},
        "youtube":            {"column": "lead_source_sub_category_c", "value": "YouTube"},
        "nobroker":           {"column": "lead_source_sub_category_c", "value": "NoBroker"},
        "no broker":          {"column": "lead_source_sub_category_c", "value": "NoBroker"},
        "inshorts":           {"column": "lead_source_sub_category_c", "value": "Inshorts"},
        "quora":              {"column": "lead_source_sub_category_c", "value": "Quora"},
        "spotify":            {"column": "lead_source_sub_category_c", "value": "Spotify"},
        "taboola":            {"column": "lead_source_sub_category_c", "value": "Taboola"},
        "mygate":             {"column": "lead_source_sub_category_c", "value": "Mygate"},
        "organic":            {"column": "lead_source_sub_category_c", "value": "Organic"},
        "sfmc":               {"column": "lead_source_sub_category_c", "value": "SFMC"},
        "adgebra":            {"column": "lead_source_sub_category_c", "value": "Adgebra"},
        "rcs":                {"column": "lead_source_sub_category_c", "value": "RCS"},
        "direct":              {"column": "lead_source_sub_category_c", "value": "Direct"},

        # ── Project keywords ──────────────────────────────────────────────────
        "wave city":          {"column": "project_c", "value": "wave city"},
        "wmcc":               {"column": "project_c", "value": "wmcc"},
        "wmcc sec 32":        {"column": "project_c", "value": "wmcc sec 32"},
        "wmcc sector 32":     {"column": "project_c", "value": "wmcc sec 32"},
        "wave estate":        {"column": "project_c", "value": "wave estate"},
        "wave amore":         {"column": "project_c", "value": "wave amore"},
        "wave executive floors": {"column": "project_c", "value": "wave executive floors"},
        "wave city phase 1":  {"column": "project_c", "value": "wave city phase 1"},

        # ── Grouping-only keywords ────────────────────────────────────────────
        "by source":              {"column": "lead_source_c"},
        "source wise":            {"column": "lead_source_c"},
        "lead source wise":       {"column": "lead_source_c"},
        "by sub source":          {"column": "lead_source_sub_category_c"},
        "sub source wise":        {"column": "lead_source_sub_category_c"},
        "by project":             {"column": "project_c"},
        "project wise":           {"column": "project_c"},
        "project bifurcation":    {"column": "project_c"},
        "project breakdown":      {"column": "project_c"},
        "project trend":          {"column": "project_c"},
        "by product":             {"column": "project_category_c"},
        "product wise":           {"column": "project_category_c"},
        "category wise":          {"column": "project_category_c"},
        "product bifurcation":    {"column": "project_category_c"},
        "product breakdown":      {"column": "project_category_c"},
        "product trend":          {"column": "project_category_c"},
        "by owner":               {"column": "owner_name_c"},
        "owner wise":             {"column": "owner_name_c"},
        "by property type":       {"column": "property_type_c"},
        "property type wise":     {"column": "property_type_c"},
        "by property size":       {"column": "property_size_c"},
        "property size wise":     {"column": "property_size_c"},
        "budget wise":            {"column": "range_budget_c"},
        "by budget":              {"column": "range_budget_c"},
        "customer code wise":     {"column": "sap_customer_code_c"},
        "by customer code":       {"column": "sap_customer_code_c"},
        "disqualification wise":  {"column": "disqualification_reason_c"},
        "by disqualification":    {"column": "disqualification_reason_c"},
        "open reason wise":       {"column": "sales_open_reason_c"},

        # ── Aggregation keywords ──────────────────────────────────────────────
        "total sales":         {"aggregation": "count", "column": "sales_order_number_c"},
        "sales count":         {"aggregation": "count", "column": "sales_order_number_c"},
        "number of sales":     {"aggregation": "count", "column": "sales_order_number_c"},
        "sales done":          {"aggregation": "count", "column": "sales_order_number_c"},
        "total opportunities": {"aggregation": "count", "column": "lead_id_c"},
        "opportunity count":   {"aggregation": "count", "column": "lead_id_c"},
        "total opportunities": {"aggregation": "count", "column": "lead_id_c"},
        "opportunities":       {"aggregation": "count", "column": "lead_id_c"},
        "total bookings":      {"aggregation": "count", "column": "sales_order_number_c"},
        "bookings":            {"aggregation": "count", "column": "sales_order_number_c"},
    }


# ============================================================================
# MULTI-PERIOD HELPER FUNCTIONS  (identical cascade to lead engine)
# ============================================================================
def mom_logic(q: str):
    today      = datetime.today()
    current_fy = DateParser.get_current_fy()

    # target_fy = current_fy
    # if "last year" in q or "previous year" in q:
    #     target_fy = current_fy - 1
    # else:
    #     for word in q.split():
    #         if word.isdigit() and len(word) == 4:
    #             target_fy = int(word); break

    target_fy = detect_fy(q,current_fy)

    periods = []
    for month in range(4, 16):
        m  = month if month <= 12 else month - 12
        y  = target_fy if month <= 12 else target_fy + 1
        _, ld = monthrange(y, m)
        s  = datetime(y, m, 1)
        e  = datetime(y, m, ld)
        periods.append({
            "label":      s.strftime("%b %Y"),
            "start_date": DateParser.date_to_yyyymmdd(s.date()),
            "end_date":   DateParser.date_to_yyyymmdd(e.date()),
        })
    return {
        "type":    QueryType.MONTH_WISE,
        "fy":      target_fy,
        "periods": periods,
        "label":   f"Month-wise FY{target_fy}",
    }


def last_n_mom_logic(q: str):
    last_n_month_match = re.search(
        r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b", q
    )
    is_mom = any(k in q for k in [
        "mom", "month over month", "month-on-month",
        "monthly", "month wise", "month on month", "months wise"
    ])
    today = datetime.today()
    try:
        raw_n = last_n_month_match.group(1)
        n     = int(raw_n) if raw_n.isdigit() else DateParser.WORD_TO_NUM.get(raw_n, 1)
        first_of_month = today.date().replace(day=1)
        end_dt   = first_of_month - timedelta(days=1)
        start_dt = end_dt.replace(day=1)
        months   = []
        for _ in range(n):
            months.append(start_dt)
            start_dt = (start_dt - timedelta(days=1)).replace(day=1)
        months   = list(reversed(months))
        start_dt = months[0]
        if is_mom:
            periods = []
            for ms in months:
                yy, mm = ms.year, ms.month
                _, ld  = monthrange(yy, mm)
                s = datetime(yy, mm, 1)
                e = datetime(yy, mm, ld)
                periods.append({
                    "label":      s.strftime("%b %Y"),
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date":   DateParser.date_to_yyyymmdd(e.date()),
                })
            return {"type": QueryType.MONTH_WISE, "periods": periods,
                    "label": f"Last {n} Months (MoM)"}
        return {
            "type":       QueryType.LAST_N_MONTHS,
            "start_date": DateParser.date_to_yyyymmdd(start_dt),
            "end_date":   DateParser.date_to_yyyymmdd(end_dt),
            "label":      f"Last {n} Months",
        }
    except Exception:
        pass


def last_n_quarter_mom_qoq(q: str):
    last_n_quarter_match = re.search(
        r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+quarters?\b", q
    )
    is_mom = any(k in q for k in [
        "mom", "month over month", "month-on-month",
        "monthly", "month wise", "month on month", "months wise"
    ])
    is_qoq = any(k in q for k in [
        "qoq", "quarter over quarter", "quarter-on-quarter",
        "quarterly", "quarter wise", "quarter on quarter"
    ])
    today      = datetime.today()
    current_fy = DateParser.get_current_fy()
    try:
        raw_n = last_n_quarter_match.group(1)
        n = max(1, int(raw_n) if raw_n.isdigit() else DateParser.WORD_TO_NUM.get(raw_n, 1))

        year_match    = extract_fy(q)
        explicit_year = int(year_match) if year_match else None
        base_fy       = explicit_year if explicit_year else current_fy
        cq            = DateParser.get_fy_quarter(today.month)
        eq            = cq - 1 if cq > 1 else 4
        efy           = base_fy if cq > 1 else base_fy - 1

        meta = []
        qu, fy2 = eq, efy
        for _ in range(n):
            meta.append((fy2, qu))
            qu -= 1
            if qu == 0:
                qu = 4; fy2 -= 1
        meta.reverse()

        def qd(fy3, qn):
            if qn == 1: return datetime(fy3, 4, 1),    datetime(fy3, 6, 30)
            elif qn == 2: return datetime(fy3, 7, 1),  datetime(fy3, 9, 30)
            elif qn == 3: return datetime(fy3, 10, 1), datetime(fy3, 12, 31)
            else:         return datetime(fy3+1, 1, 1), datetime(fy3+1, 3, 31)

        if is_qoq:
            quarters = []
            for fy3, qn in meta:
                s2, e2 = qd(fy3, qn)
                quarters.append({
                    "quarter":    f"Q{qn} FY{fy3}",
                    "start_date": DateParser.date_to_yyyymmdd(s2.date()),
                    "end_date":   DateParser.date_to_yyyymmdd(e2.date()),
                })
            return {"type": QueryType.QUARTER_WISE, "quarters": quarters,
                    "label": f"Last {n} Quarters (QoQ)"}

        if is_mom:
            periods = []
            for fy3, qn in meta:
                s2, e2 = qd(fy3, qn)
                m_iter, y_iter = s2.month, s2.year
                while (y_iter < e2.year) or (y_iter == e2.year and m_iter <= e2.month):
                    _, ld = monthrange(y_iter, m_iter)
                    s = datetime(y_iter, m_iter, 1)
                    e = datetime(y_iter, m_iter, ld)
                    periods.append({
                        "label":      s.strftime("%b %Y"),
                        "start_date": DateParser.date_to_yyyymmdd(s.date()),
                        "end_date":   DateParser.date_to_yyyymmdd(e.date()),
                    })
                    m_iter += 1
                    if m_iter == 13:
                        m_iter = 1; y_iter += 1
            return {"type": QueryType.MONTH_WISE, "periods": periods,
                    "label": f"Last {n} Quarters (MoM)"}

        s2, _  = qd(*meta[0])
        _, e2  = qd(*meta[-1])
        return {
            "type":       QueryType.LAST_N_QUARTERS,
            "start_date": DateParser.date_to_yyyymmdd(s2.date()),
            "end_date":   DateParser.date_to_yyyymmdd(e2.date()),
            "label":      f"Last {n} Quarters",
        }
    except Exception as ex:
        logger.error(f"LAST_N_QUARTERS error: {ex}")


def yoy_logic(q: str):
    
    start_fy = 2018  # your DB start year
    current_fy = DateParser.get_current_fy()
    end_fy   = current_fy

    years = list(range(start_fy, end_fy + 1))

    periods = []
    for fy in years:
        s, e = DateParser.get_fy_start_end(fy)
        periods.append({
            "year": f"FY{fy}",
            "start_date": s,
            "end_date": e
        })

    return {
        "type": QueryType.YEAR_WISE,
        "years": years,
        "periods": periods,
        "label": f"Year-on-Year (FY{start_fy}–FY{end_fy})"
    }


def last_n_year_mom_qoq_yoy(q: str):
    last_n_year_match = re.search(
        r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\b", q
    )
    is_mom = any(k in q for k in [
        "mom", "month over month", "month-on-month",
        "monthly", "month wise", "month on month", "months wise"
    ])
    is_qoq = any(k in q for k in [
        "qoq", "quarter over quarter", "quarter-on-quarter",
        "quarterly", "quarter wise", "quarter on quarter"
    ])
    is_yoy = any(k in q for k in [
        "yoy", "year on year", "yearly", "year wise",
        "by year", "annual trend", "year over year"
    ])
    current_fy = DateParser.get_current_fy()
    raw_n      = last_n_year_match.group(1)
    n          = int(raw_n) if raw_n.isdigit() else DateParser.WORD_TO_NUM.get(raw_n, 1)
    end_fy     = current_fy - 1
    start_fy   = end_fy - n + 1

    if is_yoy:
        years   = list(range(start_fy, end_fy + 1))
        periods = []
        for fy in years:
            s, e = DateParser.get_fy_start_end(fy)
            periods.append({"year": f"FY{fy}", "start_date": s, "end_date": e})
        return {"type": QueryType.YEAR_WISE, "years": years, "periods": periods,
                "label": f"Last {n} Years (YoY)"}

    if is_qoq:
        quarters = []
        for fy in range(start_fy, end_fy + 1):
            for qn in range(1, 5):
                if qn == 1:   s, e = datetime(fy, 4, 1),    datetime(fy, 6, 30)
                elif qn == 2: s, e = datetime(fy, 7, 1),    datetime(fy, 9, 30)
                elif qn == 3: s, e = datetime(fy, 10, 1),   datetime(fy, 12, 31)
                else:         s, e = datetime(fy+1, 1, 1),  datetime(fy+1, 3, 31)
                quarters.append({
                    "quarter":    f"Q{qn} FY{fy}",
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date":   DateParser.date_to_yyyymmdd(e.date()),
                })
        return {"type": QueryType.QUARTER_WISE, "quarters": quarters,
                "label": f"Last {n} Years (QoQ)"}

    if is_mom:
        periods = []
        for fy in range(start_fy, end_fy + 1):
            for month in range(4, 16):
                m = month if month <= 12 else month - 12
                y = fy if month <= 12 else fy + 1
                _, ld = monthrange(y, m)
                s = datetime(y, m, 1)
                e = datetime(y, m, ld)
                periods.append({
                    "label":      s.strftime("%b %Y"),
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date":   DateParser.date_to_yyyymmdd(e.date()),
                })
        return {"type": QueryType.MONTH_WISE, "periods": periods,
                "label": f"Last {n} Years (MoM)"}

    s, _ = DateParser.get_fy_start_end(start_fy)
    _, e = DateParser.get_fy_start_end(end_fy)
    return {"type": QueryType.LAST_N_YEARS, "start_date": s, "end_date": e,
            "label": f"Last {n} Years"}


def year_range_logic(q: str):
    if re.search(r'\b\d{1,2}\s+[a-z]{3,9}\s+20\d{2}\b', q):
        return None
    # Do NOT match if a month name immediately precedes either year token
    # e.g. "apr 2023 to may 2025" should be handled by parse_month_range_logic
    month_names = '|'.join(DateParser.MONTH_MAP.keys())
    if re.search(
        rf'\b(?:{month_names})[a-z]*\s+20\d{{2}}\s*(?:to|till|and|–|-)\s*(?:{month_names})[a-z]*\s+20\d{{2}}\b',
        q, re.IGNORECASE
    ):
        return None
    year_range_match = re.search(r'\b(20\d{2})\s*(?:to|till|–|-)\s*(20\d{2})\b', q)
    if year_range_match:
        start_year = int(year_range_match.group(1))
        end_year   = int(year_range_match.group(2))
        periods    = []
        for fy in range(start_year, end_year + 1):
            s, e = DateParser.get_fy_start_end(fy)
            periods.append({"year": f"FY{fy}", "start_date": s, "end_date": e})
        return {
            "type":    QueryType.YEAR_WISE,
            "years":   list(range(start_year, end_year + 1)),
            "periods": periods,
            "label":   f"FY{start_year} to FY{end_year}",
        }
    return None


def detect_year_and(q: str):
    matches = re.findall(r'\b(20\d{2})\b', q)
    # Do NOT match if month names surround the years (handled by parse_month_range_logic)
    month_names = '|'.join(DateParser.MONTH_MAP.keys())
    if re.search(
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+[a-z]{3,9}\s+20\d{2}\b',
            q
        ):
        return None
    if re.search(
        rf'\b(?:{month_names})[a-z]*\s+20\d{{2}}\s*(?:and|to|till|–|-)\s*(?:{month_names})[a-z]*\s+20\d{{2}}\b',
        q, re.IGNORECASE
    ):
        return None
    if len(matches) >= 2 and re.search(r'\band\b', q):
        years   = sorted(set(int(y) for y in matches))
        periods = []
        for fy in years:
            s, e = DateParser.get_fy_start_end(fy)
            periods.append({"year": f"FY{fy}", "start_date": s, "end_date": e})
        return {
            "type":    QueryType.YEAR_WISE,
            "years":   years,
            "periods": periods,
            "label":   " & ".join([f"FY{y}" for y in years]),
        }
    return None


# def parse_month_range_logic(q: str):
    # # Match patterns like:
    # #   "apr 2023 to may 2025"   (month year TO month year)
    # #   "from apr 2023 to may 2025"
    # #   "between jan 2024 and dec 2025"
    # #   "apr to may"  (no year)
    # #   "apr to may 2025"  (end year only)
    # month_year_range_match = re.search(
    #     r"(?:from\s+|between\s+)?"
    #     r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
    #     r"(?:\s+(20\d{2}))?"
    #     r"\s*(?:to|till|-|–|and)\s*"
    #     r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
    #     r"(?:\s+(20\d{2}))?",
    #     q
    # )
    # if not month_year_range_match:
    #     return None

    # # Fall back to the simpler match for use in label
    # month_range_match = month_year_range_match

    # is_mom = any(k in q for k in [
    #     "mom", "month over month", "month-on-month",
    #     "monthly", "month wise", "month on month", "months wise"
    # ])
    # today      = datetime.today()
    # current_fy = DateParser.get_current_fy()

    # m1_name = month_year_range_match.group(1)[:3]
    # y1_str  = month_year_range_match.group(2)   # may be None
    # m2_name = month_year_range_match.group(3)[:3]
    # y2_str  = month_year_range_match.group(4)   # may be None

    # m1_num  = DateParser.MONTH_MAP.get(m1_name)
    # m2_num  = DateParser.MONTH_MAP.get(m2_name)
    # if not m1_num or not m2_num:
    #     return None

    # def resolve_year(mth):
    #     return current_fy if mth >= 4 else current_fy + 1

    # # Determine years independently for start and end
    # if y1_str and y2_str:
    #     # Both years explicit: "apr 2023 to may 2025"
    #     y1 = int(y1_str)
    #     y2 = int(y2_str)
    # elif y2_str and not y1_str:
    #     # Only end year given: "apr to may 2025"
    #     y2 = int(y2_str)
    #     # Infer start year: if m1 <= m2 assume same year, else y2-1
    #     y1 = y2 if m1_num <= m2_num else y2 - 1
    # elif y1_str and not y2_str:
    #     # Only start year given: "apr 2024 to may"
    #     y1 = int(y1_str)
    #     y2 = y1 if m2_num >= m1_num else y1 + 1
    # else:
    #     # No year at all: use FY logic
    #     y1 = resolve_year(m1_num)
    #     y2 = resolve_year(m2_num)
    #     if m2_num < m1_num:
    #         y2 = y1 + 1 if m2_num < 4 else y1

    # _, ld1 = monthrange(y1, m1_num)
    # _, ld2 = monthrange(y2, m2_num)
    # start_str = DateParser.date_to_yyyymmdd(date(y1, m1_num, 1))
    # end_str   = DateParser.date_to_yyyymmdd(date(y2, m2_num, ld2))

    # if is_mom:
    #     periods = []
    #     m_iter, y_iter = m1_num, y1
    #     while (y_iter < y2) or (y_iter == y2 and m_iter <= m2_num):
    #         _, ld = monthrange(y_iter, m_iter)
    #         ms = datetime(y_iter, m_iter, 1)
    #         me = datetime(y_iter, m_iter, ld)
    #         periods.append({
    #             "label":      ms.strftime("%b %Y"),
    #             "start_date": DateParser.date_to_yyyymmdd(ms.date()),
    #             "end_date":   DateParser.date_to_yyyymmdd(me.date()),
    #         })
    #         m_iter += 1
    #         if m_iter == 13:
    #             m_iter = 1; y_iter += 1
    #     return {"type": QueryType.MONTH_RANGE_MONTH_WISE, "periods": periods,
    #             "label": f"{month_range_match.group(1).title()} to {month_range_match.group(2).title()} (MoM)"}
 
    # return {"type": QueryType.MONTH_RANGE, "start_date": start_str, "end_date": end_str,
    #         "label": f"{month_range_match.group(1).title()} to {month_range_match.group(2).title()}"}
def parse_month_range_logic(q: str):
    if not q:
        return None

    q = q.lower().strip()
    today = datetime.today()
    current_fy = today.year if today.month >= 4 else today.year - 1

    month_map = DateParser.MONTH_MAP
    month_names = sorted(month_map.keys(), key=len, reverse=True)
    month_alternation = "|".join(re.escape(name) for name in month_names)
    month_token_pattern = rf"\b(?:{month_alternation})\b"

    list_pattern = re.compile(
        rf"\b(?:{month_alternation})\b(?:\s*[,and]+\s*\b(?:{month_alternation})\b)+",
        re.IGNORECASE,
    )
    if list_pattern.search(q):
        print(f"Blocked multi-month list: {q}")
        return None

    month_range_match = re.search(
        rf"(?:from|between)?\s*(?P<start>{month_token_pattern})(?:\s+(?P<start_year>20\d{{2}}))?"
        rf"\s*(?:to|-|till|until|and)\s*(?P<end>{month_token_pattern})(?:\s+(?P<end_year>20\d{{2}}))?",
        q,
        re.IGNORECASE,
    )
    if not month_range_match:
        return None

    start_m_txt = month_range_match.group("start").lower()
    end_m_txt = month_range_match.group("end").lower()
    start_month = month_map[start_m_txt]
    end_month = month_map[end_m_txt]

    start_year_str = month_range_match.group("start_year")
    end_year_str = month_range_match.group("end_year")
    # is_last_year = bool(re.search(r"\b(last year|previous year)\b", q))
    is_last_year = bool(re.search(r"\b(last|previous|prev)\s*(fy|fiscal\s*year|year)?\b", q))

    if start_year_str or end_year_str:
        start_year = int(start_year_str) if start_year_str else None
        end_year = int(end_year_str) if end_year_str else None

        if start_year and not end_year:
            end_year = start_year
            if start_month > end_month:
                end_year += 1
        elif end_year and not start_year:
            start_year = end_year
            if start_month > end_month:
                start_year -= 1
    else:
        base_fy = current_fy - 1 if is_last_year else current_fy
        start_year = base_fy
        end_year = base_fy

        if start_month >= 4:
            end_year = base_fy if end_month >= 4 else base_fy + 1
        else:
            start_year = base_fy + 1
            end_year = base_fy + 1 if end_month < 4 else base_fy

    if start_year is None or end_year is None:
        return None

    start_date = datetime(start_year, start_month, 1)
    _, last_day = monthrange(end_year, end_month)
    end_date = datetime(end_year, end_month, last_day)

    if re.search(r"\b(month\s*wise|monthly|mom|month\s*on\s*month|by\s*month)\b", q):
        periods = []
        m, y = start_month, start_year
        while (y < end_year) or (y == end_year and m <= end_month):
            _, ld = monthrange(y, m)
            s = datetime(y, m, 1)
            e = datetime(y, m, ld)
            periods.append({
                "label": s.strftime("%b %Y"),
                "start_date": s.strftime("%Y%m%d"),
                "end_date": e.strftime("%Y%m%d"),
            })
            m += 1
            if m == 13:
                m = 1
                y += 1

        return {
            "type": QueryType.MONTH_RANGE_MONTH_WISE,
            "periods": periods,
            "label": f"{start_m_txt.title()}-{end_m_txt.title()} Month-wise",
        }

    return {
        "type": QueryType.MONTH_RANGE,
        "start_date": start_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
        "label": f"{start_m_txt.title()} to {end_m_txt.title()}",
    }

def extract_fy(q: str):
    q = q.lower()

    # fy2023 or fy 2023
    m = re.search(r"\bfy\s*(20\d{2})\b", q)
    if m:
        return int(m.group(1))

    # fy23 or fy 24
    m = re.search(r"\bfy\s*(\d{2})\b", q)
    if m:
        return 2000 + int(m.group(1))

    # plain year
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        return int(m.group(1))

    return None


def discrete_month(q: str):

    if re.search(
        r"\b([1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\s+"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
        r"|january|february|march|april|june|july|august|september|october|november|december)\b",
        q
    ):
        return None
    month_pattern = r'\b(' + '|'.join(DateParser.MONTH_MAP.keys()) + r')\b'

    found_months  = re.findall(month_pattern, q, re.IGNORECASE)
    if len(found_months) < 2:
        return None
    if re.search(r'\bto\b|\btill\b|–|-', q):
        return None

    today      = datetime.today()
    current_fy = DateParser.get_current_fy()
    year_match = extract_fy(q)
    target_year = int(year_match) if year_match else None

    periods = []
    seen    = set()
    for mn in found_months:
        mnum = DateParser.MONTH_MAP[mn.lower()]
        if mnum in seen:
            continue
        seen.add(mnum)
        y = target_year if target_year else (current_fy if mnum >= 4 else current_fy + 1)
        _, ld = monthrange(y, mnum)
        ms = datetime(y, mnum, 1)
        me = datetime(y, mnum, ld)
        periods.append({
            "label":      ms.strftime("%b %Y"),
            "start_date": DateParser.date_to_yyyymmdd(ms.date()),
            "end_date":   DateParser.date_to_yyyymmdd(me.date()),
        })

    if len(periods) < 2:
        return None
    return {"type": QueryType.MULTI_MONTH, "periods": periods,
            "label": ", ".join(p["label"] for p in periods)}

def parse_quarter_mom(q: str, today: datetime = None):
    if today is None:
        today = datetime.today()
    q = q.lower()
    if not any(k in q for k in ["mom", "month on month", "month wise", "month"]):
        return None

    parts = re.split(r'\b(?:to|till|upto|up to)\b', q, maxsplit=1)
    if len(parts) == 2:
        return None
    
    fy = DateParser.get_current_fy(today)
    current_q = DateParser.get_fy_quarter(today.month)

    quarters = []
    if "last quarter" in q or "last qtr" in q:
        if current_q == 1:
            q_num = 4; year = fy - 1
        else:
            q_num = current_q - 1; year = fy
        quarters = [f"q{q_num}"]
    elif "this quarter" in q or "current quarter" in q or "current qtr" in q or "this qtr" in q:
        q_num = current_q
        year = fy
        quarters = [f"q{q_num}"]
    else:
        q_pattern = r"(?:q(?:uarter|tr)?\s*([1-4]))"
        found_qs  = re.findall(rf"\b{q_pattern}\b", q)
        quarters  = [f"q{num}" for num in found_qs]
        if not quarters:
            return None
        year_match = extract_fy(q)
        year = int(year_match) if year_match else fy
        if any(k in q for k in ["last", "previous"]) and not year_match:
            year -= 1


    quarter_to_months = {1: [4,5,6], 2: [7,8,9], 3: [10,11,12], 4: [1,2,3]}
    periods = []
    for qtr in quarters:
        q_num  = int(qtr[1])
        months = quarter_to_months[q_num]
        for mm in months:
            y = year + 1 if q_num == 4 and mm in (1,2,3) else year
            _, ld = monthrange(y, mm)
            s = datetime(y, mm, 1)
            e = datetime(y, mm, ld)
            periods.append({
                "label":      s.strftime("%b %Y"),
                "start_date": s.strftime("%Y%m%d"),
                "end_date":   e.strftime("%Y%m%d"),
            })
    return {
        "type":    QueryType.MONTH_RANGE_MONTH_WISE,
        "periods": periods,
        "label":   f"Q{quarters} {year} MoM",
    }


# ============================================================================
# LLM INTENT PROMPT
# ============================================================================
def build_opp_llm_prompt(question: str) -> str:
    schema_lines = "\n".join([
        f"  - {col} ({meta['type']}): {meta['description']}"
        for col, meta in OppColumnMetadata.COLUMNS.items()
    ])

    return f"""
You are a strict JSON extraction engine for an opportunity/sales management system.
Your ONLY job is to extract structured intent from a natural language query about opportunities and sales.
You must NEVER infer, guess, or hallucinate values. Only extract what is explicitly stated.
You must NEVER generate SQL. Only return JSON.

=============================================================
TABLE CONTEXT
=============================================================
Table: opportunity_report
Columns:
{schema_lines}

Date column: created_date_c (stored as YYYYMMDD integer — do NOT mention date_parse or format strings)
Sales date column: sales_order_date_c (also YYYYMMDD integer — use ONLY when user explicitly says "sales order date", "booking date", or "order date")

=============================================================
ABSOLUTE GROUND RULES
=============================================================
1. Return ONLY valid JSON wrapped in <JSON_RESPONSE> tags. No explanation, no markdown, no SQL.
2. NEVER include a field in "filters" unless the user explicitly mentioned a value for it.
3. NEVER add a column to "group_by" unless user explicitly asked to group/split/break down by it.
4. NEVER infer a date_hint unless the user explicitly mentioned a time period.
   - If no time reference → set date_hint to null.
5. NEVER put date columns in "filters". Dates go ONLY in date_hint + date_column.
6. aggregation MUST always be a LIST, even when only one aggregation is mentioned.
7. If a field has no value to extract, omit it from filters entirely.

=============================================================
OUTPUT SCHEMA
=============================================================
{{
  "aggregation": [ <string> ],            // always a list; default ["opportunity_count"]
  "group_by":    [ <column_name> ],       // only explicitly requested groupings
  "filters":     {{ <column>: <value> }}, // only explicitly mentioned filter values
  "date_hint":   <string | null>,         // raw user phrase or null
  "date_column": "created_date_c" | "last_modified_date_c" | "sales_order_date_c"
}}

=============================================================
SECTION 1 — AGGREGATION MAPPING
=============================================================
Map user intent to exactly one of these tokens:

| User says                                                          | Token               |
|--------------------------------------------------------------------|---------------------|
| total opportunities / how many opportunities / opportunity count   | "opportunity_count" |
| opportunities / opps                                               | "opportunity_count" |
| total sales / sales count / number of sales / how many sales      | "sales_count"       |
| sale done / sales done / booking done / booked / total bookings   | "sales_count"       |

- If aggregation is unclear or not mentioned → default to ["opportunity_count"]
- aggregation is ALWAYS a list.
- "sales" queries must inject the filter: sales_order_number_c IS NOT NULL

=============================================================
SECTION 2 — FILTER COLUMN MAPPING
=============================================================

2A. SALE DONE (sales_order_number_c)
Triggers: "sale done", "sales done", "booking done", "booked", "total sales", "sales count"
→ sales_order_number_c: "__sale_done__"
This special token means: IS NOT NULL AND CAST(col AS VARCHAR) != ''

2C. PROPERTY TYPE (property_type_c)
Triggers: "residential", "commercial"
- "residential" → "Residential"
- "commercial"  → "Commercial"
NOTE: if "commercial" describes a product category, use project_category_c instead.

2D. PROPERTY SIZE (property_size_c)
Triggers: 1BHK, 2BHK, 3BHK, 4BHK, 5BHK, penthouse, skyvillas, plots
- "1bhk" / "1 bhk" → "1BHK"
- "2bhk" / "2 bhk" → "2BHK"
- "3bhk" / "3 bhk" → "3BHK"
- "4bhk" / "4 bhk" → "4BHK"
- "5bhk" / "5 bhk" → "5BHK"
- "penthouse"       → "Penthouse"
- "skyvillas"       → "Skyvillas"
- "plots"           → "Plots"

2E. LEAD SOURCE (lead_source_c) vs SUB-SOURCE (lead_source_sub_category_c)
Sub-source values: 99 acres, magicbricks, housing.com, google, facebook,
                   instagram, youtube, nobroker, inshorts, quora, spotify,
                   taboola, mygate, organic, sfmc, adgebra, rcs, live chat, etc.
Main source values: digital, direct walkin, channel partner, bulk sale,
                    outbound campaign, outdoor, print media, referral,direct,
                    word of mouth, electronic media, existing customer,
                    events / exhibitions, sms campaign, etc.

Rule: If value matches sub-source list → lead_source_sub_category_c
      If value matches main source list → lead_source_c
      Sub-source ALWAYS wins over main source when value exists in both.

2F. PRODUCT CATEGORY (project_category_c)
Trigger: user mentions a product name like: veridia, dream homes, eligo,
         eden, new plots, old plots, wave floor, prime floors, armonia villa,
         wave galleria, ews, lig, mayfair park, swamanorath, wave garden, etc.
OR if user says: "product", "product wise", "by product", "category wise",
                 "product bifurcation", "product trend", "product breakdown"
→ project_category_c
NEVER use project_c in Product Mode.

2G. PROJECT (project_c)
Trigger: "wave city", "wmcc", "wave estate", "wave amore", "wave executive floors",
         "wave city phase 1", "by project", "project wise",
         "project bifurcation", "project trend", "project breakdown"
→ project_c
Use LIKE matching (partial). NEVER use = for project_c.
"wmcc" ALWAYS maps to project_c, never to any other column.

2H. OWNER NAME (owner_name_c)
Triggers: a person's name mentioned in query (handled by, owned by, sales by)
- Full name: exact value
- Partial name: use LIKE

2I. BUDGET (range_budget_c)
Triggers: "budget", "price", "amount", "cost", amount values like "3 cr", "50 lacs"
→ range_budget_c: value as written

2J. SAP CODE (sap_customer_code_c)
Triggers: "sap code", "customer code", "sap customer code"
→ sap_customer_code_c: the code value

2K. DISQUALIFICATION REASON (disqualification_reason_c)
Triggers: user asks "disqualification reason", "why disqualified"
→ disqualification_reason_c

2L. SALES OPEN REASON (sales_open_reason_c)
Triggers: "open reason", "why open", "open opportunity reason"
→ sales_open_reason_c

=============================================================
SECTION 3 — DATE EXTRACTION
=============================================================
3A. date_column selection:
- Default: "created_date_c"
- Use "last_modified_date_c" ONLY if user explicitly says "modified", "updated", "last modified"
- Use "sales_order_date_c" ONLY if user explicitly says "sales order date", "booking date", "order date"
- Otherwise always use "created_date_c"

3B. date_hint: Return the raw user phrase describing the time period.
Examples:
- "today", "this week", "this month", "last month", "last quarter",
  "this quarter", "last year", "last 3 months", "last 6 months",
  "q1", "q2", "q3", "q4", "q1 month wise", "quarter wise",
  "month wise", "mom", "year wise", "yoy", "fy 2024",
  "april 2025", "jan to march", "2024 to 2025", null

=============================================================
SECTION 4 — GROUP_BY RULES
=============================================================
- Only include a column in group_by if the user explicitly says:
  "by <column>", "group by <column>", "break down by <column>",
  "month wise", "quarter wise", "project wise", "product wise",
  "category wise", "owner wise", "stage wise", "budget wise",
  "property size wise", "property type wise", "source wise", etc.
- NEVER infer group_by from filters.
- NEVER add a column to group_by just because it appears in filters.

=============================================================
SECTION 5 — PRODUCT vs PROJECT MODE
=============================================================
PRODUCT MODE activates when user says "product wise", "by product",
"category wise", "product bifurcation", "product breakdown", "product trend"
OR mentions a known product category name.
→ Use project_category_c. NEVER use project_c in product mode.

PROJECT MODE activates (only if product mode did NOT activate) when user says
"project wise", "by project", "project bifurcation", "project breakdown",
"project trend" OR mentions a known project name (wave city, wmcc, etc.).
→ Use project_c. NEVER use = on project_c. Always use LIKE.

MODES NEVER MIX.

=============================================================
SECTION 6 — WORKED EXAMPLES
=============================================================

Q: "Total sales this month"
A: {{"aggregation":["sales_count"],"group_by":[],"filters":{{"sales_order_number_c":"__sale_done__"}},"date_hint":"this month","date_column":"created_date_c"}}

Q: "Total opportunities this quarter by project"
A: {{"aggregation":["opportunity_count"],"group_by":["project_c"],"filters":{{}},"date_hint":"this quarter","date_column":"created_date_c"}}

Q: "Sales from veridia last month"
A: {{"aggregation":["sales_count"],"group_by":[],"filters":{{"project_category_c":"veridia","sales_order_number_c":"__sale_done__"}},"date_hint":"last month","date_column":"created_date_c"}}

Q: "Product wise sales in FY 2025"
A: {{"aggregation":["sales_count"],"group_by":["project_category_c"],"filters":{{"sales_order_number_c":"__sale_done__"}},"date_hint":"fy 2025","date_column":"created_date_c"}}

Q: "Opportunities from wave city q1 month wise"
A: {{"aggregation":["opportunity_count"],"group_by":[],"filters":{{"project_c":"wave city"}},"date_hint":"q1 month wise","date_column":"created_date_c"}}

Q: "Sales by owner year wise"
A: {{"aggregation":["sales_count"],"group_by":["owner_name_c"],"filters":{{"sales_order_number_c":"__sale_done__"}},"date_hint":"year wise","date_column":"created_date_c"}}

Q: "Total opportunities last 3 months month wise"
A: {{"aggregation":["opportunity_count"],"group_by":[],"filters":{{}},"date_hint":"last 3 months month wise","date_column":"created_date_c"}}

Q: "Show me total sales of new plots and old plots where property type is plots"
A: {{"aggregation":["sales_count"],"group_by":[],"filters":{{"project_category_c":["new plots","old plots"],"sales_order_number_c":"__sale_done__","property_type_c": ["Plots"]}},"date_hint":null,"date_column":"created_date_c"}}


Q: "2BHK sales from google last year"
A: {{"aggregation":["sales_count"],"group_by":[],"filters":{{"property_size_c":"2BHK","lead_source_sub_category_c":"Google","sales_order_number_c":"__sale_done__"}},"date_hint":"last year","date_column":"created_date_c"}}

Q: "WMCC project wise sales quarter wise"
A: {{"aggregation":["sales_count"],"group_by":["project_c"],"filters":{{"project_c":"wmcc","sales_order_number_c":"__sale_done__"}},"date_hint":"quarter wise","date_column":"created_date_c"}}

Q: "Residential opportunities"
A: {{"aggregation":["opportunity_count"],"group_by":[],"filters":{{"property_type_c":"Residential"}},"date_hint":null,"date_column":"created_date_c"}}

=============================================================
NOW PROCESS THE FOLLOWING QUERY
=============================================================
User Query: "{question}"

Return ONLY the JSON object wrapped in <JSON_RESPONSE> tags. No other text.

<JSON_RESPONSE>
"""


# ============================================================================
# LLM INTENT DETECTOR
# ============================================================================
class LLMIntentDetector:
    def extract_intent(self, question: str) -> Dict[str, Any]:
        prompt = build_opp_llm_prompt(question)
        try:
            response = llm_model.generate_text(prompt)
            logger.info(f"Raw LLM response: {response}")
            json_part = self._extract_first_json(response)
            intent    = json.loads(json_part)
            logger.info(f"Parsed LLM Intent: {intent}")
            return intent
        except Exception as e:
            logger.error(f"LLM intent parsing failed: {e}")
        return {
            "aggregation": ["opportunity_count"],
            "group_by":    [],
            "filters":     {},
            "date_hint":   None,
            "date_column": "created_date_c",
        }

    @staticmethod
    def _extract_first_json(text: str) -> str:
        tag_match = re.search(r"<JSON_RESPONSE>(.*?)</JSON_RESPONSE>", text, re.DOTALL)
        if tag_match:
            content = tag_match.group(1).strip()
            content = re.sub(r"^\s*```json\s*", "", content)
            content = re.sub(r"```\s*$", "", content)
            return content
        start_idx = text.find('{')
        end_idx   = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            potential_json = text[start_idx:end_idx + 1]
            potential_json = potential_json.replace('\\{', '{').replace('\\}', '}')
            brace_count = 0
            for i, ch in enumerate(potential_json):
                if ch == "{":   brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return potential_json[:i + 1]
        raise ValueError("No valid JSON object found in LLM response")


# ============================================================================
# INTENT DETECTOR  — normalises LLM output + deterministic extraction
# ============================================================================
class OppIntentDetector:
    def __init__(self):
        self.keywords     = OppColumnMetadata.KEYWORD_MAPPING
        self.valid_values = OppColumnMetadata.VALID_VALUES

    # ── Filter normalisation ──────────────────────────────────────────────────
    def normalize_filters(self, raw_filters: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        col_map = {k.lower(): k for k in OppColumnMetadata.COLUMNS.keys()}
        for col, values in raw_filters.items():
            if not values:
                continue
            target_col = col_map.get(col.lower(), col)

            # Product/project redirection
            if target_col != "project_category_c":
                v_str = " ".join([str(v) for v in (values if isinstance(values, list) else [values])]).lower()
                for cat in OppColumnMetadata.PRODUCT_CATEGORIES:
                    if cat in v_str:
                        target_col = "project_category_c"; break

            if target_col not in ("project_c", "project_category_c"):
                v_str = " ".join([str(v) for v in (values if isinstance(values, list) else [values])]).lower()
                for proj in OppColumnMetadata.PROJECT_VALUES:
                    if proj in v_str:
                        target_col = "project_c"; break

            normalized_values = []
            value_list = values if isinstance(values, list) else [values]

            # Expand comma / "and" separated strings
            expanded = []
            for v in value_list:
                if isinstance(v, str):
                    v_clean = re.sub(r'\s+and\s+', ',', v, flags=re.IGNORECASE)
                    if "," in v_clean:
                        expanded.extend([x.strip() for x in v_clean.split(",") if x.strip()])
                    else:
                        expanded.append(v)
                else:
                    expanded.append(v)

            for val in expanded:
                val_str = str(val).lower().strip()

                # Skip grouping-only keywords in filters
                if val_str in [k for k, v in self.keywords.items() if "value" not in v]:
                    continue

                # 1. Check KEYWORD_MAPPING deterministic
                mapped = False
                if val_str in self.keywords:
                    mapping = self.keywords[val_str]
                    if mapping.get("column") and mapping.get("value"):
                        normalized.setdefault(mapping["column"], [])
                        normalized[mapping["column"]].append(mapping["value"])
                        mapped = True
                if mapped:
                    continue

                # 2. Check VALID_VALUES exact/partial match
                if target_col in self.valid_values:
                    found = False
                    for possible in self.valid_values[target_col]:
                        if val_str == possible.lower():
                            normalized_values.append(possible); found = True; break
                    if not found and len(val_str) > 3:
                        for possible in self.valid_values[target_col]:
                            if val_str in possible.lower() or possible.lower() in val_str:
                                normalized_values.append(possible); found = True; break
                    if found:
                        continue

                # 3. Fallback — keep as-is
                normalized_values.append(val)

            if normalized_values:
                normalized.setdefault(target_col, [])
                normalized[target_col].extend(normalized_values)
                normalized[target_col] = list(set(normalized[target_col]))

        return normalized

    # ── Full intent normalisation ─────────────────────────────────────────────
    def normalize_intent(self, raw_intent: Dict[str, Any], question: str) -> Dict[str, Any]:
        q = question.lower()
        normalized_agg = []

        # 🔹 PRIORITY 1: Inventory override (hard override if required)
        if any(k in q for k in ["sales", "sale", "booking done", "booked", "total sales", "sales count"]):
            normalized_agg.append("total_sales")

        if any(k in q for k in ["opportunity", "opp", "opportunities"]):
            normalized_agg.append("opportunity_count")

        if not normalized_agg:
            # 1. Aggregation
            raw_agg = raw_intent.get("aggregation", ["opportunity_count"])
            if isinstance(raw_agg, str):
                raw_agg = [raw_agg]
            # Preserve sales_count if LLM detected it; otherwise default to opportunity_count
            normalized_agg = raw_agg if raw_agg else ["opportunity_count"]
        
        # 🔹 REMOVE DUPLICATES (IMPORTANT)
        normalized_agg = list(set(normalized_agg))

        # 2. Filters
        raw_filters          = raw_intent.get("filters", {})
        normalized_filters   = self.normalize_filters(raw_filters)
        deterministic_filters = self.extract_filters(question)
        # ─────────────────────────────────────────────────────────────
        # FIXED & IMPROVED PLOTS HANDLING
        # ─────────────────────────────────────────────────────────────
        if "plots" in q:
            has_specific_category = any(phrase in q for phrase in [
                "new plots", "old plots", "new plot", "old plot"
            ])

            has_explicit_property_size = any(phrase in q for phrase in [
                "property size", "property_size", "property type", "property_type",
                "size is plots", "size = plots", "plots size", 
                "where size", "size is", "size:"
            ])

            print(f"DEBUG Plots: specific_category={has_specific_category}, explicit_property={has_explicit_property_size}")

            # STRICT RULE:
            # Add property_size_c ONLY in these cases:
            #   - User explicitly mentions "property size / type"  OR
            #   - "plots" appears WITHOUT any "new/old plots" context
            if has_explicit_property_size or not has_specific_category:
                normalized_filters.setdefault("property_size_c", []).append("Plots")
                print("DEBUG: Added property_size_c = ['Plots']")
            else:
                deterministic_filters.pop("property_size_c", None)
                print("DEBUG: Skipped property_size_c because of specific 'new/old plots'")
        # ─────────────────────────────────────────────────────────────
        for col, values in deterministic_filters.items():
            if col not in normalized_filters:
                normalized_filters[col] = values
            else:
                normalized_filters[col] = list(set(normalized_filters[col] + values))
        # ___________________________________________________________
        
        if "owner_name_c" in raw_filters and "owner_name_c" in normalized_filters:
            # Convert both values to lists if they are strings
            raw_owner = raw_filters["owner_name_c"]
            norm_owner = normalized_filters["owner_name_c"]

            if not isinstance(raw_owner, list):
                raw_owner = [raw_owner]

            if not isinstance(norm_owner, list):
                norm_owner = [norm_owner]

            # Merge and remove duplicates
            normalized_filters["owner_name_c"] = list(set(norm_owner + raw_owner))
        else:
            # Remove ownername_c if it is present in only one of them
            normalized_filters.pop("owner_name_c", None)

        # 3. Group by
        normalized_groupby = self.extract_groupby(question)
        if not normalized_groupby:
            raw_groupby = raw_intent.get("group_by", [])
            if isinstance(raw_groupby, str):
                raw_groupby = [raw_groupby]
            col_map   = {k.lower(): k for k in OppColumnMetadata.COLUMNS.keys()}
            date_cols = set(OppColumnMetadata.DATE_COLUMNS)
            for g in raw_groupby:
                g_lower   = str(g).lower().strip()
                found_col = None
                if g_lower in self.keywords:
                    found_col = self.keywords[g_lower].get("column")
                if not found_col:
                    found_col = col_map.get(g_lower)
                if found_col and found_col.lower() not in date_cols:
                    normalized_groupby.append(found_col)

        return {
            "aggregation": normalized_agg,
            "filters":     normalized_filters,
            "group_by":    list(set(normalized_groupby)),
            "date_hint":   raw_intent.get("date_hint"),
            "date_column": raw_intent.get("date_column", "created_date_c"),
        }

    # ── Deterministic filter extraction from keywords ─────────────────────────
    def extract_filters(self, question: str) -> Dict[str, Any]:
        filters        = {}
        question_lower = question.lower()

        for keyword, mapping in self.keywords.items():
            if "column" in mapping and "value" in mapping:
                if re.search(rf"\b{re.escape(keyword)}\b", question_lower):
                    col = mapping["column"]
                    val = mapping["value"]
                    filters.setdefault(col, []).append(val)

        # Owner name detection
        name_match = re.search(
            r'\b(?:by|from|of|handled by|owned by|sales by|owner)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            question
        )
        if name_match:
            filters.setdefault("owner_name_c", []).append(name_match.group(1))

        # Budget detection
        budget_match = re.search(
            r'\b(\d+(?:\.\d+)?\s*(?:cr|crore|lac|lacs|lakh|lakhs|k)[\s\-to]+\d*(?:\.\d+)?\s*(?:cr|crore|lac|lacs|lakh|lakhs|k)?)\b',
            question_lower
        )
        if budget_match:
            filters.setdefault("range_budget_c", []).append(budget_match.group(1))

        # City detection
        CITY_LOOKUP = {city.lower(): city for city in CITY_LIST}

        for city_lower in sorted(CITY_LOOKUP.keys(), key=len, reverse=True):
            # Match whole city name as a complete phrase
            pattern = rf'\b{re.escape(city_lower)}\b'
            if re.search(pattern, question_lower):
                # Skip if the matched city is also a keyword
                if city_lower not in [k.lower() for k in self.keywords]:
                    filters.setdefault("city_c", []).append(CITY_LOOKUP[city_lower])
                break   # Stop after the first matched city

        
        return filters

    # ── Deterministic group-by extraction ────────────────────────────────────
    def extract_groupby(self, question: str) -> List[str]:
        question_lower = question.lower()
        group_by = []

        if re.search(r'\b(by source|source wise|lead source wise)\b', question_lower):
            group_by.append("lead_source_c")
        if re.search(r'\b(by sub source|sub source wise|sub-source wise)\b', question_lower):
            group_by.append("lead_source_sub_category_c")
        if re.search(r'\b(by project|project wise|per project|projectwise|project bifurcation|project breakdown|project trend)\b', question_lower):
            group_by.append("project_c")
        if re.search(r'\b(by product|product wise|category wise|all product|per product|productwise|product bifurcation|product breakdown|product trend)\b', question_lower):
            group_by.append("project_category_c")
        if re.search(r'\b(by owner|owner wise|per owner)\b', question_lower):
            group_by.append("owner_name_c")
        if re.search(r'\b(by property type|property type wise|per property type)\b', question_lower):
            group_by.append("property_type_c")
        if re.search(r'\b(by property size|property size wise|per property size)\b', question_lower):
            group_by.append("property_size_c")
        if re.search(r'\b(by budget|budget wise|per budget)\b', question_lower):
            group_by.append("range_budget_c")
        if re.search(r'\b(by customer code|customer code wise|per customer code)\b', question_lower):
            group_by.append("sap_customer_code_c")
        if re.search(r'\b(by disqualification|disqualification wise|disq wise)\b', question_lower):
            group_by.append("disqualification_reason_c")
        if re.search(r'\b(open reason wise|by open reason)\b', question_lower):
            group_by.append("sales_open_reason_c")

        return group_by

    # ── Date intent detection — 28-step cascade ───────────────────────────────
    def detect_date_intent(self, question: str):
        q = question.lower()
        logger.info(f"Detecting date intent: {q}")

        NUMBER_WORDS = {
            "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
            "seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,
            "thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
            "seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,
        }

        month_range_match    = re.search(
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*(?:to|till|-|–)\s*"
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*", q
        )
        last_n_year_match    = re.search(r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\b", q)
        last_n_month_match   = re.search(r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b", q)
        last_n_quarter_match = re.search(r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+quarters?\b", q)
        is_mom  = any(k in q for k in ["mom","month over month","month-on-month","monthly","month wise","month on month","months wise"])
        is_qoq  = any(k in q for k in ["qoq","quarter over quarter","quarter-on-quarter","quarterly","quarter wise","quarter on quarter"])
        is_yoy  = any(k in q for k in ["yoy","year on year","yearly","year wise","by year","annual trend","year over year"])
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', q))
        has_date_keyword = (
            any(word in q for word in [
                "day","month","quarter","year","date","last","this","current",
                "fy","q1","q2","q3","q4","qtr","mom","qoq","yoy","today",
                "yesterday","week","fiscal",
            ])
            or any(m in q for m in DateParser.MONTH_MAP.keys())
            or has_year
        )

        if not has_date_keyword:
            return None

        today      = datetime.today()
        current_fy = DateParser.get_current_fy()

        # 1️⃣ last Quarter + MoM
        last_q_mom = parse_quarter_mom(q)
        if last_q_mom:
            return last_q_mom

        # 2️⃣ QoQ full FY
        if is_qoq and not last_n_quarter_match and not last_n_year_match:
            # target_fy = current_fy
            # if "last year" in q or "previous year" in q:
            #     target_fy = current_fy - 1
            # else:
            #     for word in q.split():
            #         if word.isdigit() and len(word) == 4:
            #             target_fy = int(word); break
            target_fy = detect_fy(q, current_fy)
            quarters = []
            for q_num in range(1, 5):
                if q_num == 1:   s, e = datetime(target_fy,4,1),    datetime(target_fy,6,30)
                elif q_num == 2: s, e = datetime(target_fy,7,1),    datetime(target_fy,9,30)
                elif q_num == 3: s, e = datetime(target_fy,10,1),   datetime(target_fy,12,31)
                else:            s, e = datetime(target_fy+1,1,1),  datetime(target_fy+1,3,31)
                quarters.append({
                    "quarter":    f"Q{q_num} FY{target_fy}",
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date":   DateParser.date_to_yyyymmdd(e.date()),
                })
            return {"type": QueryType.QUARTER_WISE, "fy": target_fy, "quarters": quarters,
                    "label": f"Quarter-wise FY{target_fy}"}

        # 3️⃣ Specific Q1/Q2/Q3/Q4
        quarter_intent = DateParser.parse_quarter_intent(q)
        if quarter_intent:
            return quarter_intent

        # 4️⃣ Month range (jan to march)
        month_logic = parse_month_range_logic(q)
        if month_logic:
            return month_logic

        # 5️⃣ This quarter vs last quarter comparison
        if "this quarter vs last quarter" in q or "this quarter compared to last quarter" in q:
            this_q    = DateParser.get_fy_quarter(today.month)
            target_fy = current_fy
            last_q    = this_q - 1 if this_q > 1 else 4
            last_q_fy = target_fy if this_q > 1 else target_fy - 1

            def gqd(q_num, fy):
                if q_num == 1:   return datetime(fy,4,1),    datetime(fy,6,30)
                elif q_num == 2: return datetime(fy,7,1),    datetime(fy,9,30)
                elif q_num == 3: return datetime(fy,10,1),   datetime(fy,12,31)
                else:            return datetime(fy+1,1,1),  datetime(fy+1,3,31)

            ts, te = gqd(this_q, target_fy)
            ls, le = gqd(last_q, last_q_fy)
            return {
                "type": QueryType.MULTI_DATE_RANGE,
                "ranges": [
                    {"start_date": DateParser.date_to_yyyymmdd(ts.date()), "end_date": DateParser.date_to_yyyymmdd(te.date()), "label": f"This Quarter FY{target_fy} Q{this_q}"},
                    {"start_date": DateParser.date_to_yyyymmdd(ls.date()), "end_date": DateParser.date_to_yyyymmdd(le.date()), "label": f"Last Quarter FY{last_q_fy} Q{last_q}"},
                ],
                "label": "This Quarter vs Last Quarter",
            }

        # 6️⃣ Discrete months (june, july, august)
        discrete = discrete_month(q)
        if discrete:
            return discrete

        # 7️⃣ Today / yesterday
        if any(k in q for k in ["today", "today's", "todays"]):
            return {"type": QueryType.SPECIFIC_DATE, "date": DateParser.today_yyyymmdd(),
                    "label": today.strftime("%d %B %Y")}

        if "yesterday" in q or "last day" in q or "last date" in q or "previous day" in q:
            yd = today.date() - timedelta(days=1)
            return {"type": QueryType.SPECIFIC_DATE, "date": DateParser.date_to_yyyymmdd(yd),
                    "label": f"Yesterday {yd}"}

        # 8️⃣ This week
        if "this week" in q or "current week" in q:
            start_of_week = today.date() - timedelta(days=today.weekday())
            return {"type": QueryType.THIS_WEEK,
                    "start_date": DateParser.date_to_yyyymmdd(start_of_week),
                    "end_date":   DateParser.today_yyyymmdd(),
                    "label":      "This Week"}

        # 9️⃣ Last N days
        if ("last" in q or "previous" in q) and "day" in q:
            words = q.split(); n = None
            for i, word in enumerate(words):
                if word in ("last","previous") and i + 1 < len(words):
                    nxt = words[i+1]
                    if nxt == "day":    n = 1; break
                    if nxt.isdigit():   n = int(nxt); break
                    if nxt in NUMBER_WORDS: n = NUMBER_WORDS[nxt]; break
            if n:
                end   = today.date() - timedelta(days=1)
                start = end - timedelta(days=n - 1)
                return {"type": QueryType.LAST_N_DAYS,
                        "start_date": DateParser.date_to_yyyymmdd(start),
                        "end_date":   DateParser.date_to_yyyymmdd(end),
                        "label":      "Yesterday" if n == 1 else f"Last {n} Days"}

        # 🔟 Last N weeks
        if ("last" in q or "previous" in q) and "week" in q:
            words = q.split()
            for i, word in enumerate(words):
                if word in ("last","previous") and i + 1 < len(words):
                    try:
                        n = int(words[i+1]) if words[i+1].isdigit() else DateParser.WORD_TO_NUM.get(words[i+1], 1)
                        last_sunday = today.date() - timedelta(days=today.weekday() + 1)
                        end_dt      = last_sunday
                        start_dt    = end_dt - timedelta(weeks=n - 1, days=6)
                        return {"type": QueryType.LAST_N_WEEKS,
                                "start_date": DateParser.date_to_yyyymmdd(start_dt),
                                "end_date":   DateParser.date_to_yyyymmdd(end_dt),
                                "label":      f"Last {n} Week{'s' if n>1 else ''}"}
                    except Exception:
                        pass

        # 1️⃣1️⃣ This month
        if "this month" in q or "current month" in q:
            start = today.date().replace(day=1)
            return {"type": QueryType.THIS_MONTH,
                    "start_date": DateParser.date_to_yyyymmdd(start),
                    "end_date":   DateParser.today_yyyymmdd(),
                    "label":      f"This Month - {today.strftime('%B %Y')}"}

        # 1️⃣2️⃣ Last month
        if "last month" in q or "previous month" in q:
            first_of_this  = today.date().replace(day=1)
            last_day_prev  = first_of_this - timedelta(days=1)
            first_day_prev = last_day_prev.replace(day=1)
            return {"type": QueryType.LAST_MONTH,
                    "start_date": DateParser.date_to_yyyymmdd(first_day_prev),
                    "end_date":   DateParser.date_to_yyyymmdd(last_day_prev),
                    "label":      f"Last Month - {first_day_prev.strftime('%B %Y')}"}

        # 1️⃣3️⃣ MoM full FY
        if is_mom and not last_n_month_match and not last_n_quarter_match and not last_n_year_match:
            return mom_logic(q)

        # 1️⃣4️⃣ Last N months (with optional MoM)
        if last_n_month_match and not is_qoq and not is_yoy:
            return last_n_mom_logic(q)

        # 1️⃣5️⃣ This quarter
        if "this quarter" in q or "current quarter" in q:
            cq = DateParser.get_fy_quarter(today.month)
            if cq == 1:   s = datetime(current_fy, 4, 1)
            elif cq == 2: s = datetime(current_fy, 7, 1)
            elif cq == 3: s = datetime(current_fy, 10, 1)
            else:         s = datetime(current_fy+1, 1, 1)
            return {"type": QueryType.THIS_QUARTER,
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date":   DateParser.today_yyyymmdd(),
                    "label":      f"This Quarter FY{current_fy} Q{cq}"}

        # 1️⃣6️⃣ Last quarter
        if "last quarter" in q or "previous quarter" in q:
            cq   = DateParser.get_fy_quarter(today.month)
            lq   = cq - 1 if cq > 1 else 4
            lqfy = current_fy if cq > 1 else current_fy - 1
            if lq == 1:   s, e = datetime(lqfy,4,1),    datetime(lqfy,6,30)
            elif lq == 2: s, e = datetime(lqfy,7,1),    datetime(lqfy,9,30)
            elif lq == 3: s, e = datetime(lqfy,10,1),   datetime(lqfy,12,31)
            else:         s, e = datetime(lqfy+1,1,1),  datetime(lqfy+1,3,31)
            return {"type": QueryType.LAST_QUARTER,
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date":   DateParser.date_to_yyyymmdd(e.date()),
                    "label":      f"Last Quarter FY{lqfy} Q{lq}"}

        # 1️⃣7️⃣ Last N quarters
        if last_n_quarter_match and not is_yoy:
            return last_n_quarter_mom_qoq(q)

        # 1️⃣8️⃣ YoY / year wise
        if is_yoy and not last_n_year_match:
            return yoy_logic(q)

        # 1️⃣9️⃣ Year range (2022 to 2025)
        yr = year_range_logic(q)
        if yr:
            return yr

        # 2️⃣0️⃣ Multiple years (2023 and 2025)
        ya = detect_year_and(q)
        if ya:
            return ya

        # 2️⃣1️⃣ Last N years
        if last_n_year_match:
            return last_n_year_mom_qoq_yoy(q)

        # 2️⃣2️⃣ This year / this FY
        if re.search(r"\b(this|current)\s*(year|fy|fiscal\s*year)?\b", q):
            s, _ = DateParser.get_fy_start_end(current_fy)
            return {"type": QueryType.THIS_YEAR, "start_date": s,
                    "end_date": DateParser.today_yyyymmdd(),
                    "label": f"FY{current_fy} (YTD)"}

        # 2️⃣3️⃣ Last year / previous year
        if re.search(r"\b(last|previous|prev)\s*(year|fy|fiscal\s*year)?\b", q):
            s, e = DateParser.get_fy_start_end(current_fy - 1)
            return {"type": QueryType.LAST_YEAR, "start_date": s, "end_date": e,
                    "label": f"FY{current_fy - 1}"}

        # 2️⃣4️⃣ FY format (fy2024, fy 2025)
        if "fy" in q:
            for word in q.split():
                if word.startswith("fy") and word[2:].isdigit():
                    year = int(word[2:]) if len(word[2:]) == 4 else 2000 + int(word[2:])
                    s, e = DateParser.get_fy_start_end(year)
                    return {"type": QueryType.SPECIFIC_YEAR, "year": year,
                            "start_date": s, "end_date": e, "label": f"FY{year}"}

        # 2️⃣5️⃣ Specific date or date range
        date_intent = DateParser.parse_specific_date_or_range(q)
        if date_intent:
            return date_intent

        # 2️⃣6️⃣ From/after/since
        from_intent = DateParser.parse_from_date(q)
        if from_intent:
            return from_intent

        # 2️⃣7️⃣ FY till month
        till_month = DateParser.parse_fy_till_month(q)
        if till_month:
            return till_month

        # 2️⃣8️⃣ Specific month (april, may 2025)
        month_pattern = r'\b(' + '|'.join(DateParser.MONTH_MAP.keys()) + r')\b'
        month_match   = re.search(month_pattern, q, re.IGNORECASE)

        def _contains_day(text):
            return bool(re.search(r'\b\d{1,2}(st|nd|rd|th)?\b', text))

        found_month = None
        if month_match and not _contains_day(q):
            found_month = DateParser.MONTH_MAP[month_match.group(1).lower()]

        found_year  = None
        year_match  = extract_fy(q)
        if year_match:
            found_year = int(year_match)

        if found_month:
            fy_shift = -1 if re.search(r"\b(last|previous|prev)\s*(year|fy|fiscal\s*year)?\b", q) else 0
            if found_year:
                year = found_year
            else:
                eff_fy = current_fy + fy_shift
                year   = eff_fy if found_month >= 4 else eff_fy + 1
            start_dt = datetime(year, found_month, 1)
            _, ld    = monthrange(year, found_month)
            end_dt   = datetime(year, found_month, ld)
            return {"type": QueryType.DATE_RANGE,
                    "start_date": DateParser.date_to_yyyymmdd(start_dt.date()),
                    "end_date":   DateParser.date_to_yyyymmdd(end_dt.date()),
                    "label":      start_dt.strftime("%B %Y")}

        if found_year:
            s, e = DateParser.get_fy_start_end(found_year)
            return {"type": QueryType.SPECIFIC_YEAR, "year": found_year,
                    "start_date": s, "end_date": e, "label": f"FY{found_year}"}

        return None


# ============================================================================
# SQL GENERATOR
# ============================================================================
def _numeric_expr(col: str) -> str:
    return f"TRY_CAST({col} AS DECIMAL(18,2))"

def parse_numeric_condition(col: str, value: str) -> Optional[str]:
    v = value.strip().lower()
    patterns = [
        (r"^(>=)\s*(\d+(\.\d+)?)$", ">="),
        (r"^(<=)\s*(\d+(\.\d+)?)$", "<="),
        (r"^(>)\s*(\d+(\.\d+)?)$",  ">"),
        (r"^(<)\s*(\d+(\.\d+)?)$",  "<"),
        (r"^(=)\s*(\d+(\.\d+)?)$",  "="),
    ]
    for pattern, op in patterns:
        m = re.match(pattern, v)
        if m:
            return f"{col} {op} {m.group(2)}"
    return None


def detect_fy(q: str, current_fy: int) -> int:
    q = q.lower().strip()
    q = re.sub(r"\s+", " ", q)

    # Last FY
    if re.search(r"\b(last|previous|prev)\s*(fy|fiscal\s*year|year)?\b", q):
        return current_fy - 1

    # fy2023, fy 2023
    m = re.search(r"\bfy\s*(20\d{2})\b", q)
    if m:
        return int(m.group(1))

    # fy23
    m = re.search(r"\bfy\s*(\d{2})\b", q)
    if m:
        yy = int(m.group(1))
        return 2000 + yy

    # plain 2023
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        return int(m.group(1))

    return current_fy



class OppSQLGenerator:

    @staticmethod
    def _convert_date_filter(date_filter: str) -> str:
        """
        Convert YYYYMMDD BETWEEN format → TRY(date_parse(..., '%d-%m-%Y')) format.

        Input:  "created_date_c BETWEEN 20250401 AND 20260331"
        Output: "TRY(date_parse(created_date_c, '%d-%m-%Y'))
                 BETWEEN TRY(date_parse('01-04-2025', '%d-%m-%Y'))
                 AND TRY(date_parse('31-03-2026', '%d-%m-%Y'))"
        """
        def yyyymmdd_to_ddmmyyyy(s: str) -> str:
            if len(s) != 8 or not s.isdigit():
                return s
            return f"{s[6:8]}-{s[4:6]}-{s[:4]}"

        # BETWEEN pattern
        between_match = re.search(
            r'(\w+)\s+BETWEEN\s+(\d{8})\s+AND\s+(\d{8})',
            date_filter, re.IGNORECASE
        )
        if between_match:
            col   = between_match.group(1)
            start = yyyymmdd_to_ddmmyyyy(between_match.group(2))
            end   = yyyymmdd_to_ddmmyyyy(between_match.group(3))
            return (
                f"TRY(date_parse({col}, '%d-%m-%Y')) "
                f"BETWEEN TRY(date_parse('{start}', '%d-%m-%Y')) "
                f"AND TRY(date_parse('{end}', '%d-%m-%Y'))"
            )

        # >= pattern
        ge_match = re.search(r'(\w+)\s+(>=)\s+(\d{8})', date_filter)
        if ge_match:
            col  = ge_match.group(1)
            dstr = yyyymmdd_to_ddmmyyyy(ge_match.group(3))
            return f"TRY(date_parse({col}, '%d-%m-%Y')) >= TRY(date_parse('{dstr}', '%d-%m-%Y'))"

        # <= pattern
        le_match = re.search(r'(\w+)\s+(<=)\s+(\d{8})', date_filter)
        if le_match:
            col  = le_match.group(1)
            dstr = yyyymmdd_to_ddmmyyyy(le_match.group(3))
            return f"TRY(date_parse({col}, '%d-%m-%Y')) <= TRY(date_parse('{dstr}', '%d-%m-%Y'))"

        return date_filter

    @staticmethod
    def build_where_clause(
        filters:     Dict[str, Any],
        date_filter: Optional[str] = None,
    ) -> str:
        conditions = []

        # Date filter first — convert YYYYMMDD → dd-mm-yyyy date_parse format
        if date_filter:
            conditions.append(OppSQLGenerator._convert_date_filter(date_filter))

        for col, values in filters.items():
            col_meta       = OppColumnMetadata.COLUMNS.get(col, {})
            col_type       = col_meta.get("type", "VARCHAR")
            col_conditions = []
            value_list     = values if isinstance(values, list) else [values]

            for v in value_list:
                v_str = str(v).lower().strip()

                # ── Special token: sale done ───────────────────────────────────
                if v_str == "__sale_done__":
                    col_conditions.append(
                        f"({col} IS NOT NULL AND CAST({col} AS VARCHAR) != '')"
                    )
                    continue

                if col_type in ("INTEGER", "DOUBLE"):
                    nc = parse_numeric_condition(col, v_str)
                    if nc:
                        col_conditions.append(nc)
                    continue

                # SAP code — exact match
                if col == "sap_customer_code_c":
                    col_conditions.append(f"{col} = '{v}'")
                    continue

                # Owner name — exact for full name, LIKE for partial
                if col == "owner_name_c":
                    if " " in v_str:
                        col_conditions.append(f"LOWER({col}) = '{v_str}'")
                    else:
                        col_conditions.append(f"LOWER({col}) LIKE '%{v_str}%'")
                    continue

                # project_c — always LIKE (never =)
                if col == "project_c":
                    like_val = v_str.replace(" ", "%")
                    col_conditions.append(f"LOWER({col}) LIKE '%{like_val}%'")
                    continue

                # range_budget_c — LIKE with both space and no-space variants
                if col == "range_budget_c":
                    search_val = v_str.lower()
                    # Remove spaces before "cr" and "lac"
                    normalized_val = re.sub(r'\s+(cr|lac)\b', r'\1', search_val)

                    col_conditions.append(
                        f"""(LOWER({col}) LIKE '%{search_val}%' OR LOWER({col}) LIKE '%{normalized_val}%')"""
                    )
                    continue

                # Default: LIKE with wildcards
                col_conditions.append(f"LOWER({col}) LIKE '%{v_str}%'")

            if col_conditions:
                join_op = " AND " if col_type in ("INTEGER","DOUBLE") else " OR "
                if len(col_conditions) == 1:
                    conditions.append(col_conditions[0])
                else:
                    conditions.append("(" + join_op.join(col_conditions) + ")")

        return "WHERE " + " AND ".join(conditions) if conditions else ""

    @staticmethod
    def generate_sql(
        catalog:     str,
        schema:      str,
        table:       str,
        agg_infos:   List[Dict],
        group_by:    List[str],
        filters:     Dict[str, Any],
        date_range:  Optional[Tuple[str, str]] = None,
        date_column: str = "created_date_c",
        period_type: Optional[str] = None,
    ) -> str:
        select_parts   = []
        group_by_parts = []

        # Period column (month / quarter / year)
        if period_type == "month":
            date_expr = f"TRY(date_parse({date_column}, '%d-%m-%Y'))"
            yr_expr   = f"CAST(year({date_expr}) AS VARCHAR)"
            mo_expr   = f"LPAD(CAST(month({date_expr}) AS VARCHAR), 2, '0')"
            time_expr = f"({yr_expr} || '-' || {mo_expr})"
            select_parts.append(f"{time_expr} AS period")
            group_by_parts.append(time_expr)

        elif period_type == "quarter":
            month_expr = f"MONTH(TRY(date_parse({date_column}, '%d-%m-%Y')))"
            year_expr  = f"YEAR(TRY(date_parse({date_column}, '%d-%m-%Y')))"
            q_expr     = (
                f"CASE "
                f"WHEN {month_expr} IN (4,5,6) THEN 'Q1' "
                f"WHEN {month_expr} IN (7,8,9) THEN 'Q2' "
                f"WHEN {month_expr} IN (10,11,12) THEN 'Q3' "
                f"WHEN {month_expr} IN (1,2,3) THEN 'Q4' "
                f"END"
            )
            fy_expr = (
                f"CASE "
                f"WHEN {month_expr} >= 4 THEN {year_expr} "
                f"ELSE {year_expr} - 1 "
                f"END"
            )
            time_expr = f"CONCAT({q_expr}, ' FY', CAST({fy_expr} AS VARCHAR))"
            select_parts.append(f"{time_expr} AS period")
            group_by_parts.append(time_expr)

        elif period_type == "year":
            date_expr = f"TRY(date_parse({date_column}, '%d-%m-%Y'))"
            fy_expr   = (
                f"CASE "
                f"WHEN month({date_expr}) >= 4 "
                f"THEN CAST(year({date_expr}) AS VARCHAR) || '-' || CAST(year({date_expr}) + 1 AS VARCHAR) "
                f"ELSE CAST(year({date_expr}) - 1 AS VARCHAR) || '-' || CAST(year({date_expr}) AS VARCHAR) "
                f"END"
            )
            select_parts.append(f"{fy_expr} AS period")
            group_by_parts.append(fy_expr)

        # User-requested group_by columns
        select_parts.extend(group_by)
        group_by_parts.extend(group_by)

        # Aggregation
        for agg in agg_infos:
            if agg["type"] == AggregationType.COUNT:
                agg_expr = f'COUNT({agg["column"]}) AS "{agg["alias"]}"'
            elif agg["type"] == AggregationType.SUM:
                agg_expr = f'SUM({_numeric_expr(agg["column"])}) AS "{agg["alias"]}"'
            elif agg["type"] == AggregationType.AVG:
                agg_expr = f'AVG({agg["column"]}) AS "{agg["alias"]}"'
            elif agg["type"] == AggregationType.MIN:
                agg_expr = f'MIN({agg["column"]}) AS "{agg["alias"]}"'
            elif agg["type"] == AggregationType.MAX:
                agg_expr = f'MAX({agg["column"]}) AS "{agg["alias"]}"'
            else:
                raise ValueError(f"Unsupported aggregation: {agg['type']}")
            select_parts.append(agg_expr)

        select_clause = "SELECT " + ", ".join(select_parts)
        from_clause   = f'FROM "{catalog}"."{schema}"."{table}"'

        # Date filter — YYYYMMDD integer BETWEEN
        date_filter = None
        if date_range:
            start, end = date_range
            if start and end:
                date_filter = f"{date_column} BETWEEN {start} AND {end}"
            elif start:
                date_filter = f"{date_column} >= {start}"
            elif end:
                date_filter = f"{date_column} <= {end}"

        where_clause    = OppSQLGenerator.build_where_clause(filters=filters, date_filter=date_filter)
        group_by_clause = ("GROUP BY " + ", ".join(group_by_parts)) if group_by_parts else ""

        sql = "\n".join(filter(None, [select_clause, from_clause, where_clause, group_by_clause]))
        return sql.strip()


# ============================================================================
# SQL VALIDATOR
# ============================================================================
class OppSQLValidator:
    @staticmethod
    def validate(sql: str) -> Tuple[bool, Optional[str]]:
        if not sql or not sql.strip():
            return False, "SQL query is empty"
        sql_upper = sql.upper()
        if "SELECT" not in sql_upper:
            return False, "Missing SELECT clause"
        if "FROM" not in sql_upper:
            return False, "Missing FROM clause"
        if sql.count("(") != sql.count(")"):
            return False, "Unbalanced parentheses"
        for kw in ["DROP","DELETE","TRUNCATE","ALTER","CREATE","INSERT","UPDATE"]:
            if f" {kw} " in sql_upper:
                return False, f"Dangerous keyword detected: {kw}"
        return True, "SQL is valid"


# ============================================================================
# PRESTO EXECUTION
# ============================================================================
def run_presto_query(sql: str) -> Tuple[List[Dict], Optional[List[str]], Optional[str]]:
    try:
        with prestodb.dbapi.connect(
            host=PRESTO_HOST,
            port=PRESTO_PORT,
            user=PRESTO_USER,
            catalog=PRESTO_CATALOG,
            schema=PRESTO_SCHEMA,
            http_scheme="https",
            auth=BasicAuthentication(PRESTO_USER, PRESTO_PASSWORD),
        ) as conn:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description] if cur.description else []
            result = [dict(zip(cols, row)) for row in rows]
            return result, cols, None
    except Exception as e:
        logger.error(f"Presto execution error: {e}", exc_info=True)
        return [], [], f"Presto execution failed: {str(e)}"


# ============================================================================
# POST-PROCESSING: enforce DESC order on numeric columns
# ============================================================================
def enforce_descending_order(sql: str) -> str:
    TEMPORAL_COLS = {"period", "fy_year", "month", "quarter", "year"}
    sql = sql.rstrip(";").strip()
    if re.search(r'\bORDER BY\b', sql, re.IGNORECASE):
        def fix_order(m):
            parts     = [p.strip() for p in m.group(1).split(",")]
            new_parts = []
            for part in parts:
                tokens   = part.split()
                col_name = tokens[0].lower().strip('"').strip("'")
                if col_name in TEMPORAL_COLS:
                    new_parts.append(part)
                elif len(tokens) == 1:
                    new_parts.append(part + " DESC")
                elif tokens[-1].upper() == "ASC":
                    new_parts.append(" ".join(tokens[:-1]) + " DESC")
                else:
                    new_parts.append(part)
            return "ORDER BY " + ", ".join(new_parts)
        sql = re.sub(r"ORDER BY\s+(.+?)$", fix_order, sql, flags=re.IGNORECASE)
    else:
        alias_match = re.search(
            r'\bAS\s+"?(opportunity_count|sales_count|opp_count|total_sales|total_opportunities|count)\b"?',
            sql, re.IGNORECASE
        )
        if alias_match:
            sql += f' ORDER BY "{alias_match.group(1)}" DESC'
    return sql


# ============================================================================
# POST-PROCESSING: totals
# ============================================================================
NON_ADDITIVE_MARKERS  = ["%", ":"]
NON_ADDITIVE_KEYS     = {"fy_year","month","quarter","days","year","financial year","fiscal year"}
_TIME_DIMENSION_WORDS = {"year","month","quarter","week","day","fiscal"}

def _is_additive_key(key: str) -> bool:
    key_lower = key.lower()
    if key_lower in NON_ADDITIVE_KEYS:
        return False
    if any(w in key_lower for w in _TIME_DIMENSION_WORDS):
        return False
    return not any(m in key for m in NON_ADDITIVE_MARKERS)

def add_total_row(data: list) -> list:
    if not data or len(data) <= 1:
        return data
    total_row      = {}
    first_str_done = False
    for key in data[0].keys():
        col_values = [row.get(key) for row in data]
        non_null   = [v for v in col_values if v is not None]
        if not _is_additive_key(key):
            total_row[key] = "-" if first_str_done else "Total"
            if not first_str_done:
                first_str_done = True
            continue
        try:
            nums = [float(v) for v in non_null]
            total_row[key] = round(sum(nums), 2)
        except (ValueError, TypeError):
            if not first_str_done:
                total_row[key] = "Total"; first_str_done = True
            else:
                total_row[key] = "-"
    data.append(total_row)
    return data


# ============================================================================
# AGGREGATION MAP  — opportunity_count vs sales_count
# ============================================================================
def build_agg_map() -> Dict[str, Dict]:
    return {
        "opportunity_count": {
            "type":   AggregationType.COUNT,
            "column": "lead_id_c",
            "alias":  "Opportunity Count",
        },
        "sales_count": {
            "type":   AggregationType.COUNT,
            "column": "sales_order_number_c",
            "alias":  "Sales Count",
        },
        "total_sales": {
            "type":   AggregationType.COUNT,
            "column": "sales_order_number_c",
            "alias":  "Sales Count",
        },
    }


# ============================================================================
# MAIN ENGINE
# ============================================================================
class OppNLToSQLEngine:
    def __init__(self):
        self.llm_detector    = LLMIntentDetector()
        self.intent_detector = OppIntentDetector()
        self.sql_generator   = OppSQLGenerator()
        self.sql_validator   = OppSQLValidator()

    def process(self, request: OppSQLRequest) -> OppSQLResponse:
        question = request.question.strip()
        logger.info(f"Processing question: {question}")

        # ── Step 1: LLM extracts JSON intent ──────────────────────────────────
        llm_intent = self.llm_detector.extract_intent(question)

        # ── Step 2: Deterministic normalisation locks hallucinations ──────────
        llm_intent = self.intent_detector.normalize_intent(llm_intent, question)
        logger.info(f"Normalized Intent: {llm_intent}")

        # ── Step 3: Python deterministically parses date intent ───────────────
        date_intent = self.intent_detector.detect_date_intent(question)
        logger.info(f"Resolved date intent: {date_intent}")

        if date_intent is None:
            fy    = DateParser.get_current_fy()
            start, end = DateParser.get_fy_start_end(fy)
            date_intent = {
                "type":       QueryType.CURRENT_FY,
                "start_date": start,
                "end_date":   end,
                "label":      f"FY{fy}",
            }

        query_type = date_intent["type"]
        logger.info(f"Resolved query type: {query_type}")

        # ── Step 4: Aggregation map ────────────────────────────────────────────
        agg_map  = build_agg_map()
        raw_aggs = llm_intent.get("aggregation", ["opportunity_count"])
        agg_infos = []
        for agg in raw_aggs:
            key = agg.strip().lower().replace(" ", "_")
            if key in agg_map:
                agg_infos.append(agg_map[key])
            else:
                logger.warning(f"Unknown aggregation: {agg}")
        if not agg_infos:
            agg_infos.append(agg_map["opportunity_count"])

        # ── Step 5: Filters and group_by ──────────────────────────────────────
        filters  = llm_intent.get("filters", {})
        group_by = llm_intent.get("group_by", [])
        logger.info(f"Resolved filters: {filters}")
        logger.info(f"Resolved group_by: {group_by}")

        # ── Step 6: Date column validation ────────────────────────────────────
        date_column = llm_intent.get("date_column", "created_date_c")
        if date_column not in OppColumnMetadata.DATE_COLUMNS:
            logger.warning(f"Invalid date column: {date_column}, fallback to created_date_c")
            date_column = "created_date_c"

        # Normalise SPECIFIC_DATE → DATE_RANGE
        if date_intent.get("type") == QueryType.SPECIFIC_DATE:
            date_intent = {
                "type":       QueryType.DATE_RANGE,
                "start_date": date_intent["date"],
                "end_date":   date_intent["date"],
                "label":      date_intent.get("label"),
            }

        # ── Step 7: SQL generation ────────────────────────────────────────────
        date_ranges = []
        MULTI_PERIOD_TYPES = [
            QueryType.QUARTER_WISE, QueryType.MONTH_WISE, QueryType.YEAR_WISE,
            QueryType.MULTI_DATE_RANGE, QueryType.MONTH_RANGE_MONTH_WISE,
            QueryType.MONTH_MULTI_MONTH_WISE, QueryType.MULTI_MONTH,
        ]

        if query_type in MULTI_PERIOD_TYPES:
            periods = (
                date_intent.get("quarters") or
                date_intent.get("periods")  or
                date_intent.get("ranges")   or []
            )
            if not periods:
                raise ValueError("No periods defined for multi-period query")

            sqls = []
            for period in periods:
                period_start = period["start_date"]
                period_end   = period["end_date"]
                period_label = (
                    period.get("quarter") or
                    period.get("label")   or
                    period.get("year", f"{period_start}-{period_end}")
                )
                if query_type in [QueryType.MONTH_WISE, QueryType.MONTH_RANGE_MONTH_WISE,
                                   QueryType.MONTH_MULTI_MONTH_WISE, QueryType.MULTI_MONTH]:
                    period_type = "month"
                elif query_type in [QueryType.QUARTER_WISE, QueryType.MULTI_DATE_RANGE]:
                    period_type = "quarter"
                elif query_type == QueryType.YEAR_WISE:
                    period_type = "year"
                else:
                    period_type = None

                sql = OppSQLGenerator.generate_sql(
                    catalog=request.catalog,
                    schema=request.db_schema,
                    table=request.table,
                    agg_infos=agg_infos,
                    group_by=group_by,
                    filters=filters,
                    date_range=(period_start, period_end),
                    date_column=date_column,
                    period_type=period_type,
                )
                sqls.append(sql)
                date_ranges.append((period_start, period_end, period_label))

            metric_alias = agg_infos[0]["alias"] if agg_infos else "Opportunity Count"
            final_sql    = "\n\nUNION ALL\n\n".join(sqls)
            final_sql    = f"(\n{final_sql}\n) ORDER BY \"{metric_alias}\" DESC"
            if group_by:
                final_sql += ", " + ", ".join(group_by)

        else:
            # Single period
            final_sql = OppSQLGenerator.generate_sql(
                catalog=request.catalog,
                schema=request.db_schema,
                table=request.table,
                agg_infos=agg_infos,
                group_by=group_by,
                filters=filters,
                date_range=(date_intent["start_date"], date_intent.get("end_date")),
                date_column=date_column,
                period_type=None,
            )
            date_ranges.append((
                date_intent["start_date"],
                date_intent.get("end_date"),
                date_intent.get("label"),
            ))
            metric_alias = agg_infos[0]["alias"] if agg_infos else "Opportunity Count"
            if group_by:
                final_sql += f'\nORDER BY "{metric_alias}" DESC'

        # ── Step 8: Validate → Execute → Post-process ─────────────────────────
        final_sql = enforce_descending_order(final_sql)
        is_valid, validation_msg = self.sql_validator.validate(final_sql)
        data, schema_cols, error_msg = (
            run_presto_query(final_sql) if is_valid else ([], [], validation_msg)
        )
        data   = add_total_row(data)
        totals = {k: v for row in data for k, v in row.items()
                  if isinstance(v, (int, float)) and _is_additive_key(k)}

        logger.info(f"Generated SQL:\n{final_sql}")

        return OppSQLResponse(
            status             = "success" if is_valid and not error_msg else "error",
            query_type         = query_type,
            sql                = final_sql,
            schema_metadata    = schema_cols,
            data               = data,
            execution          = {"executed": is_valid, "row_count": len(data), "error": error_msg},
            date_ranges        = [DateRange(start_date=s, end_date=e, label=l) for s, e, l in date_ranges],
            is_valid           = is_valid,
            validation_message = validation_msg,
            metadata           = {"llm_intent": llm_intent, "date_intent": date_intent},
            intent_summary     = {
                "aggregation":      [a["alias"] for a in agg_infos],
                "filters_detected": filters,
                "group_by_columns": group_by,
                "date_range":       date_intent.get("label") or "Multiple Periods",
            },
            totals = totals,
        )


# ============================================================================
# FASTAPI APP
# ============================================================================
app    = FastAPI(title="Opportunity NL-to-SQL Engine", version="1.0.0")
engine = OppNLToSQLEngine()


@app.post("/generate-sql", response_model=OppSQLResponse)
async def generate_sql(request: OppSQLRequest):
    try:
        return engine.process(request)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", "date_format": "YYYYMMDD integer → dd-mm-yyyy date_parse"}


if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 70)
    logger.info("OPPORTUNITY NL-TO-SQL ENGINE")
    logger.info("LLM: intent JSON only | Python: date parsing + SQL generation")
    logger.info("Date format: YYYYMMDD integer (converted to date_parse in WHERE)")
    logger.info("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8002)