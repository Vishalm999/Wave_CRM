from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime, timedelta, date
from calendar import monthrange
import calendar
import logging
from enum import Enum
import prestodb
from prestodb.auth import BasicAuthentication
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from dotenv import load_dotenv
import os
import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uuid
import base64
import textwrap
import ibm_boto3
from ibm_botocore.client import Config


load_dotenv()
COS_API_KEY_ID   = os.getenv("COS_API_KEY_ID")
COS_RESOURCE_CRN = os.getenv("COS_RESOURCE_CRN")
COS_ENDPOINT     = os.getenv("COS_ENDPOINT")
COS_BUCKET       = os.getenv("COS_BUCKET")

cos = ibm_boto3.client(
    "s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_RESOURCE_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

graph_store = {}
session_data_store = {}
# ============================================================================
# LOGGING & CONFIG
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("task_nl_to_sql.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PRESTO_HOST     = os.getenv("PRESTO_hostname")
PRESTO_PORT     = int(os.getenv("PRESTO_portnumber", "30984"))
PRESTO_USER     = os.getenv("PRESTO_username")
PRESTO_PASSWORD = os.getenv("PRESTO_password")
PRESTO_CATALOG  = "salesforcereport"
PRESTO_SCHEMA   = "task_sf_report"
PRESTO_TABLE    = "task_report"

WATSONX_API_KEY    = os.getenv("API_KEY")
WATSONX_PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

credentials = Credentials(url=WATSONX_URL, api_key=PRESTO_PASSWORD)
llm_model = ModelInference(
    model_id="meta-llama/llama-3-3-70b-instruct",
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID,
    params={"max_new_tokens": 500, "temperature": 0.1}
)

# ============================================================================
# TASK SCHEMA  — mirrors LeadColumnMetadata / ColumnMetadata pattern
# ============================================================================
class TaskColumnMetadata:

    COLUMNS = {
        # ── Date columns stored as YYYYMMDD INTEGER ────────────────────────
        "created_date_c":          {"type": "INTEGER", "description": "Task creation date in YYYYMMDD integer format"},
        "lastmodified_date_c":     {"type": "INTEGER", "description": "Last modified date in YYYYMMDD integer format"},

        # ── Key dimension columns ──────────────────────────────────────────
        "activity_id_c":           {"type": "VARCHAR", "description": "Task Activity Id This column Used for unique Key"},
        "subject_c":               {"type": "VARCHAR", "description": "Task/activity subject: Follow Up, Sales Follow Up, Re-query requested, Tried Calling, Click to call, Call Back By Sales Expert, Experience Calling Follow Up, Welcome Calling Follow Up, Request call back, Live Chat Query, Phone Call Activity"},
        "status_c":                {"type": "VARCHAR", "description": "Task status: Completed, Open, In Progress, Cancelled, Closed, Deferred"},
        "ownername_c":             {"type": "VARCHAR", "description": "Task owner name (person responsible for the task)"},
        "project_c":               {"type": "VARCHAR", "description": "Project zone/sales org (wave city, wmcc, wave estate, etc.) — use LIKE, never ="},
        "product_category_c":      {"type": "VARCHAR", "description": "Product/unit category (veridia, dream homes, eden, old plots, etc.) — use LIKE, never ="},
        "customer_feedback_c":     {"type": "VARCHAR", "description": "Customer feedback: Discussion Pending, Interested, Not Interested, Junk"},
        "sales_team_feedback_c":   {"type": "VARCHAR", "description": "Sales team feedback: Qualified, Disqualified, In Followup"},
        "follow_up_status_c":      {"type": "VARCHAR", "description": "Follow Up Status: Completed, In Progress, Open , Cancelled"},
        "transfer_status_c":       {"type": "VARCHAR", "description": "Transfer Status: call back by sales expert, email requested, personal appointment booked"},
    }

    # ── Date columns (YYYYMMDD INTEGER) ──────────────────────────────────────
    DATE_COLUMNS = ["created_date_c", "lastmodified_date_c"]

    # ── Default date column ───────────────────────────────────────────────────
    DEFAULT_DATE_COLUMN = "created_date_c"

    # ── Count column (equivalent to Material_Code / lead_id_c) ───────────────
    COUNT_COLUMN = "activity_id_c"   # fallback: COUNT(*) when no distinct key

    # ── Dimension columns (groupable) ────────────────────────────────────────
    DIMENSION_COLUMNS = [
        "subject_c", "status_c", "ownername_c",
        "project_c", "product_category_c", "customer_feedback_c",
         "sales_team_feedback_c","follow_up_status_c", "transfer_status_c"
    ]

    # ── Product categories (filter on product_category_c) ────────────────────
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

    # ── Project values (filter on project_c) ─────────────────────────────────
    PROJECT_VALUES = [
        "wave city", "wmcc", "wmcc sec 32", "wmcc sector 32",
        "wave estate", "wave amore", "wave executive floors",
    ]

    # ── Valid status values ───────────────────────────────────────────────────
    VALID_STATUS = ["Completed", "Open", "In Progress", "Cancelled", "Closed", "Deferred"]

    # ── KEYWORD MAPPING — mirrors inventory/lead pattern exactly ──────────────
    KEYWORD_MAPPING = {

        # ── Task subject keywords ─────────────────────────────────────────────
        "follow up":                    {"column": "subject_c", "value": "__followup__"},
        "followup":                     {"column": "subject_c", "value": "__followup__"},
        "follow-up":                    {"column": "subject_c", "value": "__followup__"},
        "follow up task":               {"column": "subject_c", "value": "__followup__"},
        "followup task":                {"column": "subject_c", "value": "__followup__"},
        "sales follow up":              {"column": "subject_c", "value": "Sales Follow Up"},
        "sales followup":               {"column": "subject_c", "value": "Sales Follow Up"},
        "seles follow up":              {"column": "subject_c", "value": "Sales Follow Up"},
        "re query":                     {"column": "subject_c", "value": "Re-query requested"},
        "requery":                      {"column": "subject_c", "value": "Re-query requested"},
        "re-query":                     {"column": "subject_c", "value": "Re-query requested"},
        "re query requested":           {"column": "subject_c", "value": "Re-query requested"},
        "tried calling":                {"column": "subject_c", "value": "__tried_calling__"},
        "click to call":                {"column": "subject_c", "value": "__click_to_call__"},
        "click call":                   {"column": "subject_c", "value": "__click_to_call__"},
        "call back by sales expert":    {"column": "subject_c", "value": "Call Back By Sales Expert"},
        "callback by sales expert":     {"column": "subject_c", "value": "Call Back By Sales Expert"},
        "experience calling follow up": {"column": "subject_c", "value": "Experience Calling Follow Up"},
        "welcome calling follow up":    {"column": "subject_c", "value": "Welcome Calling Follow Up"},
        "request call back":            {"column": "subject_c", "value": "Request call back"},
        "request callback":             {"column": "subject_c", "value": "Request call back"},
        "live chat":                    {"column": "subject_c", "value": "Live Chat Query"},
        "live chat query":              {"column": "subject_c", "value": "Live Chat Query"},
        "personal appointment":         {"column": "transfer_status_c", "value": "Personal Appointment Booked"},
        "email request":                {"column": "transfer_status_c", "value": "Email Requested"},

        # ── Task status keywords ──────────────────────────────────────────────
        "completed task":    {"column": "status_c", "value": "Completed"},
        "completed":         {"column": "status_c", "value": "Completed"},
        "open task":         {"column": "status_c", "value": "Open"},
        "open":              {"column": "status_c", "value": "Open"},
        "in progress":       {"column": "status_c", "value": "In Progress"},
        "cancelled task":    {"column": "status_c", "value": "__cancelled__"},
        "canceled task":     {"column": "status_c", "value": "__cancelled__"},
        "cancelled":         {"column": "status_c", "value": "__cancelled__"},
        "canceled":          {"column": "status_c", "value": "__cancelled__"},
        "deferred":          {"column": "status_c", "value": "Deferred"},
        "closed":            {"column": "status_c", "value": "Closed"},

        # ── Customer feedback keywords ────────────────────────────────────────
        "discussion pending":   {"column": "customer_feedback_c", "value": "Discussion Pending"},
        "pending":              {"column": "customer_feedback_c", "value": "Discussion Pending"},
        "interested":           {"column": "customer_feedback_c", "value": "Interested"},
        "not interested":       {"column": "customer_feedback_c", "value": "Not Interested"},
        "notinterested":        {"column": "customer_feedback_c", "value": "Not Interested"},
        "junk":                 {"column": "customer_feedback_c", "value": "Junk"},

        # ── Sales team feedback keywords ──────────────────────────────────────
        "qualified":       {"column": "sales_team_feedback_c", "value": "Qualified"},
        "disqualified":    {"column": "sales_team_feedback_c", "value": "Disqualified"},
        "in followup":     {"column": "sales_team_feedback_c", "value": "In Followup"},

        # ── Project zone keywords (project_c) ─────────────────────────────────
        "wave city":      {"column": "project_c", "value": "wave city"},
        "wmcc":           {"column": "project_c", "value": "wmcc"},
        "wmcc sec 32":    {"column": "project_c", "value": "wmcc sec 32"},
        "wmcc sector 32": {"column": "project_c", "value": "wmcc sec 32"},
        "wave estate":    {"column": "project_c", "value": "wave estate"},

        # ── Grouping-only keywords ────────────────────────────────────────────
        "by project":          {"column": "project_c"},
        "project wise":        {"column": "project_c"},
        "by product":          {"column": "product_category_c"},
        "product wise":        {"column": "product_category_c"},
        "category wise":       {"column": "product_category_c"},
        "by status":           {"column": "status_c"},
        "status wise":         {"column": "status_c"},
        "by subject":          {"column": "subject_c"},
        "subject wise":        {"column": "subject_c"},
        "by owner":            {"column": "ownername_c"},
        "owner wise":          {"column": "ownername_c"},
        "user wise":           {"column": "ownername_c"},
        "by sales feedback":   {"column": "sales_team_feedback_c"},
        "sales feedback wise": {"column": "sales_team_feedback_c"},


        # ── Aggregation keywords ──────────────────────────────────────────────
        "total task":   {"aggregation": "count", "column": "*"},
        "total tasks":  {"aggregation": "count", "column": "*"},
        "task count":   {"aggregation": "count", "column": "*"},
        "count":        {"aggregation": "count", "column": "*"},
        "how many":     {"aggregation": "count", "column": "*"},
    }


# ============================================================================
# QUERY TYPE ENUM  (identical to inventory / lead)
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
class TaskSQLRequest(BaseModel):
    question:  str = Field(..., description="Natural language query about tasks")
    catalog:   str = Field(default=PRESTO_CATALOG)
    db_schema: str = Field(default=PRESTO_SCHEMA)
    table:     str = Field(default=PRESTO_TABLE)

class DateRange(BaseModel):
    start_date: str
    end_date:   Optional[str] = None
    label:      Optional[str] = None

class TaskSQLResponse(BaseModel):
    status:             str
    query_type:         str
    sql:                str
    schema_metadata:    Optional[List[str]]     = Field(default=None, alias="schema")
    data:               Optional[List[Dict[str, Any]]] = None
    execution:          Optional[Dict[str, Any]] = None
    date_ranges:        List[DateRange]
    is_valid:           bool
    validation_message: Optional[str]          = None
    metadata:           Optional[Dict[str, Any]] = None
    intent_summary:     Optional[Dict[str, Any]] = None
    totals:             Optional[Dict[str, Any]] = None

    class Config:
        validate_by_name = True
        populate_by_name = True
        use_enum_values = True
        json_encoders = {
            QueryType: lambda v: v.value if isinstance(v, QueryType) else str(v),
        }


# ============================================================================
# DATE PARSER  — identical to inventory/lead (YYYYMMDD, Apr–Mar FY)
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

        # yyyy-mm-dd
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            pass

        # dd/mm/yyyy or dd-mm-yyyy
        try:
            return datetime.strptime(text.replace("-", "/"), "%d/%m/%Y").date()
        except ValueError:
            pass

        # 15 sep 2024 | 15 sep | 1st sep | 11th september 2024 | 2nd jan | 3rd march
        # m = re.search(r'(\d{1,2})\s+([a-z]{3,9})(?:\s+(\d{4}))?', text)
        m = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})(?:\s+(\d{4}))?', text)
        if m:
            try:
                d  = int(m.group(1))
                mth = DateParser.MONTH_MAP.get(m.group(2)) or DateParser.MONTH_MAP.get(m.group(2)[:3])
                y  = int(m.group(3)) if m.group(3) else resolve_year(mth)
                if mth:
                    return date(y, mth, d)
            except Exception:
                pass

        # sep 15 | september 1st | jan 2nd, 2024 | march 11th | april 3rd, 2025
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
            r'(from|for|after|since)\s+'
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
            if original_day is None and not is_numeric:
                last_day = calendar.monthrange(dt.year, dt.month)[1]
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
        parts = re.split(r'\b(?:till|upto|up to)\b', q, maxsplit=1)
        print(parts)
        if len(parts) != 2:
            return None
        left_part, right_part = parts[0].strip(), parts[1].strip()
        if DateParser.extract_date_tokens(left_part):
            return None
        end_date = DateParser.parse_flexible_date(right_part)
        print(end_date,'999999999999999')
        if not end_date:
            return None
        print(left_part,right_part)
        fy_start_year = end_date.year if end_date.month >= 4 else end_date.year - 1
        start_date    = date(fy_start_year, 4, 1)
        if start_date > end_date:
            return None
        return {
            "type":       QueryType.DATE_RANGE,
            "start_date": DateParser.date_to_yyyymmdd(start_date),
            "end_date":   DateParser.date_to_yyyymmdd(end_date),
            "label":      f"FY {fy_start_year} till {end_date.strftime('%d %b %Y')}",
        }

    @staticmethod
    def parse_fy_till_month(q: str):
        if not q or not isinstance(q, str):
            return None
        q = q.lower().strip()
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
        last_day   = calendar.monthrange(year, month_num)[1]
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
    def parse_specific_date_or_range(q: str):
        if not q or not isinstance(q, str):
            return None
        q = q.lower().strip()
        today = date.today()

        if re.search(r'\b([a-z]+)\s+\d{4}\b', q):
            if not re.search(r'\b([1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\b', q):
                return None
        print("parse_fy_till_date, jdhjhj")
        fy_till = DateParser.parse_fy_till_date(q)
        if fy_till:
            return fy_till
        
        print("skip the fy_till")

        # range_match = re.search(r'(.+?)\s+(and|to|until|till|through|–|-)\s+(.+)', q)
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
            print("enter the range match")
            raw_start = range_match.group(1).strip() if range_match else between_match.group(2).strip()
            raw_end   = range_match.group(3).strip() if range_match else between_match.group(3).strip()
            print(raw_start,raw_end)
            start_tokens = DateParser.extract_date_tokens(raw_start)
            print(start_tokens,"[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]")
            end_tokens   = DateParser.extract_date_tokens(raw_end)
            print("end_token",end_tokens)
            start_text = start_tokens[0] if start_tokens else raw_start
            end_text   = end_tokens[0]   if end_tokens   else raw_end
            end_date   = DateParser.parse_flexible_date(end_text)
            print(end_date,"88888888888888888888")
            if not end_date:
                return None
            start_date = DateParser.parse_flexible_date(start_text, default_year=end_date.year)
            print(start_date,end_date)

            if start_date and start_date > end_date:
                start_date = DateParser.parse_flexible_date(start_text, default_year=end_date.year - 1)
            if not start_date:
                day_match = re.search(r'\b([1-9]|[12][0-9]|3[01])\b', start_text)
                if day_match:
                    start_date = date(end_date.year, end_date.month, int(day_match.group(1)))
            if not start_date or start_date > end_date:
                return None
            return {
                "type":       QueryType.DATE_RANGE,
                "start_date": DateParser.date_to_yyyymmdd(start_date),
                "end_date":   DateParser.date_to_yyyymmdd(end_date),
                "label":      f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}",
            }

        if any(k in q for k in ["after", "since"]):
            return None

        tokens = DateParser.extract_date_tokens(q)
        if not tokens:
            return None
        parsed_date = DateParser.parse_flexible_date(tokens[0])
        if not parsed_date:
            return None
        return {
            "type":  QueryType.SPECIFIC_DATE,
            "date":  DateParser.date_to_yyyymmdd(parsed_date),
            "label": parsed_date.strftime("%d %B %Y"),
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
            target_fy = int(year_match)

        if re.search(r"\b(last|previous)\s+(year|fy)\b", q):
            target_fy = current_fy - 1

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


# ── Month-range helper ────────────────────────────────────────────────────────
# def parse_month_range_logic(q: str):
#     if not q:
#         return None

#     q = q.lower().strip()
#     today = datetime.today()
#     current_fy = today.year if today.month >= 4 else today.year - 1

#     MONTH_MAP = DateParser.MONTH_MAP   # assuming this exists

#     month_pattern = (
#         r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
#         r"january|february|march|april|june|july|august|september|october|november|december)"
#     )

#     list_pattern = re.compile(
#         rf"\b{month_pattern}\b(?:\s*[,and]+\s*\b{month_pattern}\b)+",
#         re.IGNORECASE
#     )

#     if list_pattern.search(q):
#         # Optional: You can log or handle multi-month selection differently later
#         print(f"Blocked multi-month list: {q}")
#         return None

#     # Improved flexible regex
#     month_range_match = re.search(
#         r"(?:from|between)?\s*" +
#         rf"(?P<start>{month_pattern})(?:\s+(?P<start_year>20\d{{2}}))?" +
#         r"\s*(?:to|–|-|till|until|and)\s*" +
#         rf"(?P<end>{month_pattern})(?:\s+(?P<end_year>20\d{{2}}))?",
#         q,
#         re.IGNORECASE
#     )

#     if not month_range_match:
#         return None

#     start_m_txt = month_range_match.group("start").lower()
#     end_m_txt = month_range_match.group("end").lower()

#     start_month = MONTH_MAP[start_m_txt]
#     end_month = MONTH_MAP[end_m_txt]

#     # Extract explicit years
#     start_year_str = month_range_match.group("start_year")
#     end_year_str = month_range_match.group("end_year")

#     is_last_year = bool(re.search(r"\b(last year|previous year)\b", q))

#     # ====================== YEAR LOGIC ======================
#     if start_year_str or end_year_str:
#         # At least one year is mentioned → use explicit years
#         start_year = int(start_year_str) if start_year_str else None
#         end_year = int(end_year_str) if end_year_str else None

#         if start_year and not end_year:
#             end_year = start_year
#             if start_month > end_month:          # e.g., apr to jan 2024
#                 end_year += 1

#         elif end_year and not start_year:
#             start_year = end_year
#             if start_month > end_month:          # rare, but safe
#                 start_year -= 1

#         # Both years present (most accurate)
#         elif start_year and end_year:
#             pass  # use as is (handles apr 2023 to may 2025 perfectly)

#     else:
#         # No explicit year → fall back to Financial Year logic
#         base_fy = current_fy - 1 if is_last_year else current_fy
#         start_year = base_fy
#         end_year = base_fy

#         if start_month >= 4:
#             if end_month >= 4:
#                 end_year = base_fy
#             else:
#                 end_year = base_fy + 1
#         else:
#             start_year = base_fy + 1
#             end_year = base_fy + 1 if end_month < 4 else base_fy

#     # ====================== BUILD DATES ======================
#     if start_year is None or end_year is None:
#         return None
#     start_date = datetime(start_year, start_month, 1)
#     _, last_day = monthrange(end_year, end_month)
#     end_date = datetime(end_year, end_month, last_day)

#     # ====================== MONTH-WISE ======================
#     if re.search(r"\b(month\s*wise|monthly|mom|month\s*on\s*month|by\s*month)\b", q):
#         periods = []
#         m, y = start_month, start_year
#         while (y < end_year) or (y == end_year and m <= end_month):
#             _, ld = monthrange(y, m)
#             s = datetime(y, m, 1)
#             e = datetime(y, m, ld)
#             periods.append({
#                 "label": s.strftime("%b %Y"),
#                 "start_date": s.strftime("%Y%m%d"),
#                 "end_date": e.strftime("%Y%m%d"),
#             })
#             m += 1
#             if m == 13:
#                 m = 1
#                 y += 1

#         return {
#             "type": QueryType.MONTH_RANGE_MONTH_WISE,
#             "periods": periods,
#             "label": f"{start_m_txt.title()}–{end_m_txt.title()} Month-wise",
#         }

#     # ====================== NORMAL RANGE ======================
#     return {
#         "type": QueryType.MONTH_RANGE,
#         "start_date": start_date.strftime("%Y%m%d"),
#         "end_date": end_date.strftime("%Y%m%d"),
#         "label": f"{start_m_txt.title()} to {end_m_txt.title()}",
#     }


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
    is_last_year = detect_fy(q, current_fy)

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
    if not q:
        return None
    q = q.lower().strip()
    if re.search(
        r"\b([1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\s+"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
        r"|january|february|march|april|june|july|august|september|october|november|december)\b",
        q
    ):
        return None
    
    if re.search(r"\b(after|till|to|upto|up to|on)\b", q):
        return None

    months_found = []
    current_fy   = DateParser.get_current_fy()
    explicit_year = extract_fy(q)
    # explicit_year = int(year_match.group(1)) if year_match else None
    for name, num in DateParser.MONTH_MAP.items():
        if re.search(rf"\b{name}\b", q):
            months_found.append(num)
    months_found = sorted(set(months_found))
    if not months_found:
        return None

    target_fy = detect_fy(q, current_fy)

    periods = []
    for mm in months_found:
        if explicit_year:
            # ✅ Use calendar year directly
            year = explicit_year
        else:
            # ✅ Use FY logic only if no year is given
            year = target_fy if mm >= 4 else target_fy + 1

        _, ld = monthrange(year, mm)
        s = datetime(year, mm, 1)
        e = datetime(year, mm, ld)
        periods.append({
            "label":      s.strftime("%b %Y"),
            "start_date": DateParser.date_to_yyyymmdd(s.date()),
            "end_date":   DateParser.date_to_yyyymmdd(e.date()),
        })
    return {
        "type":    QueryType.MONTH_MULTI_MONTH_WISE,
        "periods": periods,
        "label":   " & ".join([datetime(2000, mm, 1).strftime("%b") for mm in months_found]) + f" FY{target_fy}",
    }

def parse_quarter_mom(q: str, today: datetime = None):
    if today is None:
        today = datetime.today()
    q = q.lower()
    if not any(k in q for k in ["mom", "month on month", "month wise", "month"]):
        return None

    fy = DateParser.get_current_fy(today)
    current_q = DateParser.get_fy_quarter(today.month)

    quarters = []
    if any(k for k in ["last quarter","previous quarter","last qtr","previous qtr"] if k in q):
        if current_q == 1:
            q_num = 4; year = fy - 1
        else:
            q_num = current_q - 1; year = fy
        quarters = [f"q{q_num}"]
    else:
        q_pattern = r"(?:q(?:uarter|tr)?\s*([1-4]))"
        found_qs  = re.findall(rf"\b{q_pattern}\b", q)
        quarters  = [f"q{num}" for num in found_qs]
        if not quarters:
            return None
        explicit_year = extract_fy(q)
        year = explicit_year if explicit_year else fy

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


def detect_year_and(q: str):
    if not q: return None
    if re.search(
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+[a-z]{3,9}\s+20\d{2}\b',
            q
        ):
        return None
    year_matches = re.findall(r'\b(20\d{2})\b', q)
    if not year_matches or len(year_matches) < 2: return None
    years   = sorted(set(int(y) for y in year_matches))
    periods = []
    for fy in years:
        s, e = DateParser.get_fy_start_end(fy)
        periods.append({"label": f"FY{fy}", "start_date": s, "end_date": e})
    return {"type": QueryType.YEAR_WISE, "years": years, "periods": periods,
            "label": " & ".join([f"FY{y}" for y in years])}


def year_range_logic(q: str):
    if re.search(r'\b\d{1,2}\s+[a-z]{3,9}\s+20\d{2}\b', q): return None
    m = re.search(r'\b(20\d{2})\s*(?:to|till|–|-)\s*(20\d{2})\b', q)
    if m:
        sy, ey = int(m.group(1)), int(m.group(2))
        periods = []
        for fy in range(sy, ey + 1):
            s, e = DateParser.get_fy_start_end(fy)
            periods.append({"year": f"FY{fy}", "start_date": s, "end_date": e})
        return {"type": QueryType.YEAR_WISE, "years": list(range(sy, ey+1)),
                "periods": periods, "label": f"FY{sy} to FY{ey}"}
    return None


# ============================================================================
# LLM INTENT PROMPT  — JSON extraction only, no SQL
# ============================================================================
def build_task_llm_prompt(question: str) -> str:
    schema_lines = "\n".join([
        f"  - {col} ({meta['type']}): {meta['description']}"
        for col, meta in TaskColumnMetadata.COLUMNS.items()
    ])

    return f"""
You are a strict JSON extraction engine for a task management system.
Your ONLY job is to extract structured intent from a natural language query about tasks.
You must NEVER infer, guess, or hallucinate values. Only extract what is explicitly stated.
You must NEVER generate SQL. Only return JSON.

=============================================================
TABLE CONTEXT
=============================================================
Table: task_report
Columns:
{schema_lines}

Date columns: created_date_c, lastmodified_date_c
All date columns are stored as YYYYMMDD INTEGER — do NOT mention date_parse.
Default date column: created_date_c

=============================================================
ABSOLUTE GROUND RULES
=============================================================
1. Return ONLY valid JSON wrapped in <JSON_RESPONSE> tags. No explanation, no markdown, no SQL.
2. NEVER include date columns in "filters". Dates go ONLY in date_hint + date_column.
3. NEVER add a column to "group_by" unless user explicitly asked to group/split/break by it.
4. NEVER infer date_hint unless user explicitly mentioned a time period → null otherwise.
5. aggregation MUST always be a LIST.
6. Omit any field from filters if no value to extract.

=============================================================
OUTPUT SCHEMA
=============================================================
{{
  "aggregation": [ <string> ],            // always a list; default ["task_count"]
  "group_by":    [ <column_name> ],       // only explicitly requested groupings
  "filters":     {{ <column>: <value> }}, // only explicitly mentioned filter values
  "date_hint":   <string | null>,         // raw user phrase or null
  "date_column": "created_date_c" | "lastmodified_date_c"
}}

=============================================================
SECTION 1 — AGGREGATION
=============================================================
| User says                                          | Token        |
|----------------------------------------------------|--------------|
| total task / task count / how many tasks / count   | "task_count" |
| tasks                                              | "task_count" |

Default if unclear → ["task_count"]

=============================================================
SECTION 2 — FILTER COLUMN MAPPING
=============================================================

2A. TASK SUBJECT (subject_c)
Triggers: followup, follow up, sales follow up, re-query, tried calling,
          click to call, call back by sales expert, experience calling follow up,
          welcome calling follow up, request call back, live chat
- "follow up" / "followup" / "follow-up" / "followup task" / "follow up task"
  → subject_c: "__followup__"
  (Python will expand to: IN ('Follow Up','Sales Follow Up','Experience Calling Follow Up','Welcome Calling Follow Up'))
- "sales follow up" / "sales followup"   → subject_c: "Sales Follow Up"
- "re query" / "requery" / "re-query"   → subject_c: "Re-query requested"
- "tried calling"                        → subject_c: "__tried_calling__"
- "click to call" / "click call"         → subject_c: "__click_to_call__"
- "call back by sales expert"            → subject_c: "Call Back By Sales Expert"
- "experience calling follow up"         → subject_c: "Experience Calling Follow Up"
- "welcome calling follow up"            → subject_c: "Welcome Calling Follow Up"
- "request call back" / "request callback" → subject_c: "Request call back"
- "live chat" / "live chat query"        → subject_c: "Live Chat Query"

- If user says "subject <free text>" (example: "subject phone call activity"),
  use that exact free text as subject_c. Do not map it to follow up, and do not
  add subject_c to group_by unless the user also says "by subject" or "subject wise".

2B. TASK STATUS (status_c)
Triggers: completed, open, in progress, cancelled, canceled, deferred, closed
- "completed task" / "completed"         → status_c: "Completed"
- "open task" / "open"                   → status_c: "Open"
- "in progress"                          → status_c: "In Progress"
- "cancelled" / "canceled" / "cancel"   → status_c: "__cancelled__"
  (Python will expand to: IN ('Cancelled','Canceled','Cancel'))
- "deferred"                             → status_c: "Deferred"
- "closed"                               → status_c: "Closed"

2C. CUSTOMER FEEDBACK (customer_feedback_c)
Triggers: discussion pending, pending, interested, not interested, junk, open task
Precedence rules:
  1. If user says "feedback"/"customer feedback"/"discussion pending" → customer_feedback_c
  2. If user says "status"/"task status"/"open"/"completed" → status_c
  3. If user says just "pending" (no status keyword) → customer_feedback_c: "Discussion Pending"
- "discussion pending" / "pending" / "open task" → customer_feedback_c: "Discussion Pending"
- "interested"                           → customer_feedback_c: "Interested"
- "not interested" / "notinterested"    → customer_feedback_c: "Not Interested"
- "junk"                                 → customer_feedback_c: "Junk"

2D. SALES TEAM FEEDBACK (sales_team_feedback_c)
Triggers: qualified, disqualified, in followup
- "qualified"    → sales_team_feedback_c: "Qualified"
- "disqualified" → sales_team_feedback_c: "Disqualified"
- "in followup"  → sales_team_feedback_c: "In Followup"

2E. PRODUCT CATEGORY (product_category_c)
Trigger: user mentions any product name like veridia, dream homes, eligo, eden,
         new plots, old plots, wave floor, prime floors, armonia villa, etc.
OR user says: "product", "product wise", "by product", "category wise"
→ product_category_c (LIKE matching, never =)

2F. PROJECT (project_c)
Trigger: "wave city", "wmcc", "wave estate", "by project", "project wise",
         "project called", "project name"
→ project_c (LIKE matching, never =)
IMPORTANT: NEVER use project_c for product category values.

2G. OWNER NAME (ownername_c)
Trigger: "for [name]", "by [name]", "[name]'s tasks", specific person name
→ ownername_c: the name value


2H. DATE COLUMN SELECTION
- Default: "created_date_c"
- Use "lastmodified_date_c" if user says "modified", "updated", "last modified"

=============================================================
SECTION 3 — GROUP_BY RULES
=============================================================
Only include a column in group_by if user explicitly says:
"by <column>", "group by <column>", "wise", "user wise", "owner wise",
"product wise", "category wise", "project wise", "status wise",
"subject wise", "feedback wise", etc.
NEVER infer group_by from filters.

=============================================================
SECTION 4 — DATE HINT
=============================================================
Return the raw user phrase. Examples:
"today", "this week", "this month", "last month", "last quarter",
"this quarter", "last year", "last 3 months", "last 6 months",
"q1", "q2 month wise", "quarter wise", "month wise", "mom",
"year wise", "yoy", "fy 2024", "april 2025", "jan to march",
"2024 to 2025", null

=============================================================
SECTION 5 — WORKED EXAMPLES
=============================================================

Q: "Total tasks this month"
A: {{"aggregation":["task_count"],"group_by":[],"filters":{{}},"date_hint":"this month","date_column":"created_date_c"}}

Q: "Follow up tasks last quarter"
A: {{"aggregation":["task_count"],"group_by":[],"filters":{{"subject_c":"__followup__"}},"date_hint":"last quarter","date_column":"created_date_c"}}

Q: "Completed tasks by owner this year"
A: {{"aggregation":["task_count"],"group_by":["ownername_c"],"filters":{{"status_c":"Completed"}},"date_hint":"this year","date_column":"created_date_c"}}

Q: "Product wise tasks for veridia last month"
A: {{"aggregation":["task_count"],"group_by":["product_category_c"],"filters":{{"product_category_c":"veridia"}},"date_hint":"last month","date_column":"created_date_c"}}

Q: "User wise cancelled tasks month wise"
A: {{"aggregation":["task_count"],"group_by":["ownername_c"],"filters":{{"status_c":"__cancelled__"}},"date_hint":"month wise","date_column":"created_date_c"}}

Q: "Total qualified tasks year wise"
A: {{"aggregation":["task_count"],"group_by":[],"filters":{{"sales_team_feedback_c":"Qualified"}},"date_hint":"year wise","date_column":"created_date_c"}}

Q: "Tried calling tasks for wave city q1"
A: {{"aggregation":["task_count"],"group_by":[],"filters":{{"subject_c":"__tried_calling__","project_c":"wave city"}},"date_hint":"q1","date_column":"created_date_c"}}

Q: "Quarter on quarter total tasks for subject phone call activity for the last 3 years"
A: {{"aggregation":["task_count"],"group_by":[],"filters":{{"subject_c":"phone call activity"}},"date_hint":"quarter on quarter for the last 3 years","date_column":"created_date_c"}}

Q: "Tasks from eden product wise quarter on quarter"
A: {{"aggregation":["task_count"],"group_by":["product_category_c"],"filters":{{"product_category_c":"eden"}},"date_hint":"quarter on quarter","date_column":"created_date_c"}}

Q: "Discussion pending tasks for ashwariya"
A: {{"aggregation":["task_count"],"group_by":[],"filters":{{"customer_feedback_c":"Discussion Pending","ownername_c":"ashwariya"}},"date_hint":null,"date_column":"created_date_c"}}

Q: "Total tasks"
A: {{"aggregation":["task_count"],"group_by":[],"filters":{{}},"date_hint":null,"date_column":"created_date_c"}}

=============================================================
NOW PROCESS THE FOLLOWING QUERY
=============================================================
User Query: "{question}"

Return ONLY the JSON wrapped in <JSON_RESPONSE> tags. No other text.

<JSON_RESPONSE>
"""


# ============================================================================
# LLM INTENT DETECTOR
# ============================================================================
class LLMIntentDetector:
    def extract_intent(self, question: str) -> Dict[str, Any]:
        prompt = build_task_llm_prompt(question)
        try:
            response = llm_model.generate_text(prompt)
            logger.info(f"Raw LLM response: {response}")
            json_part = self._extract_first_json(response)
            intent    = json.loads(json_part)
            logger.info(f"Parsed LLM intent: {intent}")
            return intent
        except Exception as e:
            logger.error(f"LLM intent parsing failed: {e}")
        return {"aggregation": ["task_count"], "group_by": [], "filters": {},
                "date_hint": None, "date_column": "created_date_c"}

    @staticmethod
    def _extract_first_json(text: str) -> str:
        tag = re.search(r"<JSON_RESPONSE>(.*?)</JSON_RESPONSE>", text, re.DOTALL)
        if tag:
            c = tag.group(1).strip()
            c = re.sub(r"^\s*```json\s*", "", c)
            c = re.sub(r"```\s*$", "", c)
            return c
        s, e = text.find('{'), text.rfind('}')
        if s != -1 and e != -1 and e > s:
            pj = text[s:e+1].replace('\\{', '{').replace('\\}', '}')
            bc = 0
            for i, ch in enumerate(pj):
                if ch == '{':  bc += 1
                elif ch == '}':
                    bc -= 1
                    if bc == 0:
                        return pj[:i+1]
        raise ValueError("No valid JSON in LLM response")


def yoy_logic(q):
    today      = datetime.today()
    current_fy = DateParser.get_current_fy()
    start_fy = 2020   # your DB start year
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

def last_n_mom_logic(q):
    last_n_month_match    = re.search(r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b", q)
    is_mom  = any(k in q for k in ["mom","month over month","month-on-month","monthly","month wise","month on month","months wise"])
    today      = datetime.today()
    current_fy = DateParser.get_current_fy()
    try:
        raw_n = last_n_month_match.group(1)
        n     = int(raw_n) if raw_n.isdigit() else DateParser.WORD_TO_NUM.get(raw_n, 1)
        first_of_month = today.date().replace(day=1)
        end_dt  = first_of_month - timedelta(days=1)
        start_dt = end_dt.replace(day=1)
        months = []
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
                periods.append({"label": s.strftime("%b %Y"),
                                "start_date": DateParser.date_to_yyyymmdd(s.date()),
                                "end_date":   DateParser.date_to_yyyymmdd(e.date())})
            return {"type": QueryType.MONTH_WISE, "periods": periods,
                    "label": f"Last {n} Months (MoM)"}
        return {"type": QueryType.LAST_N_MONTHS,
                "start_date": DateParser.date_to_yyyymmdd(start_dt),
                "end_date":   DateParser.date_to_yyyymmdd(end_dt),
                "label": f"Last {n} Months"}
    except Exception:
        pass

def last_n_year_mom_qoq_yoy(q):
    print(f"Processing YoY/QoQ/MoM logic for query: {q}")

    last_n_year_match     = re.search(r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\b", q)
    is_mom  = any(k in q for k in ["mom","month over month","month-on-month","monthly","month wise","month on month","months wise"])
    is_qoq  = any(k in q for k in ["qoq","quarter over quarter","quarter-on-quarter","quarterly","quarter wise","quarter on quarter"])
    is_yoy  = any(k in q for k in ["yoy","year on year","yearly","year wise","by year","annual trend","year over year"])
    today      = datetime.today()
    current_fy = DateParser.get_current_fy()
    raw_n = last_n_year_match.group(1)

    n = int(raw_n) if raw_n.isdigit() else DateParser.WORD_TO_NUM.get(raw_n, 1)

    end_fy = current_fy - 1
    start_fy = end_fy - n + 1

    # =========================
    # ✅ YoY (year-wise)
    # =========================
    if is_yoy:
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
            "label": f"Last {n} Years (YoY)"
        }

    # =========================
    # ✅ QoQ (quarter-wise)
    # =========================
    if is_qoq:
        quarters = []

        for fy in range(start_fy, end_fy + 1):
            for qn in range(1, 5):
                if qn == 1:
                    s, e = datetime(fy,4,1), datetime(fy,6,30)
                elif qn == 2:
                    s, e = datetime(fy,7,1), datetime(fy,9,30)
                elif qn == 3:
                    s, e = datetime(fy,10,1), datetime(fy,12,31)
                else:
                    s, e = datetime(fy+1,1,1), datetime(fy+1,3,31)

                quarters.append({
                    "quarter": f"Q{qn} FY{fy}",
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date": DateParser.date_to_yyyymmdd(e.date())
                })

        return {
            "type": QueryType.QUARTER_WISE,
            "quarters": quarters,
            "label": f"Last {n} Years (QoQ)"
        }

    # =========================
    # ✅ MoM (month-wise)
    # =========================
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
                    "label": s.strftime("%b %Y"),
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date": DateParser.date_to_yyyymmdd(e.date())
                })

        return {
            "type": QueryType.MONTH_WISE,
            "periods": periods,
            "label": f"Last {n} Years (MoM)"
        }

    # =========================
    # ✅ Normal range
    # =========================
    s, _ = DateParser.get_fy_start_end(start_fy)
    _, e = DateParser.get_fy_start_end(end_fy)

    return {
        "type": QueryType.LAST_N_YEARS,
        "start_date": s,
        "end_date": e,
        "label": f"Last {n} Years"
    }

def last_n_quarte_mom_qoq(q):
    last_n_quarter_match  = re.search(r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+quarters?\b", q)
    is_mom  = any(k in q for k in ["mom","month over month","month-on-month","monthly","month wise","month on month","months wise"])
    is_qoq  = any(k in q for k in ["qoq","quarter over quarter","quarter-on-quarter","quarterly","quarter wise","quarter on quarter"])
    today      = datetime.today()
    current_fy = DateParser.get_current_fy()

    try:
        raw_n = last_n_quarter_match.group(1)
        n = max(1, int(raw_n) if raw_n.isdigit() else DateParser.WORD_TO_NUM.get(raw_n, 1))

        # ✅ Year override
        # year_match = re.search(r"\b(20\d{2})\b", q)
        # explicit_year = int(year_match.group(1)) if year_match else None
        explicit_year = extract_fy(q)
        # 👉 Decide base FY
        base_fy = explicit_year if explicit_year else current_fy

        # 👉 Current quarter (based on today OR override year)
        cq = DateParser.get_fy_quarter(today.month)

        eq = cq - 1 if cq > 1 else 4
        efy = base_fy if cq > 1 else base_fy - 1

        # 👉 Collect last N quarters
        meta = []
        qu, fy2 = eq, efy
        for _ in range(n):
            meta.append((fy2, qu))
            qu -= 1
            if qu == 0:
                qu = 4
                fy2 -= 1
        meta.reverse()

        # 👉 Quarter date function
        def qd(fy3, qn):
            if qn == 1: return datetime(fy3,4,1), datetime(fy3,6,30)
            elif qn == 2: return datetime(fy3,7,1), datetime(fy3,9,30)
            elif qn == 3: return datetime(fy3,10,1), datetime(fy3,12,31)
            else: return datetime(fy3+1,1,1), datetime(fy3+1,3,31)

        # =========================
        # ✅ QoQ
        # =========================
        if is_qoq:
            quarters = []
            for fy3, qn in meta:
                s2, e2 = qd(fy3, qn)
                quarters.append({
                    "quarter": f"Q{qn} FY{fy3}",
                    "start_date": DateParser.date_to_yyyymmdd(s2.date()),
                    "end_date": DateParser.date_to_yyyymmdd(e2.date())
                })

            return {
                "type": QueryType.QUARTER_WISE,
                "quarters": quarters,
                "label": f"Last {n} Quarters (QoQ)"
            }

        # =========================
        # ✅ MoM inside quarters 🔥
        # =========================
        if is_mom:
            periods = []

            for fy3, qn in meta:
                s2, e2 = qd(fy3, qn)

                # iterate month-wise inside quarter
                m, y = s2.month, s2.year
                while (y < e2.year) or (y == e2.year and m <= e2.month):
                    _, ld = monthrange(y, m)
                    s = datetime(y, m, 1)
                    e = datetime(y, m, ld)

                    periods.append({
                        "label": s.strftime("%b %Y"),
                        "start_date": DateParser.date_to_yyyymmdd(s.date()),
                        "end_date": DateParser.date_to_yyyymmdd(e.date())
                    })

                    m += 1
                    if m == 13:
                        m = 1
                        y += 1

            return {
                "type": QueryType.MONTH_WISE,
                "periods": periods,
                "label": f"Last {n} Quarters (MoM)"
            }

        # =========================
        # ✅ Normal range
        # =========================
        s2, _ = qd(*meta[0])
        _, e2 = qd(*meta[-1])

        return {
            "type": QueryType.LAST_N_QUARTERS,
            "start_date": DateParser.date_to_yyyymmdd(s2.date()),
            "end_date": DateParser.date_to_yyyymmdd(e2.date()),
            "label": f"Last {n} Quarters"
        }

    except Exception as ex:
        logger.error(f"LAST_N_QUARTERS error: {ex}")


def mom_logic(q):
    today      = datetime.today()
    current_fy = DateParser.get_current_fy()

    # target_fy = current_fy
    # if "last year" in q or "previous year" in q:
    #     target_fy = current_fy - 1
    # else:
    #     for word in q.split():
    #         if word.isdigit() and len(word) == 4:
    #             target_fy = int(word); break

    target_fy = detect_fy(q, current_fy)

    periods = []
    for month in range(4, 16):
        m  = month if month <= 12 else month - 12
        y  = target_fy if month <= 12 else target_fy + 1
        _, ld = monthrange(y, m)
        s  = datetime(y, m, 1)
        e  = datetime(y, m, ld)
        periods.append({"label": s.strftime("%b %Y"),
                        "start_date": DateParser.date_to_yyyymmdd(s.date()),
                        "end_date":   DateParser.date_to_yyyymmdd(e.date())})
    return {"type": QueryType.MONTH_WISE, "fy": target_fy, "periods": periods,
            "label": f"Month-wise FY{target_fy}"}


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

# ============================================================================
# TASK INTENT DETECTOR — normalize + deterministic extraction
# ============================================================================
class TaskIntentDetector:
    def __init__(self):
        self.keywords     = TaskColumnMetadata.KEYWORD_MAPPING
        self.valid_values = {
            "status_c":             TaskColumnMetadata.VALID_STATUS,
            "customer_feedback_c": ["Discussion Pending","Interested","Not Interested","Junk"],
            "sales_team_feedback_c":["Qualified","Disqualified","In Followup"]
        }

    @staticmethod
    def _dedupe_keep_order(values: List[Any]) -> List[Any]:
        seen = set()
        ordered = []
        for value in values:
            key = str(value).lower().strip() if isinstance(value, str) else value
            if key in seen:
                continue
            seen.add(key)
            ordered.append(value)
        return ordered

    def _extract_status_values(self, raw_value: Any) -> List[str]:
        value = str(raw_value).lower().strip()
        normalized_value = re.sub(r'[_\W]+', ' ', value)
        matches = []

        for status in self.valid_values["status_c"]:
            match = re.search(rf"\b{re.escape(status.lower())}\b", normalized_value)
            if match:
                matches.append((match.start(), status))

        matches.sort(key=lambda item: item[0])
        return self._dedupe_keep_order([status for _, status in matches])

    @staticmethod
    def _extract_explicit_subject_value(question: str) -> Optional[str]:
        q = question.lower()
        match = re.search(
            r"\bsubject\s+(?:is\s+|equals\s+|called\s+|named\s+)?([a-z0-9][a-z0-9\s/_-]*)",
            q,
        )
        if not match:
            return None

        value = match.group(1)
        value = re.split(
            r"\s+\b(?:for|from|during|between|last|this|quarter|month|year|qoq|mom|yoy|by|group by|wise)\b",
            value,
            maxsplit=1,
        )[0]
        value = re.sub(r"\s+", " ", value).strip(" -_/")
        if not value or value in {"wise", "group", "by"}:
            return None
        return value

    # ── Filter normalisation ──────────────────────────────────────────────────
    def normalize_filters(self, raw_filters: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        col_map    = {k.lower(): k for k in TaskColumnMetadata.COLUMNS.keys()}

        for col, values in raw_filters.items():
            if not values:
                continue
            target_col = col_map.get(col.lower(), col)

            # Redirect product values to product_category_c
            v_str = " ".join(str(v) for v in (values if isinstance(values, list) else [values])).lower()
            if target_col not in ("product_category_c", "project_c"):
                for cat in TaskColumnMetadata.PRODUCT_CATEGORIES:
                    if cat in v_str:
                        target_col = "product_category_c"
                        break
            if target_col not in ("project_c", "product_category_c"):
                for proj in TaskColumnMetadata.PROJECT_VALUES:
                    if proj in v_str:
                        target_col = "project_c"
                        break

            normalized_values = []
            value_list = values if isinstance(values, list) else [values]
            expanded   = []
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

                # Special tokens — keep as-is for SQL builder
                if val_str in ("__followup__", "__tried_calling__", "__click_to_call__", "__cancelled__"):
                    normalized.setdefault(target_col, [])
                    normalized[target_col].append(val_str)
                    continue

                # Keyword mapping
                mapped = False
                if val_str in self.keywords:
                    m = self.keywords[val_str]
                    if m.get("column") and m.get("value"):
                        normalized.setdefault(m["column"], [])
                        normalized[m["column"]].append(m["value"])
                        mapped = True
                if mapped:
                    continue

                if target_col == "status_c":
                    matched_statuses = self._extract_status_values(val)
                    if matched_statuses:
                        normalized_values.extend(matched_statuses)
                        continue

                # Valid values exact/partial
                if target_col in self.valid_values:
                    found = False
                    for poss in self.valid_values[target_col]:
                        if val_str == poss.lower():
                            normalized_values.append(poss); found = True; break
                    if not found and len(val_str) > 3:
                        for poss in self.valid_values[target_col]:
                            if val_str in poss.lower() or poss.lower() in val_str:
                                normalized_values.append(poss); found = True; break
                    if found:
                        continue

                normalized_values.append(val)

            if normalized_values:
                normalized.setdefault(target_col, [])
                normalized[target_col].extend(normalized_values)
                normalized[target_col] = self._dedupe_keep_order(normalized[target_col])

        return normalized

    # ── Full intent normalisation ─────────────────────────────────────────────
    def normalize_intent(self, raw_intent: Dict[str, Any], question: str) -> Dict[str, Any]:
        # Aggregation always task_count
        normalized_agg = ["task_count"]

        # Filters
        raw_filters           = raw_intent.get("filters", {})
        normalized_filters    = self.normalize_filters(raw_filters)
        det_filters           = self.extract_filters(question)
        explicit_subject      = self._extract_explicit_subject_value(question)
        for col, vals in det_filters.items():
            if col not in normalized_filters:
                normalized_filters[col] = vals
            else:
                normalized_filters[col] = self._dedupe_keep_order(normalized_filters[col] + vals)

        if explicit_subject:
            normalized_filters["subject_c"] = [explicit_subject]

        if "ownername_c" in raw_filters and "ownername_c" in normalized_filters:
            # Convert both values to lists if they are strings
            raw_owner = raw_filters["ownername_c"]
            norm_owner = normalized_filters["ownername_c"]

            if not isinstance(raw_owner, list):
                raw_owner = [raw_owner]

            if not isinstance(norm_owner, list):
                norm_owner = [norm_owner]

            # Merge and remove duplicates
            normalized_filters["ownername_c"] = self._dedupe_keep_order(norm_owner + raw_owner)
        else:
            # Remove ownername_c if it is present in only one of them
            normalized_filters.pop("ownername_c", None)


        # Group-by
        normalized_groupby = self.extract_groupby(question)
        if not normalized_groupby:
            raw_gb  = raw_intent.get("group_by", [])
            if isinstance(raw_gb, str):
                raw_gb = [raw_gb]
            col_map  = {k.lower(): k for k in TaskColumnMetadata.COLUMNS.keys()}
            date_cols = set(TaskColumnMetadata.DATE_COLUMNS)
            for g in raw_gb:
                g_l = str(g).lower().strip()
                fc  = self.keywords.get(g_l, {}).get("column") or col_map.get(g_l)
                if explicit_subject and fc == "subject_c":
                    continue
                if fc == "subject_c" and "subject_c" in normalized_filters:
                    continue
                if fc and fc.lower() not in date_cols:
                    normalized_groupby.append(fc)

        return {
            "aggregation": normalized_agg,
            "filters":     normalized_filters,
            "group_by":    self._dedupe_keep_order(normalized_groupby),
            "date_hint":   raw_intent.get("date_hint"),
            "date_column": raw_intent.get("date_column", "created_date_c"),
        }

    # ── Deterministic filter extraction ──────────────────────────────────────
    def extract_filters(self, question: str) -> Dict[str, Any]:
        filters = {}
        q       = question.lower()
        matches_by_col = {}

        for keyword, mapping in self.keywords.items():
            if "column" in mapping and "value" in mapping:
                if re.search(rf"\b{re.escape(keyword)}\b", q):
                    col = mapping["column"]
                    val = mapping["value"]
                    matches_by_col.setdefault(col, []).append((keyword, val))

        for col, matches in matches_by_col.items():
            if col == "subject_c":
                max_len = max(len(keyword) for keyword, _ in matches)
                matches = [(keyword, val) for keyword, val in matches if len(keyword) == max_len]
            for _, val in matches:
                filters.setdefault(col, []).append(val)

        # Owner name
        name_m = re.search(
            r'\b(?:for|by|of|task for|tasks for|tasks by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            question
        )
        if name_m:
            filters.setdefault("ownername_c", []).append(name_m.group(1))

        # Possessive: "Ram's tasks"
        poss_m = re.search(r"\b([A-Z][a-z]+)'s\s+tasks?\b", question)
        if poss_m:
            filters.setdefault("ownername_c", []).append(poss_m.group(1))

        return filters

    # ── Deterministic group-by extraction ────────────────────────────────────
    def extract_groupby(self, question: str) -> List[str]:
        q        = question.lower()
        group_by = []
        patterns = [
            (r'\b(by project|project wise|by project)\b',            "project_c"),
            (r'\b(by product|product wise|category wise|all product)\b', "product_category_c"),
            (r'\b(by status|status wise|by status)\b',              "status_c"),
            (r'\b(by subject|subject wise|by subject)\b',            "subject_c"),
            (r'\b(by owner|owner wise|user wise)\b',      "ownername_c"),
            (r'\b(by feedback|feedback wise)\b',          "customer_feedback_c"),
            (r'\b(by sales feedback|sales feedback wise)\b', "sales_team_feedback_c"),
        ]
        for pattern, col in patterns:
            if re.search(pattern, q):
                group_by.append(col)
        return group_by

    # ── Date intent detection — full cascade identical to inventory/lead ───────
    def detect_date_intent(self, question: str):
        q = question.lower()
        logger.info(f"Task date intent detection: {q}")

        NUMBER_WORDS = {
            "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
            "seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,
            "thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
            "seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,
        }

        last_n_year_match = re.search(
            r"\b(last|previous|past)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\b",
            q
        )
        last_n_month_match = re.search(
            r"\b(last|previous|past)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b",
            q
        )
        last_n_quarter_match = re.search(
            r"\b(last|previous|past)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+quarters?\b",
            q
        )
        is_mom  = any(k in q for k in ["mom","month over month","month-on-month","monthly","month wise","month on month","months wise","per month","by month"])
        is_qoq  = any(k in q for k in ["qoq","quarter over quarter","quarter-on-quarter","quarterly","quarter wise","quarter on quarter","compare quarters"])
        is_yoy  = any(k in q for k in ["yoy","year on year","yearly","year wise","by year","annual trend","year over year","compare years","per year"])
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', q))
        has_date_keyword = (
            any(w in q for w in ["day","month","quarter","year","date","last","this","current",
                                   "fy","q1","q2","q3","q4","qtr","mom","qoq","yoy","today",
                                   "yesterday","week","fiscal","previous"])
            or any(m in q for m in DateParser.MONTH_MAP.keys())
            or has_year
        )
        if not has_date_keyword:
            return None

        today      = datetime.today()
        current_fy = DateParser.get_current_fy()

        # 1️⃣ Quarter + MoM
        qmom = parse_quarter_mom(q)
        if qmom:
            return qmom

        # 2️⃣ QoQ (quarter wise)
        if is_qoq and not last_n_quarter_match and not last_n_year_match and not last_n_month_match:
            # target_fy = current_fy
            # if "last year" in q or "previous year" in q:
            #     target_fy = current_fy - 1
            # else:
            #     for w in q.split():
            #         if w.isdigit() and len(w) == 4: target_fy = int(w); break
            target_fy = detect_fy(q, current_fy)
            quarters = []
            for qn in range(1,5):
                if qn==1:   s,e = datetime(target_fy,4,1),   datetime(target_fy,6,30)
                elif qn==2: s,e = datetime(target_fy,7,1),   datetime(target_fy,9,30)
                elif qn==3: s,e = datetime(target_fy,10,1),  datetime(target_fy,12,31)
                else:       s,e = datetime(target_fy+1,1,1), datetime(target_fy+1,3,31)
                quarters.append({"quarter": f"Q{qn} FY{target_fy}",
                                  "start_date": DateParser.date_to_yyyymmdd(s.date()),
                                  "end_date":   DateParser.date_to_yyyymmdd(e.date())})
            return {"type": QueryType.QUARTER_WISE, "fy": target_fy, "quarters": quarters,
                    "label": f"Quarter-wise FY{target_fy}"}

        # 3️⃣ Specific quarters (Q1, Q2, Q1 to Q3, Q1 and Q4)
        qi = DateParser.parse_quarter_intent(q)
        if qi:
            return qi

        # 4️⃣ Month range (jan to march)
        ml = parse_month_range_logic(q)
        if ml:
            return ml

        # 5️⃣ This quarter vs last quarter
        if "this quarter vs last quarter" in q or "this quarter compared to last quarter" in q:
            tq  = DateParser.get_fy_quarter(today.month)
            lq  = tq-1 if tq>1 else 4; lqfy = current_fy if tq>1 else current_fy-1
            def gqd(qn,fy):
                if qn==1:   return datetime(fy,4,1),   datetime(fy,6,30)
                elif qn==2: return datetime(fy,7,1),   datetime(fy,9,30)
                elif qn==3: return datetime(fy,10,1),  datetime(fy,12,31)
                else:       return datetime(fy+1,1,1), datetime(fy+1,3,31)
            ts,te = gqd(tq,current_fy); ls,le = gqd(lq,lqfy)
            return {"type": QueryType.MULTI_DATE_RANGE, "ranges": [
                {"start_date": DateParser.date_to_yyyymmdd(ts.date()), "end_date": DateParser.date_to_yyyymmdd(te.date()), "label": f"This Quarter Q{tq}"},
                {"start_date": DateParser.date_to_yyyymmdd(ls.date()), "end_date": DateParser.date_to_yyyymmdd(le.date()), "label": f"Last Quarter Q{lq}"},
            ], "label": "This Quarter vs Last Quarter"}

        # 6️⃣ Discrete months
        dm = discrete_month(q)
        if dm:
            return dm

        # 7️⃣ Today
        if any(k in q for k in ["today","today's","todays"]):
            return {"type": QueryType.SPECIFIC_DATE, "date": DateParser.today_yyyymmdd(),
                    "label": today.strftime("%d %B %Y")}

        # 8️⃣ Yesterday
        if "yesterday" in q or "last day" in q or "previous day" in q or "last date" in q:
            yd = today.date() - timedelta(days=1)
            return {"type": QueryType.SPECIFIC_DATE, "date": DateParser.date_to_yyyymmdd(yd),
                    "label": f"Yesterday {yd}"}

        # 9️⃣ This week
        if "this week" in q or "current week" in q:
            sow = today.date() - timedelta(days=today.weekday())
            return {"type": QueryType.THIS_WEEK,
                    "start_date": DateParser.date_to_yyyymmdd(sow),
                    "end_date":   DateParser.today_yyyymmdd(), "label": "This Week"}

        # 🔟 Last N days
        if any(k for k in ["last","previous","past","yesterday"]) and "day" in q:
            words = q.split(); n = None
            for i,w in enumerate(words):
                if w in ("last","previous","past","yesterday") and i+1 < len(words):
                    nxt = words[i+1]
                    if nxt == "day": n=1; break
                    if nxt.isdigit(): n=int(nxt); break
                    if nxt in NUMBER_WORDS: n=NUMBER_WORDS[nxt]; break
            if n:
                end = today.date()-timedelta(days=1); start = end-timedelta(days=n-1)
                return {"type": QueryType.LAST_N_DAYS,
                        "start_date": DateParser.date_to_yyyymmdd(start),
                        "end_date":   DateParser.date_to_yyyymmdd(end),
                        "label": "Yesterday" if n==1 else f"Last {n} Days"}

        # 1️⃣1️⃣ Last N weeks
        if any(k for k in ["last","previous","past"]) and "week" in q:
            words = q.split()
            for i,w in enumerate(words):
                if w in ("last","previous","past","yesterday") and i+1 < len(words):
                    try:
                        n = int(words[i+1]) if words[i+1].isdigit() else DateParser.WORD_TO_NUM.get(words[i+1],1)
                        ls = today.date()-timedelta(days=today.weekday()+1)
                        ss = ls-timedelta(weeks=n-1,days=6)
                        return {"type": QueryType.LAST_N_WEEKS,
                                "start_date": DateParser.date_to_yyyymmdd(ss),
                                "end_date":   DateParser.date_to_yyyymmdd(ls),
                                "label": f"Last {n} Week{'s' if n>1 else ''}"}
                    except Exception: pass

        # 1️⃣2️⃣ This month
        if "this month" in q or "current month" in q:
            start = today.date().replace(day=1)
            return {"type": QueryType.THIS_MONTH,
                    "start_date": DateParser.date_to_yyyymmdd(start),
                    "end_date":   DateParser.today_yyyymmdd(),
                    "label": f"This Month - {today.strftime('%B %Y')}"}

        # 1️⃣3️⃣ Last month
        if "last month" in q or "previous month" in q:
            fot  = today.date().replace(day=1)
            ldp  = fot-timedelta(days=1)
            fdp  = ldp.replace(day=1)
            return {"type": QueryType.LAST_MONTH,
                    "start_date": DateParser.date_to_yyyymmdd(fdp),
                    "end_date":   DateParser.date_to_yyyymmdd(ldp),
                    "label": f"Last Month - {fdp.strftime('%B %Y')}"}

        # 1️⃣4️⃣ MoM full FY
        if is_mom and not last_n_month_match and not last_n_quarter_match and not last_n_year_match:
            return mom_logic(q)

        # 1️⃣5️⃣ Last N months (with optional MoM)
        if last_n_month_match and not is_qoq and not is_yoy:
            return last_n_mom_logic(q)

        # 1️⃣6️⃣ This quarter
        if "this quarter" in q or "current quarter" in q:
            cq = DateParser.get_fy_quarter(today.month)
            if cq==1:   s=datetime(current_fy,4,1)
            elif cq==2: s=datetime(current_fy,7,1)
            elif cq==3: s=datetime(current_fy,10,1)
            else:       s=datetime(current_fy+1,1,1)
            return {"type": QueryType.THIS_QUARTER,
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date":   DateParser.today_yyyymmdd(),
                    "label": f"This Quarter FY{current_fy} Q{cq}"}

        # 1️⃣7️⃣ Last quarter
        if "last quarter" in q or "previous quarter" in q:
            cq  = DateParser.get_fy_quarter(today.month)
            lq  = cq-1 if cq>1 else 4; lqfy = current_fy if cq>1 else current_fy-1
            if lq==1:   s,e=datetime(lqfy,4,1),   datetime(lqfy,6,30)
            elif lq==2: s,e=datetime(lqfy,7,1),   datetime(lqfy,9,30)
            elif lq==3: s,e=datetime(lqfy,10,1),  datetime(lqfy,12,31)
            else:       s,e=datetime(lqfy+1,1,1), datetime(lqfy+1,3,31)
            return {"type": QueryType.LAST_QUARTER,
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date":   DateParser.date_to_yyyymmdd(e.date()),
                    "label": f"Last Quarter FY{lqfy} Q{lq}"}

        # 1️⃣7️⃣ Last N quarters
        if last_n_quarter_match and not is_yoy:
            return last_n_quarte_mom_qoq(q)

        # 1️⃣9️⃣ YoY
        # 1️⃣8️⃣ YoY / year wise
        if is_yoy and not last_n_year_match:
            return yoy_logic(q)

        # 2️⃣0️⃣ Year range
        yr = year_range_logic(q)
        if yr: return yr

        # 2️⃣1️⃣ Multiple years
        ya = detect_year_and(q)
        if ya: return ya

        # 2️⃣2️⃣ Last N years
        if last_n_year_match:
            return last_n_year_mom_qoq_yoy(q)

        # 2️⃣3️⃣ This year / this FY
        if re.search(r"\b(this|current)\s*(fy|fiscal\s*year|year)?\b", q):
            s,_ = DateParser.get_fy_start_end(current_fy)
            return {"type": QueryType.THIS_YEAR,"start_date":s,
                    "end_date":DateParser.today_yyyymmdd(),"label":f"FY{current_fy} (YTD)"}

        # 2️⃣4️⃣ Last year
        if re.search(r"\b(last|previous|prev)\s*(fy|fiscal\s*year|year)?\b", q):
            s,e = DateParser.get_fy_start_end(current_fy-1)
            return {"type": QueryType.LAST_YEAR,"start_date":s,"end_date":e,
                    "label":f"FY{current_fy-1}"}

        # 2️⃣5️⃣ FY format
        if "fy" in q:
            for w in q.split():
                if w.startswith("fy") and w[2:].isdigit():
                    year = int(w[2:]) if len(w[2:])==4 else 2000+int(w[2:])
                    s,e  = DateParser.get_fy_start_end(year)
                    return {"type": QueryType.SPECIFIC_YEAR,"year":year,
                            "start_date":s,"end_date":e,"label":f"FY{year}"}

        # 2️⃣6️⃣ Specific date or range
        di = DateParser.parse_specific_date_or_range(q)
        if di: return di

        print("No date intent detected")
        # 2️⃣7️⃣ From/after/since
        fi = DateParser.parse_from_date(q)
        print(f"From/after/since intent: {fi}")
        if fi: return fi

        print("No from/after/since intent detected")

        # 2️⃣8️⃣ FY till month
        tm = DateParser.parse_fy_till_month(q)
        if tm: return tm

        # 2️⃣9️⃣ Specific month
        def _has_day(t): return bool(re.search(r'\b\d{1,2}(st|nd|rd|th)?\b', t))
        mp = r'\b(' + '|'.join(DateParser.MONTH_MAP.keys()) + r')\b'
        mm = re.search(mp, q, re.IGNORECASE)
        found_month = None
        if mm and not _has_day(q):
            found_month = DateParser.MONTH_MAP[mm.group(1).lower()]
        found_year = None
        ym = re.search(r'\b(20\d{2})\b', q)
        if ym: found_year = int(ym.group(1))
        if found_month:
            fy_shift = -1 if re.search(r"\b(last|previous)\s+year\b",q) else 0
            if found_year: year=found_year
            else: eff=current_fy+fy_shift; year=eff if found_month>=4 else eff+1
            sd=datetime(year,found_month,1); _,ld=monthrange(year,found_month); ed=datetime(year,found_month,ld)
            return {"type":QueryType.DATE_RANGE,
                    "start_date":DateParser.date_to_yyyymmdd(sd.date()),
                    "end_date":  DateParser.date_to_yyyymmdd(ed.date()),
                    "label":     sd.strftime("%B %Y")}
        if found_year:
            s,e = DateParser.get_fy_start_end(found_year)
            return {"type":QueryType.SPECIFIC_YEAR,"year":found_year,
                    "start_date":s,"end_date":e,"label":f"FY{found_year}"}
        return None


# ============================================================================
# SQL GENERATOR — programmatic, no LLM
# ============================================================================
def _numeric_expr(col: str) -> str:
    return f"TRY_CAST({col} AS DECIMAL(18,2))"


class TaskSQLGenerator:

    @staticmethod
    def _convert_date_filter(date_filter: str) -> str:
        """
        Convert YYYYMMDD BETWEEN format to date_parse format
        Input:  "created_date_c BETWEEN 20260401 AND 20260331"
        Output: "TRY(date_parse(created_date_c, '%d-%m-%Y')) BETWEEN TRY(date_parse('01-04-2025', '%d-%m-%Y')) AND TRY(date_parse('31-03-2025', '%d-%m-%Y'))"
        """
        
        def yyyymmdd_to_ddmmyyyy(yyyymmdd_str: str) -> str:
            """Convert YYYYMMDD to DD-MM-YYYY format"""
            if len(yyyymmdd_str) != 8 or not yyyymmdd_str.isdigit():
                return yyyymmdd_str
            yyyy = yyyymmdd_str[:4]
            mm = yyyymmdd_str[4:6]
            dd = yyyymmdd_str[6:8]
            return f"{dd}-{mm}-{yyyy}"
        
        # Match patterns like "created_date_c BETWEEN 20260401 AND 20260331"
        between_match = re.search(
            r'(\w+)\s+BETWEEN\s+(\d{8})\s+AND\s+(\d{8})',
            date_filter,
            re.IGNORECASE
        )
        
        if between_match:
            col_name = between_match.group(1)
            start_date_yyyymmdd = between_match.group(2)
            end_date_yyyymmdd = between_match.group(3)
            
            start_ddmmyyyy = yyyymmdd_to_ddmmyyyy(start_date_yyyymmdd)
            end_ddmmyyyy = yyyymmdd_to_ddmmyyyy(end_date_yyyymmdd)
            
            return (f"TRY(date_parse({col_name}, '%d-%m-%Y')) "
                    f"BETWEEN TRY(date_parse('{start_ddmmyyyy}', '%d-%m-%Y')) "
                    f"AND TRY(date_parse('{end_ddmmyyyy}', '%d-%m-%Y'))")
        
        # Match patterns like "created_date_c >= 20260401"
        ge_match = re.search(r'(\w+)\s+(>=)\s+(\d{8})', date_filter)
        if ge_match:
            col_name = ge_match.group(1)
            date_yyyymmdd = ge_match.group(3)
            date_ddmmyyyy = yyyymmdd_to_ddmmyyyy(date_yyyymmdd)
            return f"TRY(date_parse({col_name}, '%d-%m-%Y')) >= TRY(date_parse('{date_ddmmyyyy}', '%d-%m-%Y'))"
        
        # Match patterns like "created_date_c <= 20260331"
        le_match = re.search(r'(\w+)\s+(<=)\s+(\d{8})', date_filter)
        if le_match:
            col_name = le_match.group(1)
            date_yyyymmdd = le_match.group(3)
            date_ddmmyyyy = yyyymmdd_to_ddmmyyyy(date_yyyymmdd)
            return f"TRY(date_parse({col_name}, '%d-%m-%Y')) <= TRY(date_parse('{date_ddmmyyyy}', '%d-%m-%Y'))"
        
        return date_filter

    @staticmethod
    def build_where_clause(
        filters:     Dict[str, Any],
        date_filter: Optional[str] = None,
    ) -> str:
        conditions = []
        if date_filter:
            converted_filter = TaskSQLGenerator._convert_date_filter(date_filter)
            conditions.append(converted_filter)

        for col, values in filters.items():
            col_meta   = TaskColumnMetadata.COLUMNS.get(col, {})
            col_type   = col_meta.get("type", "VARCHAR")
            col_conditions = []
            value_list = values if isinstance(values, list) else [values]

            for v in value_list:
                v_str = str(v).strip()
                v_low = v_str.lower()

                # ── Special tokens ────────────────────────────────────────────
                # Follow-up: expand to all follow-up subjects
                if v_low == "__followup__":
                    col_conditions.append(
                        f"LOWER({col}) IN ('follow up','sales follow up',"
                        f"'experience calling follow up','welcome calling follow up')"
                    )
                    continue

                # Tried Calling: LIKE pattern (dynamic timestamp)
                if v_low == "__tried_calling__":
                    col_conditions.append(f"LOWER({col}) LIKE 'tried calling%'")
                    continue

                # Click to call: LIKE pattern
                if v_low == "__click_to_call__":
                    col_conditions.append(f"LOWER({col}) LIKE 'click to call%'")
                    continue

                # Cancelled: expand to all spelling variants
                if v_low == "__cancelled__":
                    col_conditions.append(
                        f"LOWER({col}) IN ('cancelled','canceled','cancel','cancellation')"
                    )
                    continue

                # ── project_c — always LIKE, spaces → % ──────────────────────
                if col == "project_c":
                    like_val = v_low.replace(" ", "%")
                    col_conditions.append(f"LOWER({col}) LIKE '%{like_val}%'")
                    continue

                # ── product_category_c — always LIKE ─────────────────────────
                if col == "product_category_c":
                    col_conditions.append(f"LOWER({col}) LIKE '%{v_low}%'")
                    continue

                # ── Owner name — exact for full name, LIKE for partial ─────────
                if col in ("ownername_c", "activity_ownername_c"):
                    if " " in v_low:
                        col_conditions.append(f"LOWER({col}) = '{v_low}'")
                    else:
                        col_conditions.append(f"LOWER({col}) LIKE '%{v_low}%'")
                    continue

                # ── status_c — normalize cancelled variants ───────────────────
                if col == "status_c":
                    col_conditions.append(f"LOWER({col}) = '{v_low}'")
                    continue

                # ── sales_team_feedback_c — exact match ───────────────────────
                if col == "sales_team_feedback_c":
                    col_conditions.append(f"{col} = '{v_str}'")
                    continue

                # ── customer_feedback_c — exact match ────────────────────────
                if col == "customer_feedback_c":
                    col_conditions.append(f"{col} = '{v_str}'")
                    continue

                # ── priority_c — exact match ──────────────────────────────────
                if col == "priority_c":
                    col_conditions.append(f"{col} = '{v_str}'")
                    continue

                # ── INTEGER/DOUBLE numeric ────────────────────────────────────
                if col_type in ("INTEGER", "DOUBLE"):
                    from task_sql import parse_numeric_condition
                    nc = parse_numeric_condition(col, v_low)
                    if nc:
                        col_conditions.append(nc)
                    continue

                # ── Default LIKE ──────────────────────────────────────────────
                col_conditions.append(f"LOWER({col}) LIKE '%{v_low}%'")

            if col_conditions:
                join_op = " AND " if col_type in ("INTEGER","DOUBLE") else " OR "
                if len(col_conditions) == 1:
                    conditions.append(col_conditions[0])
                else:
                    conditions.append("(" + join_op.join(col_conditions) + ")")

        return "WHERE " + " AND ".join(conditions) if conditions else ""

    @staticmethod
    def generate_sql(
        catalog:           str,
        schema:            str,
        table:             str,
        agg_infos:         List[Dict],
        group_by:          List[str],
        filters:           Dict[str, Any],
        date_range:        Optional[Tuple[str, str]] = None,
        date_column:       str = "created_date_c",
        period_type:       Optional[str] = None,
        period_sort_value: Optional[str] = None,
    ) -> str:
        select_parts   = []
        group_by_parts = []

        # ── Period column (month / quarter / year) — YYYYMMDD integer math ────
        if period_type == "month":
            date_expr = f"TRY(date_parse({date_column}, '%d-%m-%Y'))"
            yr_expr  = f"CAST(year({date_expr}) AS VARCHAR)"
            mo_expr  = f"LPAD(CAST(month({date_expr}) AS VARCHAR), 2, '0')"
            time_expr = f"({yr_expr} || '-' || {mo_expr})"
            select_parts.append(f"{time_expr} AS period")
            group_by_parts.append(time_expr)

        elif period_type == "quarter":
            month_expr = f"MONTH(TRY(date_parse({date_column}, '%d-%m-%Y')))"
            year_expr  = f"YEAR(TRY(date_parse({date_column}, '%d-%m-%Y')))"
            q_expr = (
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
            fy_expr = (
                f"CASE "
                f"WHEN month({date_expr}) >= 4 "
                f"THEN CAST(year({date_expr}) AS VARCHAR) || '-' || CAST(year({date_expr}) + 1 AS VARCHAR) "
                f"ELSE CAST(year({date_expr}) - 1 AS VARCHAR) || '-' || CAST(year({date_expr}) AS VARCHAR) "
                f"END"
            )
            select_parts.append(f"{fy_expr} AS period")
            group_by_parts.append(fy_expr)

        # Add period_sort for multi-period queries (for UNION ALL ordering)
        if period_sort_value:
            select_parts.append(f"'{period_sort_value}' AS period_sort")
            group_by_parts.append(f"'{period_sort_value}'")

        # User group_by
        select_parts.extend(group_by)
        group_by_parts.extend(group_by)

        # Aggregation
        for agg in agg_infos:
            if agg["type"] == AggregationType.COUNT:
                agg_expr = f'COUNT(*) AS "{agg["alias"]}"'
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

        # Date filter — pure YYYYMMDD integer BETWEEN
        date_filter = None
        if date_range:
            start, end = date_range
            if start and end:
                date_filter = f"{date_column} BETWEEN {start} AND {end}"
            elif start:
                date_filter = f"{date_column} >= {start}"
            elif end:
                date_filter = f"{date_column} <= {end}"

        where_clause    = TaskSQLGenerator.build_where_clause(filters=filters, date_filter=date_filter)
        group_by_clause = ("GROUP BY " + ", ".join(group_by_parts)) if group_by_parts else ""

        sql = "\n".join(filter(None, [select_clause, from_clause, where_clause, group_by_clause]))
        return sql.strip()


# ============================================================================
# SQL VALIDATOR
# ============================================================================
class TaskSQLValidator:
    @staticmethod
    def validate(sql: str) -> Tuple[bool, Optional[str]]:
        if not sql or not sql.strip():
            return False, "SQL query is empty"
        sql_u = sql.upper()
        if "SELECT" not in sql_u: return False, "Missing SELECT"
        if "FROM"   not in sql_u: return False, "Missing FROM"
        if sql.count("(") != sql.count(")"): return False, "Unbalanced parentheses"
        for kw in ["DROP","DELETE","TRUNCATE","ALTER","CREATE","INSERT","UPDATE"]:
            if f" {kw} " in sql_u: return False, f"Dangerous keyword: {kw}"
        return True, "SQL is valid"


# ============================================================================
# PRESTO EXECUTION
# ============================================================================
def run_presto_query(sql: str) -> Tuple[List[Dict], Optional[List[str]], Optional[str]]:
    try:
        with prestodb.dbapi.connect(
            host=PRESTO_HOST, port=PRESTO_PORT, user=PRESTO_USER,
            catalog=PRESTO_CATALOG, schema=PRESTO_SCHEMA,
            http_scheme="https",
            auth=BasicAuthentication(PRESTO_USER, PRESTO_PASSWORD),
        ) as conn:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall() or []
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, r)) for r in rows], cols, None
    except Exception as e:
        logger.error(f"Presto error: {e}", exc_info=True)
        return [], [], f"Presto execution failed: {str(e)}"


# ============================================================================
# POST-PROCESSING
# ============================================================================
TEMPORAL_COLS    = {"period_sort","period","fy_year","month","quarter","year"}
NON_ADDITIVE_MARKERS  = ["%",":"]
NON_ADDITIVE_KEYS     = {"fy_year","month","quarter","days","year","financial year","fiscal year"}
_TIME_DIMENSION_WORDS = {"year","month","quarter","week","day","fiscal"}

def _is_additive_key(key: str) -> bool:
    kl = key.lower()
    if kl in NON_ADDITIVE_KEYS: return False
    if any(w in kl for w in _TIME_DIMENSION_WORDS): return False
    return not any(m in key for m in NON_ADDITIVE_MARKERS)

def parse_numeric_condition(col: str, value: str) -> Optional[str]:
    v = value.strip().lower()
    for pattern, op in [
        (r"^(>=)\s*(\d+(\.\d+)?)$",">="), (r"^(<=)\s*(\d+(\.\d+)?)$","<="),
        (r"^(>)\s*(\d+(\.\d+)?)$",">"),   (r"^(<)\s*(\d+(\.\d+)?)$","<"),
        (r"^(=)\s*(\d+(\.\d+)?)$","="),
    ]:
        m = re.match(pattern, v)
        if m: return f"{col} {op} {m.group(2)}"
    return None

def enforce_descending_order(sql: str) -> str:
    sql = sql.rstrip(";").strip()
    if re.search(r'\bORDER BY\b', sql, re.IGNORECASE):
        def fix_order(m):
            parts = [p.strip() for p in m.group(1).split(",")]
            new   = []
            for p in parts:
                toks = p.split()
                cn   = toks[0].lower().strip('"').strip("'")
                if cn in TEMPORAL_COLS:           new.append(p)
                elif len(toks) == 1:              new.append(p + " DESC")
                elif toks[-1].upper() == "ASC":   new.append(" ".join(toks[:-1]) + " DESC")
                else:                             new.append(p)
            return "ORDER BY " + ", ".join(new)
        sql = re.sub(r"ORDER BY\s+(.+?)$", fix_order, sql, flags=re.IGNORECASE)
    else:
        am = re.search(
            r'\bAS\s+"?(task_count|total_task|count|tasks)\b"?', sql, re.IGNORECASE
        )
        if am:
            sql += f' ORDER BY "{am.group(1)}" DESC'
    return sql

def add_total_row(data: list) -> list:
    SKIP = {"period_sort"}
    if not data or len(data) <= 1: return data
    total_row = {}; fsd = False
    for key in data[0].keys():
        if key in SKIP: total_row[key]=None; continue
        non_null = [r.get(key) for r in data if r.get(key) is not None]
        if not _is_additive_key(key):
            total_row[key] = "Total" if not fsd else "-"; fsd=True; continue
        try:
            total_row[key] = round(sum(float(v) for v in non_null), 2)
        except (ValueError, TypeError):
            total_row[key] = "Total" if not fsd else "-"; fsd=True
    data.append(total_row)
    return data

def generate_graph_from_sql_json(data, question):
    if not data:
        return None

    keys = list(data[0].keys())

    # Single value (one row, one column) → simple bar
    if len(data) == 1 and len(keys) == 1:
        value = list(data[0].values())[0]
        q = question.lower()
        value_label = "Tasks" if "task" in q else "Count"

        plt.figure(figsize=(4, 5))
        plt.bar([value_label], [value])
        plt.ylabel(value_label)
        plt.title(question)
        plt.text(0, value, f"{int(value) if value else 0:,}", ha="center", va="bottom", fontsize=11)

        os.makedirs("graphs", exist_ok=True)
        today_date = datetime.utcnow().strftime("%Y-%m-%d")
        graph_id = uuid.uuid4().hex
        local_path = f"graphs/{graph_id}.png"
        plt.tight_layout()
        plt.savefig(local_path, dpi=150)
        plt.close()

        cos_key = f"graphs/{today_date}/{graph_id}.png"
        try:
            cos.upload_file(Filename=local_path, Bucket=COS_BUCKET, Key=cos_key)
        except Exception as e:
            print("COS upload error:", e)
            return graph_id, None

        url = f"{COS_ENDPOINT}/{COS_BUCKET}/{cos_key}"

        graph_store[graph_id] = {"graph_type": "single_value_bar", "value": value, "url": url, "image_path": local_path}
        return graph_id, url

    # Multi-row → vertical bar chart
    # Multi-row → vertical bar chart
    # Remove any Total rows before plotting
    data = [row for row in data if not any(str(v) in ("Total", "-") for v in row.values())]

    if not data:
        return None

    # Prefer 'period' as category if present, else first string column
    # Prefer 'period' as category if present, else first string column
    if "period" in keys:
        category_key = "period"
    else:
        category_key = next((k for k in keys if isinstance(data[0][k], str)), None)
    value_key = next((k for k in keys if not isinstance(data[0][k], str)), None)

    if not category_key or not value_key:
        return None

    # If multiple string columns exist (e.g. period + project_category_c),
    # combine them into one label so each bar is unique
    extra_str_keys = [k for k in keys if k != category_key and k != value_key and isinstance(data[0][k], str)]
    if extra_str_keys:
        for row in data:
            row[category_key] = row[category_key] + " | " + " | ".join(str(row[k]) for k in extra_str_keys)

    q = question.lower()
    value_label    = "Tasks" if "task" in q else "Count"
    category_label = "Stage" if "stage" in q else "Owner" if "owner" in q else "Category"

    data = sorted(data, key=lambda x: x[value_key], reverse=True)
    categories = [str(row[category_key]) for row in data]
    values     = [int(row[value_key]) if row[value_key] is not None else 0 for row in data]

    total_items   = len(categories)
    max_label_len = max(len(c) for c in categories)

    fig_width  = max(8, total_items * 0.6)
    font_size  = 10
    rotation   = 30
    if total_items > 8:  rotation = 45
    if total_items > 15: rotation, font_size = 60, 9
    if total_items > 25: rotation, font_size = 75, 8
    if total_items > 40: rotation, font_size = 90, 7
    if max_label_len > 20: font_size -= 1
    if max_label_len > 35: font_size -= 1

    wrapped = ["\n".join(textwrap.wrap(c, width=12)) for c in categories]

    plt.figure(figsize=(fig_width, 6))
    bars = plt.bar(wrapped, values)
    plt.ylabel(value_label, fontsize=font_size + 1)
    plt.xlabel(category_label, fontsize=font_size + 1)
    plt.title(question, fontsize=font_size + 3)
    plt.xticks(rotation=rotation, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    

    max_value = max(values) if values else 1
    for bar, val in zip(bars, values):
        if max_value > 0 and val > max_value * 0.07:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_value * 0.01,
                     f"{val:,}", ha="center", va="bottom", fontsize=font_size - 1)

    plt.tight_layout()

    os.makedirs("graphs", exist_ok=True)
    today_date = datetime.utcnow().strftime("%Y-%m-%d")
    graph_id   = uuid.uuid4().hex
    local_path = f"graphs/{graph_id}.png"
    plt.savefig(local_path, dpi=150)
    plt.close()

    cos_key = f"graphs/{today_date}/{graph_id}.png"
    try:
        cos.upload_file(Filename=local_path, Bucket=COS_BUCKET, Key=cos_key)
        print("COS upload SUCCESS:", cos_key)
    except Exception as e:
        print("COS upload FAILED:", e)
        return graph_id, None

    url = f"{COS_ENDPOINT}/{COS_BUCKET}/{cos_key}"

    graph_store[graph_id] = {"graph_type": "vertical_bar", "url": url, "image_path": local_path}
    return graph_id, url



# ============================================================================
# MAIN ENGINE — mirrors inventory NLToSQLEngine exactly
# ============================================================================
class TaskNLToSQLEngine:
    def __init__(self):
        self.llm_detector    = LLMIntentDetector()
        self.intent_detector = TaskIntentDetector()
        self.sql_generator   = TaskSQLGenerator()
        self.sql_validator   = TaskSQLValidator()

    def process(self, request: TaskSQLRequest) -> TaskSQLResponse:
        question = request.question.strip()
        logger.info(f"Processing: {question}")

        # Step 1: LLM → JSON intent
        llm_intent = self.llm_detector.extract_intent(question)
        # Step 2: Deterministic normalisation
        llm_intent = self.intent_detector.normalize_intent(llm_intent, question)
        logger.info(f"Normalized intent: {llm_intent}")

        # Step 3: Python date parsing
        date_intent = self.intent_detector.detect_date_intent(question)
        logger.info(f"Date intent: {date_intent}")
        if date_intent is None:
            fy = DateParser.get_current_fy()
            s, e = DateParser.get_fy_start_end(fy)
            date_intent = {"type": QueryType.CURRENT_FY, "start_date": s,
                           "end_date": e, "label": f"FY{fy}"}

        query_type = date_intent["type"]
        logger.info(f"Query type: {query_type}")

        # Step 4: Aggregation map
        agg_map = {
            "task_count": {"type": AggregationType.COUNT, "column": "*", "alias": "Task Count"},
        }
        agg_infos = []
        for agg in llm_intent.get("aggregation", ["task_count"]):
            key = agg.strip().lower().replace(" ", "_")
            if key in agg_map:
                agg_infos.append(agg_map[key])
            else:
                logger.warning(f"Unknown aggregation: {agg}")
        if not agg_infos:
            agg_infos.append(agg_map["task_count"])

        # Step 5: Filters and group_by
        filters  = llm_intent.get("filters", {})
        group_by = llm_intent.get("group_by", [])
        logger.info(f"Filters: {filters} | Group by: {group_by}")

        # Step 6: Date column
        date_column = llm_intent.get("date_column", "created_date_c")
        if date_column not in TaskColumnMetadata.DATE_COLUMNS:
            logger.warning(f"Invalid date column {date_column}, fallback to created_date_c")
            date_column = "created_date_c"

        # Normalise SPECIFIC_DATE → DATE_RANGE
        if date_intent.get("type") == QueryType.SPECIFIC_DATE:
            date_intent = {"type": QueryType.DATE_RANGE,
                           "start_date": date_intent["date"],
                           "end_date":   date_intent["date"],
                           "label":      date_intent.get("label")}

        # Step 7: SQL generation
        date_ranges = []
        MULTI_PERIOD = [
            QueryType.QUARTER_WISE, QueryType.MONTH_WISE, QueryType.YEAR_WISE,
            QueryType.MULTI_DATE_RANGE, QueryType.MONTH_RANGE_MONTH_WISE,
            QueryType.MONTH_MULTI_MONTH_WISE, QueryType.MULTI_MONTH,
        ]

        if query_type in MULTI_PERIOD:
            periods = (date_intent.get("quarters") or date_intent.get("periods")
                       or date_intent.get("ranges") or [])
            if not periods:
                raise ValueError("No periods defined for multi-period query")

            sqls = []
            for period in periods:
                ps = period["start_date"]; pe = period["end_date"]
                pl = period.get("quarter") or period.get("label") or period.get("year", f"{ps}-{pe}")
                if query_type in [QueryType.MONTH_WISE, QueryType.MONTH_RANGE_MONTH_WISE,
                                   QueryType.MONTH_MULTI_MONTH_WISE, QueryType.MULTI_MONTH]:
                    pt = "month"
                elif query_type in [QueryType.QUARTER_WISE, QueryType.MULTI_DATE_RANGE]:
                    pt = "quarter"
                elif query_type == QueryType.YEAR_WISE:
                    pt = "year"
                else:
                    pt = None

                sql = TaskSQLGenerator.generate_sql(
                    catalog=request.catalog, schema=request.db_schema, table=request.table,
                    agg_infos=agg_infos, group_by=group_by, filters=filters,
                    date_range=(ps, pe), date_column=date_column,
                    period_type=pt, period_sort_value=ps,
                )
                sqls.append(sql)
                date_ranges.append((ps, pe, pl))

            final_sql = "\n\nUNION ALL\n\n".join(sqls)
            final_sql = f"(\n{final_sql}\n) ORDER BY period_sort"
            if group_by:
                final_sql += ", " + ", ".join(group_by)

        else:
            final_sql = TaskSQLGenerator.generate_sql(
                catalog=request.catalog, schema=request.db_schema, table=request.table,
                agg_infos=agg_infos, group_by=group_by, filters=filters,
                date_range=(date_intent["start_date"], date_intent.get("end_date")),
                date_column=date_column, period_type=None,
            )
            date_ranges.append((date_intent["start_date"], date_intent.get("end_date"),
                                 date_intent.get("label")))

        # Step 8: Validate → Execute → Post-process
        final_sql = enforce_descending_order(final_sql)
        is_valid, val_msg = self.sql_validator.validate(final_sql)
        data, schema_cols, err_msg = (
            run_presto_query(final_sql) if is_valid else ([], [], val_msg)
        )
        data   = add_total_row(data)
        totals = {k: v for row in data for k, v in row.items()
                  if isinstance(v, (int, float)) and _is_additive_key(k)}

        logger.info(f"Generated SQL:\n{final_sql}")

        q_lower = question.lower()
        # graph_url  = None
        # graph_urls = None   # will hold list of {label, url} dicts for multi-graph
        # markdown   = None

        # # ── Detect whether we need multi-graph (dimension × period) ──────────
        # # Condition: data has a 'period' column AND at least one group_by column
        # has_period_col   = data and "period" in data[0]
        # has_group_by_col = bool(group_by)

        # if has_period_col and has_group_by_col:
        #     # One graph per dimension value, each graph shows all periods for that value
        #     from collections import defaultdict
        #     dim_col      = group_by[0]   # e.g. property_size_c
        #     metric_alias = agg_infos[0]["alias"]

        #     # Group rows by dimension value
        #     dim_groups = defaultdict(list)
        #     for row in data:
        #         dim_val = row.get(dim_col, "Unknown")
        #         dim_groups[dim_val].append(row)

        #     generated = []
        #     for dim_value, rows in sorted(dim_groups.items(), key=lambda x: str(x[0])):
        #         # Each graph: x-axis = period, y-axis = metric
        #         # Build sub_data with period as category key
        #         sub_data = [
        #             {"period": row["period"], metric_alias: int(row[metric_alias]) if row[metric_alias] is not None else 0}
        #             for row in rows
        #         ]
        #         sub_data = sorted(sub_data, key=lambda x: x["period"])
        #         if not sub_data:
        #             continue
        #         title = f"{dim_value} — {question}"
        #         result = generate_graph_from_sql_json(sub_data, title)
        #         if result:
        #             generated.append({"label": str(dim_value), "url": result[1]})

        #     if generated:
        #         graph_urls = generated
        #         graph_url  = generated[0]["url"]
        #         markdown   = "\n".join(
        #             f"**{g['label']}**\n![graph]({g['url']})" for g in generated
        #         )
        # else:
        #     graph_result = generate_graph_from_sql_json(data, question)
        #     graph_url = graph_result[1] if graph_result else None
        #     markdown  = f"![Task graph]({graph_url})" if graph_url else None


        return TaskSQLResponse(
            status             = "success" if is_valid and not err_msg else "error",
            query_type         = str(query_type.value) if isinstance(query_type, QueryType) else str(query_type),
            sql                = final_sql,
            schema_metadata    = schema_cols,
            data               = data,
            execution          = {"executed": is_valid, "row_count": len(data), "error": err_msg},
            date_ranges        = [DateRange(start_date=s, end_date=e, label=l) for s,e,l in date_ranges],
            is_valid           = is_valid,
            validation_message = val_msg,
            metadata           = {"llm_intent": llm_intent, "date_intent": date_intent},
            intent_summary     = {
                "aggregation":      [a["alias"] for a in agg_infos],
                "filters_detected": filters,
                "group_by_columns": group_by,
                "date_range":       date_intent.get("label") or "Multiple Periods",
            },
            totals = totals
        )


# ============================================================================
# FASTAPI APP
# ============================================================================
app    = FastAPI(title="Task NL-to-SQL Engine (Inventory Architecture)", version="1.0.0")
engine = TaskNLToSQLEngine()


@app.post("/generate-sql", response_model=TaskSQLResponse)
async def generate_sql(request: TaskSQLRequest):
    try:
        return engine.process(request)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return TaskSQLResponse(
            status="error",
            query_type="",
            sql="",
            schema_metadata=[],
            data=[],
            execution={"executed": False, "row_count": 0, "error": str(e)},
            date_ranges=[],
            is_valid=False,
            validation_message=str(e),
            metadata={"error": str(e)},
            intent_summary={"error": str(e)},
            totals={}
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0",
            "date_format": "YYYYMMDD integer — no date_parse"}


if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 70)
    logger.info("TASK NL-TO-SQL ENGINE — INVENTORY ARCHITECTURE")
    logger.info("LLM: JSON intent only | Python: date parsing + SQL generation")
    logger.info("Date format: YYYYMMDD integer BETWEEN (no date_parse, no TRY)")
    logger.info("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8002)
