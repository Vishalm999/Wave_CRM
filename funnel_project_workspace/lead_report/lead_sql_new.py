

# --------------------------------new code ----------------------------------------------

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

load_dotenv()

# ============================================================================
# LOGGING & CONFIG
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("lead_nl_to_sql.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PRESTO_HOST     = os.getenv("PRESTO_hostname")
PRESTO_PORT     = int(os.getenv("PRESTO_portnumber", "31351"))
PRESTO_USER     = os.getenv("PRESTO_username")
PRESTO_PASSWORD = os.getenv("PRESTO_password")
PRESTO_CATALOG  = "salesforcereport"
PRESTO_SCHEMA   = "lead_fy_year"
PRESTO_TABLE    = "lead_fy_report"

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
# LEAD SCHEMA  — mirrors ColumnMetadata in inventory
# ============================================================================
class LeadColumnMetadata:
    # All columns: name → {type, description}
    COLUMNS = {
        "lead_id_c":                   {"type": "VARCHAR", "description": "Lead unique ID — use COUNT(DISTINCT lead_id_c) for lead count"},
        "id":                          {"type": "VARCHAR", "description": "Alternate lead ID"},
        "ownername_c":                 {"type": "VARCHAR", "description": "Name of the lead owner / sales person"},
        "created_date_c":              {"type": "INTEGER", "description": "Lead creation date in YYYYMMDD integer format"},
        "lastmodified_date_c":         {"type": "INTEGER", "description": "Last modified date in YYYYMMDD integer format"},
        "lead_source_c":               {"type": "VARCHAR", "description": "High-level lead source (bulk sale, digital, referral, etc.)"},
        "lead_source_sub_category_c":  {"type": "VARCHAR", "description": "Detailed lead sub-source (facebook, google, 99 acres, etc.)"},
        "project_c":                   {"type": "VARCHAR", "description": "Project zone / sales org (wave city, wmcc, wave estate, etc.)"},
        "product_category_c":          {"type": "VARCHAR", "description": "Product / unit name (veridia, dream homes, eden, plots, etc.)"},
        "property_size_c":             {"type": "VARCHAR", "description": "Property size type (1BHK, 2BHK, 3BHK, plots, skyvillas, etc.)"},
        "property_type_c":             {"type": "VARCHAR", "description": "Property type (Residential, Commercial)"},
        "budget_range_c":              {"type": "VARCHAR", "description": "Customer budget range (free text, e.g. 3cr-4cr, 50 Lacs)"},
        "customer_feedback_c":         {"type": "VARCHAR", "description": "Customer feedback / lead qualification (Junk, Interested, Discussion Pending, Not Interested)"},
        "disqualification_reason_c":   {"type": "VARCHAR", "description": "Reason for disqualification of leads"},
        "junk_reason_c":               {"type": "VARCHAR", "description": "Reason lead is junk"},
        "city_c":                      {"type": "VARCHAR", "description": "Customer city"},
        "status":                      {"type": "VARCHAR", "description": "Lead status (new, nurturing, qualified, unqualified)"},
        "rating_c":                    {"type": "VARCHAR", "description": "Lead rating (Hot, Warm, Cold)"},
        "transfer_status_c":           {"type": "VARCHAR", "description": "Transfer status of the lead"},
        "contact_medium_c":            {"type": "VARCHAR", "description": "How customer was contacted (walk-in, call-in, call-out, email, website, social media, live chat, referral)"},
        "is_appointment_booked_c":     {"type": "VARCHAR", "description": "Whether appointment/meeting is booked (1 = yes)"},
        "sap_customer_code_c":         {"type": "VARCHAR", "description": "SAP customer code"},
    }

    # The date column — stored as YYYYMMDD INTEGER (same as inventory Status_Date)
    DATE_COLUMNS  = ["created_date_c", "lastmodified_date_c"]

    # COUNT column (equivalent to Material_Code in inventory)
    COUNT_COLUMN  = "lead_id_c"

    # Columns that can be summed (none for leads — leads only count)
    SUM_COLUMNS: List[str] = []

    # Groupable dimension columns
    DIMENSION_COLUMNS = [
        "lead_source_c", "lead_source_sub_category_c", "project_c",
        "product_category_c", "property_size_c", "property_type_c",
        "customer_feedback_c", "status", "rating_c", "contact_medium_c",
        "city_c", "ownername_c", "transfer_status_c",
        "disqualification_reason_c", "junk_reason_c", "budget_range_c",
        "is_appointment_booked_c", "sap_customer_code_c",
    ]

    # ── Known valid values for normalisation ─────────────────────────────────
    VALID_VALUES = {
        "lead_source_c": [
            "Bulk Sale", "Channel Partner", "Digital", "Direct", "Direct Walkin",
            "Electronic Media", "Existing Customer", "Lead Reassigned",
            "Outbound Campaign", "Outdoor", "Print Media", "Reference Sale",
            "Referral", "Referral Sale", "Transfered", "Unit Shifting",
            "Word of Mouth",
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
            "SFMC", "Social Media-FB Page", "Spotify", "Taboola",'Event',
            "TOI.com", "Transfered", "Unit Shift", "Word of Mouth", "YouTube",
        ],
        "customer_feedback_c": [
            "Junk", "Interested", "Discussion Pending",
            "Not Interested", "Callback",
        ],
        "status": ["new", "nurturing", "qualified", "unqualified"],
        "rating_c": ["Hot", "Warm", "Cold"],
        "property_type_c": ["Residential", "Commercial","plots","renting",'buying','leasing'],
        "property_size_c": [
            "1BHK", "2BHK", "3BHK", "4BHK", "5BHK",
            "Penthouse", "Plots", "Skyvillas",
            "Commercial Office Space", "Commercial Space",
        ],
        "contact_medium_c": [
            "call - out", "call-in", "email", "live chat",
            "walk - in", "website", "social media", "referral",
        ],
    }

    # ── Known product/category values for product_category_c ─────────────────
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

    # ── Known project values for project_c ────────────────────────────────────
    PROJECT_VALUES = [
        "wave city", "wmcc", "wmcc sec 32", "wmcc sector 32",
        "wave estate", "wave amore", "wave executive floors",
    ]

    # ── KEYWORD MAPPING  (mirrors inventory's KEYWORD_MAPPING) ───────────────
    # Each entry maps a user keyword → {column, value} for filters
    # OR → {aggregation, column} for aggregation keywords
    # OR → {column} alone for group-by-only keywords
    KEYWORD_MAPPING = {
        # ── Lead status keywords ─────────────────────────────────────────────
        "open lead":         {"column": "customer_feedback_c", "value": "Discussion Pending"},
        "open leads":        {"column": "customer_feedback_c", "value": "Discussion Pending"},
        "discussion pending":{"column": "customer_feedback_c", "value": "Discussion Pending"},
        "junk lead":         {"column": "customer_feedback_c", "value": "Junk"},
        "junk leads":        {"column": "customer_feedback_c", "value": "Junk"},
        "junk":              {"column": "customer_feedback_c", "value": "Junk"},
        "qualified lead":    {"column": "customer_feedback_c", "value": "Interested"},
        "qualified leads":   {"column": "customer_feedback_c", "value": "Interested"},
        "interested":        {"column": "customer_feedback_c", "value": "Interested"},
        "disqualified lead": {"column": "customer_feedback_c", "value": "Not Interested"},
        "disqualified leads":{"column": "customer_feedback_c", "value": "Not Interested"},
        "not interested":    {"column": "customer_feedback_c", "value": "Not Interested"},
        "valid lead":        {"column": "customer_feedback_c", "value": "__valid__"},   # special
        "valid leads":       {"column": "customer_feedback_c", "value": "__valid__"},

        # ── Lead status column ────────────────────────────────────────────────
        "new lead":      {"column": "status", "value": "new"},
        "new leads":     {"column": "status", "value": "new"},
        "nurturing":     {"column": "status", "value": "nurturing"},
        "nurturing lead":{"column": "status", "value": "nurturing"},
        "status qualified":   {"column": "status", "value": "qualified"},
        "status unqualified": {"column": "status", "value": "unqualified"},
        "status disqualified":{"column": "status", "value": "unqualified"},

        # ── Rating keywords ───────────────────────────────────────────────────
        "hot lead":  {"column": "rating_c", "value": "Hot"},
        "hot leads": {"column": "rating_c", "value": "Hot"},
        "hot":       {"column": "rating_c", "value": "Hot"},
        "warm lead": {"column": "rating_c", "value": "Warm"},
        "warm leads":{"column": "rating_c", "value": "Warm"},
        "warm":      {"column": "rating_c", "value": "Warm"},
        "cold lead": {"column": "rating_c", "value": "Cold"},
        "cold leads":{"column": "rating_c", "value": "Cold"},
        "cold":      {"column": "rating_c", "value": "Cold"},

        # ── Appointment / meeting ─────────────────────────────────────────────
        "meeting booked":      {"column": "is_appointment_booked_c", "value": "1"},
        "appointment booked":  {"column": "is_appointment_booked_c", "value": "1"},
        "appointment":         {"column": "is_appointment_booked_c", "value": "1"},

        # ── Property type ─────────────────────────────────────────────────────
        "residential": {"column": "property_type_c", "value": "Residential"},
        "commercial":  {"column": "property_type_c", "value": "Commercial"},

        # ── Property size ─────────────────────────────────────────────────────
        "1bhk":      {"column": "property_size_c", "value": "1BHK"},
        "1 bhk":     {"column": "property_size_c", "value": "1BHK"},
        "2bhk":      {"column": "property_size_c", "value": "2BHK"},
        "2 bhk":     {"column": "property_size_c", "value": "2BHK"},
        "3bhk":      {"column": "property_size_c", "value": "3BHK"},
        "3 bhk":     {"column": "property_size_c", "value": "3BHK"},
        "4bhk":      {"column": "property_size_c", "value": "4BHK"},
        "4 bhk":     {"column": "property_size_c", "value": "4BHK"},
        "5bhk":      {"column": "property_size_c", "value": "5BHK"},
        "5 bhk":     {"column": "property_size_c", "value": "5BHK"},
        "penthouse":  {"column": "property_size_c", "value": "Penthouse"},
        "skyvillas":  {"column": "property_size_c", "value": "Skyvillas"},
        "sky villa":  {"column": "property_size_c", "value": "Skyvillas"},

        # ── Contact medium ────────────────────────────────────────────────────
        "call out":     {"column": "contact_medium_c", "value": "call - out"},
        "call-out":     {"column": "contact_medium_c", "value": "call - out"},
        "outbound call":{"column": "contact_medium_c", "value": "call - out"},
        "call in":      {"column": "contact_medium_c", "value": "call-in"},
        "call-in":      {"column": "contact_medium_c", "value": "call-in"},
        "inbound call": {"column": "contact_medium_c", "value": "call-in"},
        "email":        {"column": "contact_medium_c", "value": "email"},
        "e-mail":       {"column": "contact_medium_c", "value": "email"},
        "e mail":       {"column": "contact_medium_c", "value": "email"},
        "live chat":    {"column": "contact_medium_c", "value": "live chat"},
        "chat":         {"column": "contact_medium_c", "value": "live chat"},
        "walk in":      {"column": "contact_medium_c", "value": "walk - in"},
        "walkin":       {"column": "contact_medium_c", "value": "walk - in"},
        "walk-in":      {"column": "contact_medium_c", "value": "walk - in"},
        "website":      {"column": "contact_medium_c", "value": "website"},
        "web":          {"column": "contact_medium_c", "value": "website"},
        "social media": {"column": "contact_medium_c", "value": "social media"},
        "social":       {"column": "contact_medium_c", "value": "social media"},
        "referral":     {"column": "contact_medium_c", "value": "referral"},

        # ── Lead source (main) ────────────────────────────────────────────────
        "digital":          {"column": "lead_source_c", "value": "Digital"},
        "direct":           {"column": "lead_source_c", "value": "Direct"},
        "direct walkin":    {"column": "lead_source_c", "value": "Direct Walkin"},
        "channel partner":  {"column": "lead_source_c", "value": "Channel Partner"},
        "bulk sale":        {"column": "lead_source_c", "value": "Bulk Sale"},
        "outbound campaign":{"column": "lead_source_c", "value": "Outbound Campaign"},
        "outdoor":          {"column": "lead_source_c", "value": "Outdoor"},
        "print media":      {"column": "lead_source_c", "value": "Print Media"},
        "electronic media": {"column": "lead_source_c", "value": "Electronic Media"},
        "existing customer":{"column": "lead_source_c", "value": "Existing Customer"},
        "lead reassigned":  {"column": "lead_source_c", "value": "Lead Reassigned"},
        "reference sale":   {"column": "lead_source_c", "value": "Reference Sale"},
        "referral sale":    {"column": "lead_source_c", "value": "Referral Sale"},
        "word of mouth":    {"column": "lead_source_c", "value": "Word of Mouth"},
        "unit shifting":    {"column": "lead_source_c", "value": "Unit Shifting"},
        "transfered":       {"column": "lead_source_c", "value": "Transfered"},

        # ── Lead sub-source ───────────────────────────────────────────────────
        "99 acres":    {"column": "lead_source_sub_category_c", "value": "99 Acres"},
        "magicbricks": {"column": "lead_source_sub_category_c", "value": "MagicBricks"},
        "magic bricks":{"column": "lead_source_sub_category_c", "value": "Magic Bricks"},
        "housing.com": {"column": "lead_source_sub_category_c", "value": "Housing.com"},
        "google":      {"column": "lead_source_sub_category_c", "value": "Google"},
        "facebook":    {"column": "lead_source_sub_category_c", "value": "Facebook"},
        "instagram":   {"column": "lead_source_sub_category_c", "value": "Instagram"},
        "youtube":     {"column": "lead_source_sub_category_c", "value": "YouTube"},
        "nobroker":    {"column": "lead_source_sub_category_c", "value": "NoBroker"},
        "no broker":   {"column": "lead_source_sub_category_c", "value": "NoBroker"},
        "inshorts":    {"column": "lead_source_sub_category_c", "value": "Inshorts"},
        "quora":       {"column": "lead_source_sub_category_c", "value": "Quora"},
        "spotify":     {"column": "lead_source_sub_category_c", "value": "Spotify"},
        "taboola":     {"column": "lead_source_sub_category_c", "value": "Taboola"},
        "mygate":      {"column": "lead_source_sub_category_c", "value": "Mygate"},
        "organic":     {"column": "lead_source_sub_category_c", "value": "Organic"},
        "sfmc":        {"column": "lead_source_sub_category_c", "value": "SFMC"},
        "adgebra":     {"column": "lead_source_sub_category_c", "value": "Adgebra"},
        "rcs":         {"column": "lead_source_sub_category_c", "value": "RCS"},

        # ── Project (project_c) keywords ──────────────────────────────────────
        "wave city":      {"column": "project_c", "value": "wave city"},
        "wmcc":           {"column": "project_c", "value": "wmcc"},
        "wmcc sec 32":    {"column": "project_c", "value": "wmcc sec 32"},
        "wmcc sector 32": {"column": "project_c", "value": "wmcc sec 32"},
        "wave estate":    {"column": "project_c", "value": "wave estate"},

        # ── Grouping-only keywords ────────────────────────────────────────────
        "by source":          {"column": "lead_source_c"},
        "source wise":        {"column": "lead_source_c"},
        "lead source wise":   {"column": "lead_source_c"},
        "by sub source":      {"column": "lead_source_sub_category_c"},
        "sub source wise":    {"column": "lead_source_sub_category_c"},
        "by project":         {"column": "project_c"},
        "project wise":       {"column": "project_c"},
        "by product":         {"column": "product_category_c"},
        "product wise":       {"column": "product_category_c"},
        "category wise":      {"column": "product_category_c"},
        "by status":          {"column": "status"},
        "status wise":        {"column": "status"},
        "by feedback":        {"column": "customer_feedback_c"},
        "feedback wise":      {"column": "customer_feedback_c"},
        "by rating":          {"column": "rating_c"},
        "rating wise":        {"column": "rating_c"},
        "by city":            {"column": "city_c"},
        "city wise":          {"column": "city_c"},
        "by owner":           {"column": "ownername_c"},
        "owner wise":         {"column": "ownername_c"},
        "by medium":          {"column": "contact_medium_c"},
        "medium wise":        {"column": "contact_medium_c"},
        "contact medium wise":{"column": "contact_medium_c"},
        "by property type":   {"column": "property_type_c"},
        "property type wise": {"column": "property_type_c"},
        "by property size":   {"column": "property_size_c"},
        "property size wise": {"column": "property_size_c"},
        "budget wise":        {"column": "budget_range_c"},
        "by budget":          {"column": "budget_range_c"},
        "customer code wise": {"column": "sap_customer_code_c"},

        # ── Aggregation keywords ──────────────────────────────────────────────
        "lead count":  {"aggregation": "count", "column": "lead_id_c"},
        "total lead":  {"aggregation": "count", "column": "lead_id_c"},
        "total leads": {"aggregation": "count", "column": "lead_id_c"},
        "how many":    {"aggregation": "count", "column": "lead_id_c"},
        "count":       {"aggregation": "count", "column": "lead_id_c"},
        "leads":       {"aggregation": "count", "column": "lead_id_c"},
    }

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
# QUERY TYPE ENUM  (same as inventory)
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
class LeadSQLRequest(BaseModel):
    question:  str  = Field(..., description="Natural language query about leads")
    catalog:   str  = Field(default=PRESTO_CATALOG)
    db_schema: str  = Field(default=PRESTO_SCHEMA)
    table:     str  = Field(default=PRESTO_TABLE)

class DateRange(BaseModel):
    start_date: str
    end_date:   Optional[str] = None
    label:      Optional[str] = None

class LeadSQLResponse(BaseModel):
    status:             str
    query_type:         str
    sql:                str
    schema_metadata:    list | None = Field(default=None, alias="schema")
    data:               list | None = None
    execution:          dict | None = None
    date_ranges:        List[DateRange]
    is_valid:           bool
    validation_message: Optional[str]          = None
    metadata:           Optional[Dict[str, Any]] = None
    intent_summary:     Optional[Dict[str, Any]] = None
    totals:             Optional[Dict[str, Any]] = None

    class Config:
        validate_by_name = True


# ============================================================================
# DATE PARSER  — identical to inventory (YYYYMMDD format, Apr–Mar FY)
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
        if 4 <= month <= 6:  return 1
        if 7 <= month <= 9:  return 2
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
        logger.debug(f"extract_date_tokens: text='{text}'")
        fy_year = r'(?:fy\s*)?(?:20\d{2}|\d{2})'  # matches: 2024, FY2024, FY 2024, FY24, fy 24
        
        return re.findall(
            r'\d{4}-\d{2}-\d{2}'
            r'|\d{1,2}[/-]\d{1,2}[/-]\d{4}'
            # 15 sep | 1st september | 11th mar 2024 | 11th mar FY2024 | 11th mar FY24
            rf'|\b\d{{1,2}}(?:st|nd|rd|th)?\s+'
            rf'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
            rf'|january|february|march|april|june|july|august|september|october|november|december)'
            rf'(?:,?\s*{fy_year})?\b'
            # sep 15 | september 1st | mar 3rd, 2025 | mar 3rd FY2025 | mar 3rd FY25
            rf'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
            rf'|january|february|march|april|june|july|august|september|october|november|december)'
            rf'\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*{fy_year})?\b'
            # month YYYY  (e.g. "april 2024", "january 2023") — must come BEFORE standalone month
            r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
            r'|january|february|march|april|june|july|august|september|october|november|december)'
            r'\s+(?:20\d{2})\b'
            # Standalone month names
            r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
            r'|january|february|march|april|june|july|august|september|october|november|december)\b'
            # Bare 4-digit year (e.g. 2023) and FY-prefixed year (e.g. fy2023, fy 2023, fy23)
            r'|\bfy\s*(?:20\d{2}|\d{2})\b'
            r'|\b20\d{2}\b',
            text,
            re.IGNORECASE
        )
    # def extract_date_tokens(text: str):
    #     print(text, '========================')
    #     return re.findall(
    #         r'\d{4}-\d{2}-\d{2}'
    #         r'|\d{1,2}[/-]\d{1,2}[/-]\d{4}'
    #         # 15 sep | 1st september | 11th mar 2024
    #         r'|\b\d{1,2}(?:st|nd|rd|th)?\s+'
    #         r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
    #         r'|january|february|march|april|june|july|august|september|october|november|december)'
    #         r'(?:,?\s*\d{4})?\b'
    #         # sep 15 | september 1st | mar 3rd, 2025
    #         r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
    #         r'|january|february|march|april|june|july|august|september|october|november|december)'
    #         r'\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b'
    #         # Standalone month names
    #         r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
    #         r'|january|february|march|april|june|july|august|september|october|november|december)\b',
    #         text,
    #         re.IGNORECASE
    #     )
    # def extract_date_tokens(text: str):
    #     print(text,'========================')
    #     return re.findall(
    #         r'\d{4}-\d{2}-\d{2}'
    #         r'|\d{1,2}[/-]\d{1,2}[/-]\d{4}'
    #         r'|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
    #         r'|january|february|march|april|june|july|august|september|october|november|december)'
    #         r'(?:,?\s*\d{4})?\b'
    #         r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
    #         r'|january|february|march|april|june|july|august|september|october|november|december)'
    #         r'\s+\d{1,2}(?:,?\s*\d{4})?\b'
    #         r'|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
    #         r'|january|february|march|april|june|july|august|september|october|november|december)\b',
    #         text, re.IGNORECASE
    #     )


    @staticmethod
    def parse_flexible_date(text: str, default_year=None):
        text = text.lower().strip()
        today = date.today()

        def resolve_year(mth):
            if default_year:
                return default_year
            current_fy = DateParser.get_current_fy(today)
            return current_fy if mth >= 4 else current_fy + 1

        # yyyy-mm-dd
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except ValueError:
            pass

        # dd/mm/yyyy or dd-mm-yyyy
        try:
            clean_text = text.replace("-", "/")
            return datetime.strptime(clean_text, "%d/%m/%Y").date()
        except ValueError:
            pass

        # 15 sep 2024 | 15 sep
        m = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})(?:\s+(\d{4}))?', text)
        if m:
            try:
                d = int(m.group(1))
                mth_name = m.group(2)
                mth = DateParser.MONTH_MAP.get(mth_name) or DateParser.MONTH_MAP.get(mth_name[:3])

                if not mth:
                    return None

                if m.group(3):
                    y = int(m.group(3))
                else:
                    y = resolve_year(mth)

                return date(y, mth, d)
            except:
                pass

        # month YYYY  (e.g. "april 2024") → first day of that month
        # Must come BEFORE the "sep 15" pattern to avoid "april 2024" being misread as day=20
        m = re.search(r'^([a-z]{3,9})\s+(20\d{2})$', text)
        if m:
            try:
                mth_name = m.group(1)
                mth = DateParser.MONTH_MAP.get(mth_name) or DateParser.MONTH_MAP.get(mth_name[:3])
                y = int(m.group(2))
                if mth:
                    return date(y, mth, 1)
            except:
                pass

        # sep 15, 2024 | sep 15
        # m = re.search(r'([a-z]{3,9})\s+(\d{1,2})(?:,\s*(\d{4}))?', text)
        m = re.search(r'([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?', text)
        if m:
            try:
                mth_name = m.group(1)
                mth = DateParser.MONTH_MAP.get(mth_name) or DateParser.MONTH_MAP.get(mth_name[:3])
                d = int(m.group(2))

                if not mth:
                    return None

                if m.group(3):
                    y = int(m.group(3))
                else:
                    y = resolve_year(mth)

                return date(y, mth, d)
            except:
                pass

        return None
    # def parse_flexible_date(text: str, default_year=None):
    #     if not text:
    #         return None
    #     text = text.lower().strip()
    #     today = date.today()
    #     current_fy = DateParser.get_current_fy(today)

    #     def resolve_year(mth):
    #         if default_year:
    #             return default_year
    #         return current_fy if mth >= 4 else current_fy + 1

    #     # yyyy-mm-dd
    #     try:
    #         return datetime.strptime(text, "%Y-%m-%d").date()
    #     except ValueError:
    #         pass

    #     # dd/mm/yyyy or dd-mm-yyyy
    #     try:
    #         return datetime.strptime(text.replace("-", "/"), "%d/%m/%Y").date()
    #     except ValueError:
    #         pass

    #     # 15 sep 2024 | 15 sep
    #     # m = re.search(r'(\d{1,2})\s+([a-z]{3,9})(?:\s+(\d{4}))?', text)
    #     m = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})(?:\s+(\d{4}))?', text)
    #     if m:
    #         try:
    #             d  = int(m.group(1))
    #             mth = DateParser.MONTH_MAP.get(m.group(2)) or DateParser.MONTH_MAP.get(m.group(2)[:3])
    #             y  = int(m.group(3)) if m.group(3) else resolve_year(mth)
    #             if mth:
    #                 return date(y, mth, d)
    #         except Exception:
    #             pass

    #     # sep 15, 2024 | sep 15
    #     # m = re.search(r'([a-z]{3,9})\s+(\d{1,2})(?:,\s*(\d{4}))?', text)
    #     m = re.search(r'([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?', text)
    #     if m:
    #         try:
    #             mth = DateParser.MONTH_MAP.get(m.group(1)) or DateParser.MONTH_MAP.get(m.group(1)[:3])
    #             d   = int(m.group(2))
    #             y   = int(m.group(3)) if m.group(3) else resolve_year(mth)
    #             if mth:
    #                 return date(y, mth, d)
    #         except Exception:
    #             pass

    #     return None

    @staticmethod
    def parse_from_date(
        query: str,
        today: date | None = None,
        fy_start_month: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Parses 'from / after / since' date expressions.
        Returns a DATE_RANGE dict with only start_date set.
        """

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
        
        original_day = day  # ✅ preserve original user input
        current_fy = DateParser.get_current_fy(today)

        # Case 1: Numeric date
        if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', month_part):
            clean = month_part.replace("/", "-")
            fmt = "%d-%m-%Y" if len(clean.split("-")[-1]) == 4 else "%d-%m-%y"
            dt = datetime.strptime(clean, fmt).date()

        # Case 2: Month name
        else:
            month = DateParser.MONTH_MAP.get(month_part[:3])
            if not month:
                return None
            fy_shift = 0
            if year:
                found_year = int(year)
            else:
                effective_fy = current_fy + fy_shift
                # Apr–Dec → FY year
                if month >= 4:
                    found_year = effective_fy
                else:
                    # Jan–Mar → FY+1
                    found_year = effective_fy + 1

            day = int(day) if day else 1
            dt = date(found_year, month, day)

        # 'after' means exclusive → next day
        if keyword == "after":
            is_numeric = re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', month_part)
            print(is_numeric)
            print(day)
            if original_day is None and not is_numeric:
                # Month-level input (e.g., "July 2024")
                last_day = calendar.monthrange(dt.year, dt.month)[1]
                dt = date(dt.year, dt.month, last_day) + timedelta(days=1)
            else:
                # Exact date input
                dt = dt + timedelta(days=1)

        return {
            "type": QueryType.DATE_RANGE,
            "start_date": DateParser.date_to_yyyymmdd(dt),
            "end_date": None,
            "label": f"From {dt.strftime('%d %b %Y')}"
        } 

    @staticmethod
    def parse_fy_till_date(q: str):
        if not q or not isinstance(q, str):
            return None

        q = q.lower().strip()
        print("parse_fy_till_date: q=", q)
        # Split on till / upto / up to
        parts = re.split(r'\b(?:till|upto|up to)\b', q, maxsplit=1)
        print("parse_fy_till_date: parts=", parts)
        if len(parts) != 2:
            return None

        left_part, right_part = parts[0].strip(), parts[1].strip()
        print("parse_fy_till_date: left_part=", left_part, "right_part=", right_part)

        # Handle an explicit start date with a rolling end date, e.g.
        # "1st june 2026 till date".
        if re.fullmatch(r'(date|today|now)', right_part, re.IGNORECASE):
            left_tokens = DateParser.extract_date_tokens(left_part)
            print("parse_fy_till_date: left_tokens=", left_tokens)
            if left_tokens:
                start_date = DateParser.parse_flexible_date(left_tokens[0])
                print("parse_fy_till_date: start_date=", start_date)
                if start_date:
                    end_date = date.today()
                    return {
                        "type": QueryType.DATE_RANGE,
                        "start_date": DateParser.date_to_yyyymmdd(start_date),
                        "end_date": DateParser.date_to_yyyymmdd(end_date),
                        "label": f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}",
                    }

        # If left side already contains a date,
        # treat it as a normal date range.
        if DateParser.extract_date_tokens(left_part):
            return None

        # =========================================================
        # CASE 1: "till date"
        # =========================================================
        if re.fullmatch(r'(date|today|now)', right_part, re.IGNORECASE):
            end_date = date.today()

            return {
                "type": QueryType.DATE_RANGE,
                "start_date": None,
                "end_date": DateParser.date_to_yyyymmdd(end_date),
                "label": f"Till {end_date.strftime('%d %b %Y')}"
            }

        # =========================================================
        # CASE 2: "till April 2025"
        # =========================================================
        end_date = DateParser.parse_flexible_date(right_part)

        if not end_date:
            return None

        # If only month + year is supplied,
        # use the LAST day of that month.
        month_year_match = re.fullmatch(
            r'(january|february|march|april|may|june|july|'
            r'august|september|october|november|december)\s+\d{4}',
            right_part,
            re.IGNORECASE
        )

        if month_year_match:
            import calendar

            last_day = calendar.monthrange(
                end_date.year,
                end_date.month
            )[1]

            end_date = date(
                end_date.year,
                end_date.month,
                last_day
            )

        return {
            "type": QueryType.DATE_RANGE,
            "start_date": None,
            "end_date": DateParser.date_to_yyyymmdd(end_date),
            "label": f"Till {end_date.strftime('%d %b %Y')}"
        }
    # def parse_fy_till_date(q: str):
    #     if not q or not isinstance(q, str):
    #         return None
    #     q = q.lower().strip()
    #     parts = re.split(r'\b(?:till|upto|up to)\b', q, maxsplit=1)
    #     if len(parts) != 2:
    #         return None
    #     left_part, right_part = parts[0].strip(), parts[1].strip()
    #     if DateParser.extract_date_tokens(left_part):
    #         return None
    #     end_date = DateParser.parse_flexible_date(right_part)
    #     print(end_date,'999999999999999')
    #     if not end_date:
    #         return None
    #     print(left_part,right_part)
    #     fy_start_year = end_date.year if end_date.month >= 4 else end_date.year - 1
    #     start_date    = date(fy_start_year, 4, 1)
    #     if start_date > end_date:
    #         return None
    #     return {
    #         "type":       QueryType.DATE_RANGE,
    #         "start_date": DateParser.date_to_yyyymmdd(start_date),
    #         "end_date":   DateParser.date_to_yyyymmdd(end_date),
    #         "label":      f"FY {fy_start_year} till {end_date.strftime('%d %b %Y')}",
    #     }

    @staticmethod
    def parse_fy_till_month(q: str):
        if not q or not isinstance(q, str):
            return None
        q = q.lower().strip()
        parts = re.split(r'\b(?:till|through|upto|up to)\b', q, maxsplit=1)
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
        # year_match = re.search(r'\b(20\d{2})\b', right_part)
        year_match = extract_fy(right_part)

        today      = date.today()
        current_fy = DateParser.get_current_fy(today)
        # year = int(year_match.group(1)) if year_match else (
        #     current_fy if month_num >= 4 else current_fy + 1
        # )

        year = int(year_match) if year_match else (
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

        # 🚫 Block month-year only (e.g. "sep 2024") but NOT "fy 2024" or range queries
        if re.search(r'\b(?!fy\b)([a-z]+)\s+\d{4}\b', q):
            if (not re.search(r'\b([1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\b', q)
                    and not re.search(r'\b(to|till|until|through|now|present|current|today)\b', q)):
                return None
        
        # ==================================================
        # 0️⃣ FY till-date (NEW FUNCTION CALL)
        # ==================================================
        logger.debug("parse_specific_date_or_range: checking parse_fy_till_date")
        fy_till = DateParser.parse_fy_till_date(q)
        if fy_till:
            return fy_till

        logger.debug("parse_specific_date_or_range: checking date range")

        # ==================================================
        # 1️⃣ DATE RANGE (HIGHEST PRIORITY)
        # ==================================================
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
            if range_match:
                raw_start = range_match.group(1).strip()
                raw_end = range_match.group(3).strip()
            else:
                raw_start = between_match.group(2).strip()
                raw_end = between_match.group(3).strip()

            # 🔥 Resolve "now / today / present / current" keywords to today's date
            _NOW_WORDS = {"now", "today", "present", "current", "date"}
            if raw_end.strip() in _NOW_WORDS:
                raw_end = today.strftime("%Y-%m-%d")
            if raw_start.strip() in _NOW_WORDS:
                raw_start = today.strftime("%Y-%m-%d")

            start_tokens = DateParser.extract_date_tokens(raw_start)
            end_tokens = DateParser.extract_date_tokens(raw_end)

            logger.debug(f"date range tokens: start={start_tokens} end={end_tokens}")
            start_text = start_tokens[0] if start_tokens else raw_start
            end_text = end_tokens[0] if end_tokens else raw_end
            logger.debug(f"date range text: start='{start_text}' end='{end_text}'")
            # Try parse end first (for propagation)
            end_date = DateParser.parse_flexible_date(end_text)
            # 🔥 HANDLE end month-only
            if not end_date:
                end_date = DateParser.parse_month_only(end_text, today.year, is_start=False)
            # 🔥 HANDLE bare year as end (e.g. "2024") → last day of that FY (31 Mar year+1)
            if not end_date:
                _yr_m = re.fullmatch(r'(20\d{2})', end_text.strip())
                if _yr_m:
                    _yr = int(_yr_m.group(1))
                    end_date = date(_yr + 1, 3, 31)

            if not end_date:
                return None

            # Try start
            start_date = DateParser.parse_flexible_date(start_text, default_year=end_date.year)
            # 🔥 HANDLE start month-only
            if not start_date:
                start_date = DateParser.parse_month_only(start_text, end_date.year, is_start=True)
            # 🔥 HANDLE bare year as start (e.g. "2023" or "fy2023") → April 1 of that FY year
            # Use re.search so it also works when start_text is a full sentence containing a year.
            if not start_date:
                _yr_m = re.search(r'\bfy\s*(20\d{2}|\d{2})\b|\b(20\d{2})\b', start_text.strip(), re.IGNORECASE)
                if _yr_m:
                    _raw = int(_yr_m.group(1) or _yr_m.group(2))
                    _yr = _raw if _raw > 100 else 2000 + _raw
                    start_date = date(_yr, 4, 1)

            logger.debug(f"date range resolved: start={start_date} end={end_date}")
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

        logger.debug("parse_specific_date_or_range: checking specific date")
        # ==================================================
        # 2️⃣ SPECIFIC DATE
        # ==================================================

        if any(k in q for k in ["after", "since"]):
            return None
        tokens = DateParser.extract_date_tokens(q)
        if not tokens:
            return None
        logger.debug(f"specific date tokens: {tokens}")
        parsed_date = DateParser.parse_flexible_date(tokens[0])
        if not parsed_date:
            return None

        logger.debug(f"parsed_date: {parsed_date}")

        return {
            "type": QueryType.SPECIFIC_DATE,
            "date": DateParser.date_to_yyyymmdd(parsed_date),
            "label": parsed_date.strftime("%d %B %Y")
        }
    # @staticmethod
    # def parse_specific_date_or_range(q: str):
    #     if not q or not isinstance(q, str):
    #         return None
    #     q = q.lower().strip()
    #     today = date.today()
    #     print("hello parse_specific_date_or_range")
    #     # if re.search(r'\b([a-z]+)\s+\d{4}\b', q):
    #     #     if not re.search(r'\b([1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\b', q):
    #     #         return None

    #     # 🚫 Block month-year only (e.g. "sep 2024") but NOT "fy 2024" or range queries
    #     if re.search(r'\b(?!fy\b)([a-z]+)\s+\d{4}\b', q):
    #         if (not re.search(r'\b([1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?\b', q)
    #                 and not re.search(r'\b(to|till|until|through|now|present|current|today)\b', q)):
    #             return None
    #     print("parse_fy_till_date, jdhjhj")
    #     fy_till = DateParser.parse_fy_till_date(q)
    #     if fy_till:
    #         return fy_till
        
    #     print("skip the fy_till")

    #     # range_match = re.search(r'(.+?)\s+(and|to|until|till|through|–|-)\s+(.+)', q)

    #     range_match = re.search(
    #         r'(.+?)\s+(to|until|till|through|–|-)\s+(.+)',
    #         q
    #     )

    #     between_match = re.search(
    #         r'(.+?)\s+between\s+(.+?)\s+and\s+(.+)',
    #         q,
    #         re.IGNORECASE
    #     )
    #     if range_match or between_match:
    #         print("enter the range match")
    #         raw_start = range_match.group(1).strip() if range_match else between_match.group(2).strip()
    #         raw_end   = range_match.group(3).strip() if range_match else between_match.group(3).strip()
    #         print(raw_start,raw_end)
    #         start_tokens = DateParser.extract_date_tokens(raw_start)
    #         print(start_tokens,"[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]")
    #         end_tokens   = DateParser.extract_date_tokens(raw_end)
    #         print("end_token",end_tokens)
    #         start_text = start_tokens[0] if start_tokens else raw_start
    #         end_text   = end_tokens[0]   if end_tokens   else raw_end
    #         end_date   = DateParser.parse_flexible_date(end_text)
    #         print(end_date,"88888888888888888888")
    #         if not end_date:
    #             return None
    #         start_date = DateParser.parse_flexible_date(start_text, default_year=end_date.year)
    #         print(start_date,end_date)

    #         if start_date and start_date > end_date:
    #             start_date = DateParser.parse_flexible_date(start_text, default_year=end_date.year - 1)
    #         if not start_date:
    #             day_match = re.search(r'\b([1-9]|[12][0-9]|3[01])\b', start_text)
    #             if day_match:
    #                 start_date = date(end_date.year, end_date.month, int(day_match.group(1)))
    #         if not start_date or start_date > end_date:
    #             return None
    #         return {
    #             "type":       QueryType.DATE_RANGE,
    #             "start_date": DateParser.date_to_yyyymmdd(start_date),
    #             "end_date":   DateParser.date_to_yyyymmdd(end_date),
    #             "label":      f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}",
    #         }

    #     if any(k in q for k in ["after", "since"]):
    #         return None

    #     tokens = DateParser.extract_date_tokens(q)
    #     if not tokens:
    #         return None
    #     parsed_date = DateParser.parse_flexible_date(tokens[0])
    #     if not parsed_date:
    #         return None
    #     return {
    #         "type":  QueryType.SPECIFIC_DATE,
    #         "date":  DateParser.date_to_yyyymmdd(parsed_date),
    #         "label": parsed_date.strftime("%d %B %Y"),
    #     }

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
        # year_match = re.search(r"\b(20\d{2})\b", q)
        # if year_match:
        #     target_fy = int(year_match.group(1))

        year_match = extract_fy(q)
        if year_match:
            target_fy = int(year_match)

        if re.search(r"\b(last|previous)\s+(year|fy)\b", q):
            target_fy = current_fy - 1


        # ✅ Quarter extraction
        q_pattern = r"(?:q(?:uarter|tr)?\s*([1-4]))"

        range_match = re.search(
            rf"\b{q_pattern}\s*(?:to|till|through|until|–|-)\s*{q_pattern}\b", q
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
#     today      = datetime.today()
#     current_fy = today.year if today.month >= 4 else today.year - 1

#     MONTH_MAP = DateParser.MONTH_MAP
#     month_pattern = (
#         r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
#         r"january|february|march|april|june|july|august|september|october|november|december)"
#     )
#     month_range_match = re.search(
#         rf"\b{month_pattern}\b\s*(to|–|-|till|until)\s*\b{month_pattern}\b", q
#     )
#     if not month_range_match:
#         return None

#     start_m_txt = month_range_match.group(1)
#     end_m_txt   = month_range_match.group(3)
#     start_month = MONTH_MAP[start_m_txt]
#     end_month   = MONTH_MAP[end_m_txt]

#     explicit_year = None
#     year_match    = re.search(r"\b(20\d{2})\b", q)
#     if year_match:
#         explicit_year = int(year_match.group(1))

#     if explicit_year:
#         start_year = explicit_year
#         end_year   = explicit_year
#         if start_month > end_month:
#             start_year -= 1
#     else:
#         start_year = current_fy if start_month >= 4 else current_fy + 1
#         end_year   = current_fy if end_month   >= 4 else current_fy + 1
#         if start_month > end_month:
#             end_year += 1

#     start_date = datetime(start_year, start_month, 1)
#     _, last_day = monthrange(end_year, end_month)
#     end_date   = datetime(end_year, end_month, last_day)

#     if re.search(r"\b(month\s*wise|monthly|mom|month\s*on\s*month|by\s*month)\b", q):
#         periods = []
#         m, y    = start_month, start_year
#         while (y < end_year) or (y == end_year and m <= end_month):
#             _, ld = monthrange(y, m)
#             s = datetime(y, m, 1)
#             e = datetime(y, m, ld)
#             periods.append({
#                 "label":      s.strftime("%b %Y"),
#                 "start_date": s.strftime("%Y%m%d"),
#                 "end_date":   e.strftime("%Y%m%d"),
#             })
#             m += 1
#             if m == 13:
#                 m = 1; y += 1
#         return {
#             "type":    QueryType.MONTH_RANGE_MONTH_WISE,
#             "periods": periods,
#             "label":   f"{start_m_txt.title()}–{end_m_txt.title()} Month-wise",
#         }

#     return {
#         "type":       QueryType.MONTH_RANGE,
#         "start_date": start_date.strftime("%Y%m%d"),
#         "end_date":   end_date.strftime("%Y%m%d"),
#         "label":      f"{start_m_txt.title()} to {end_m_txt.title()}",
#     }

# def parse_month_range_logic(q: str):
#     if not q:
#         return None

#     q = q.lower().strip()
#     today = datetime.today()

#     # Financial Year (Apr–Mar)
#     current_fy = today.year if today.month >= 4 else today.year - 1

#     MONTH_MAP = DateParser.MONTH_MAP

#     month_pattern = (
#         r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
#         r"january|february|march|april|june|july|august|september|october|november|december)"
#     )

#     if not re.search(r"\b(to|till|until|–)\b", q):
#         return None

#     print("enter the  parse_month_range_logic function")
#     # ✅ Named groups (fix wrong extraction issue)
#     month_range_match = re.search(
#         rf"\b(?P<start>{month_pattern})\b\s*(to|–|-|till|until)\s*\b(?P<end>{month_pattern})\b",
#         q
#     )
#     print("DEBUG:", q, month_range_match)
#     if not month_range_match:
#         return None

#     start_m_txt = month_range_match.group("start")
#     end_m_txt = month_range_match.group("end")

#     start_month = MONTH_MAP[start_m_txt]
#     end_month = MONTH_MAP[end_m_txt]

#     # Detect explicit year
#     explicit_year = None
#     year_match = re.search(r"\b(20\d{2})\b", q)
#     if year_match:
#         explicit_year = int(year_match.group(1))
    
#     is_last_year = bool(re.search(r"\b(last year|previous year)\b", q))
#     # =========================
#     # ✅ YEAR LOGIC (FIXED)
#     # =========================
#     if explicit_year:
#         start_year = explicit_year
#         end_year = explicit_year

#         # Cross calendar year (e.g., Nov → Feb)
#         if start_month > end_month:
#             end_year += 1

#     else:
#         # Anchor to Financial Year
#         base_fy = current_fy - 1 if is_last_year else current_fy
#         start_year = base_fy
#         if start_month >= 4:
#             # Start in FY start year
#             if end_month >= 4:
#                 # Same FY
#                 end_year = current_fy
#             else:
#                 # Cross year (Apr → Jan)
#                 end_year = current_fy + 1
#         else:
#             # Start month is Jan/Feb/Mar → belongs to next calendar year
#             start_year = current_fy + 1

#             if end_month >= 4:
#                 end_year = current_fy
#             else:
#                 end_year = current_fy + 1

#     # =========================
#     # DATE BUILDING
#     # =========================
#     start_date = datetime(start_year, start_month, 1)
#     _, last_day = monthrange(end_year, end_month)
#     end_date = datetime(end_year, end_month, last_day)

#     # =========================
#     # MONTH-WISE LOGIC
#     # =========================
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

#     # =========================
#     # NORMAL RANGE
#     # =========================
#     return {
#         "type": QueryType.MONTH_RANGE,
#         "start_date": start_date.strftime("%Y%m%d"),
#         "end_date": end_date.strftime("%Y%m%d"),
#         "label": f"{start_m_txt.title()} to {end_m_txt.title()}",
#     }

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
        rf"\s*(?:to|-|till|through|until|and)\s*(?P<end>{month_token_pattern})(?:\s+(?P<end_year>20\d{{2}}))?",
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
    
    if re.search(r"\b(after|till|through|to|upto|up to|on)\b", q):
        return None

    months_found = []
    current_fy   = DateParser.get_current_fy()
    explicit_year = extract_fy(q)
    

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
        year = fy
        # year_match = re.search(r"\b(20\d{2})\b", q)
        # year = int(year_match.group(1)) if year_match else fy
        year_match = extract_fy(q)
        if year_match:
            year = int(year_match)

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
    if not q:
        return None
    month_names = '|'.join(DateParser.MONTH_MAP.keys())
    if re.search(
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+[a-z]{3,9}\s+20\d{2}\b',
            q
        ):
        return None
    if re.search(
        rf'\b(?:{month_names})[a-z]*\s+20\d{{2}}\s*(?:and|to|till|through|–|-)\s*(?:{month_names})[a-z]*\s+20\d{{2}}\b',
        q, re.IGNORECASE
    ):
        return None
    year_matches = re.findall(r'\b(20\d{2})\b', q)
    if not year_matches or len(year_matches) < 2:
        return None
    years   = sorted(set(int(y) for y in year_matches))
    periods = []
    for fy in years:
        s, e = DateParser.get_fy_start_end(fy)
        periods.append({"label": f"FY{fy}", "start_date": s, "end_date": e})
    return {
        "type":    QueryType.YEAR_WISE,
        "years":   years,
        "periods": periods,
        "label":   " & ".join([f"FY{y}" for y in years]),
    }


def year_range_logic(q: str):
    if re.search(r'\b\d{1,2}\s+[a-z]{3,9}\s+20\d{2}\b', q):
        return None
    year_range_match = re.search(r'\b(20\d{2})\s*(?:to|till|through|–|-)\s*(20\d{2})\b', q)
    if year_range_match:
        start_year = int(year_range_match.group(1))
        end_year   = int(year_range_match.group(2))
        periods = []
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


# ============================================================================
# LLM INTENT PROMPT  — LLM extracts JSON only; Python does everything else
# ============================================================================
def build_lead_llm_prompt(question: str) -> str:
    schema_lines = "\n".join([
        f"  - {col} ({meta['type']}): {meta['description']}"
        for col, meta in LeadColumnMetadata.COLUMNS.items()
    ])

    return f"""
You are a strict JSON extraction engine for a lead management system.
Your ONLY job is to extract structured intent from a natural language query about leads.
You must NEVER infer, guess, or hallucinate values. Only extract what is explicitly stated.
You must NEVER generate SQL. Only return JSON.

=============================================================
TABLE CONTEXT
=============================================================
Table: lead_fy_report
Columns:
{schema_lines}

Date column: created_date_c (stored as YYYYMMDD integer — do NOT mention date_parse or format strings)

=============================================================
ABSOLUTE GROUND RULES
=============================================================
1. Return ONLY valid JSON wrapped in <JSON_RESPONSE> tags. No explanation, no markdown, no SQL.
2. NEVER include a field in "filters" unless the user explicitly mentioned a value for it.
3. NEVER add a column to "group_by" unless user explicitly asked to group/split/break down by it.
4. NEVER infer a date_hint unless the user explicitly mentioned a time period.
   - If no time reference → set date_hint to null.
5. NEVER put date columns (created_date_c, lastmodified_date_c) in "filters". Dates go ONLY in date_hint + date_column.
6. aggregation MUST always be a LIST, even when only one aggregation is mentioned.
7. If a field has no value to extract, omit it from filters entirely.

=============================================================
OUTPUT SCHEMA
=============================================================
{{
  "aggregation": [ <string> ],            // always a list; default ["lead_count"]
  "group_by":    [ <column_name> ],       // only explicitly requested groupings
  "filters":     {{ <column>: <value> }}, // only explicitly mentioned filter values
  "date_hint":   <string | null>,         // raw user phrase or null
  "date_column": "created_date_c" | "lastmodified_date_c"
}}

=============================================================
SECTION 1 — AGGREGATION MAPPING
=============================================================
Map user intent to exactly one of these tokens:

| User says                                              | Token          |
|--------------------------------------------------------|----------------|
| lead count / total lead / how many leads / count leads | "lead_count"   |
| leads                                                  | "lead_count"   |

- If aggregation is unclear or not mentioned → default to ["lead_count"]
- aggregation is ALWAYS a list.

=============================================================
SECTION 2 — FILTER COLUMN MAPPING
=============================================================

2A. LEAD STATUS (customer_feedback_c)
Triggers: "open leads", "discussion pending", "junk", "junk leads",
          "valid leads", "qualified leads", "interested",
          "disqualified", "not interested"
IMPORTANT: Use EXACT values (case-insensitive). Python will handle matching.
- "open leads"       → customer_feedback_c: "Discussion Pending"
- "junk" / "junk leads" → customer_feedback_c: "Junk"
- "valid leads"      → customer_feedback_c: "__valid__"  (special token — Python handles it)
- "qualified leads" / "interested" → customer_feedback_c: "Interested" (EXACT match, not substring)
- "disqualified" / "not interested" → customer_feedback_c: "Not Interested" (EXACT match, not substring)

2B. STATUS COLUMN (status)
Triggers: "new lead", "nurturing", "qualified lead", "unqualified", "disqualified lead"
- "new lead" / "new" (not "new plots") → status: "new"
- "nurturing" → status: "nurturing"
- "qualified lead" / "status qualified" → status: "qualified"
- "unqualified" / "status disqualified" → status: "unqualified"
IMPORTANT: If user says "new plots" → do NOT set status. That is a product filter.

2C. RATING (rating_c)
Triggers: "hot", "warm", "cold", "hot lead", "warm lead", "cold lead"
- "hot"  → rating_c: "Hot"
- "warm" → rating_c: "Warm"
- "cold" → rating_c: "Cold"

2D. CONTACT MEDIUM (contact_medium_c)
Triggers: "call out", "call in", "email", "live chat", "walk in",
          "website", "social media", "referral"
Value mapping:
- "call out" / "outbound" → "call - out"
- "call in" / "inbound"   → "call-in"
- "email"                 → "email"
- "live chat" / "chat"    → "live chat"
- "walk in" / "walkin"    → "walk - in"
- "website" / "web"       → "website"
- "social media" / "social" / "facebook" / "instagram" → "social media"
- "referral"              → "referral"

2E. LEAD SOURCE (lead_source_c) vs SUB-SOURCE (lead_source_sub_category_c)
Sub-source values: 99 acres, magicbricks, housing.com, google, facebook,
                   instagram, youtube, nobroker, inshorts, quora, spotify,
                   taboola, mygate, organic, sfmc, adgebra, rcs, live chat, etc.
Main source values: digital, direct, channel partner, bulk sale,
                    outbound campaign, outdoor, print media, referral,
                    word of mouth, electronic media, existing customer, etc.

Rule: If value matches sub-source list → lead_source_sub_category_c
      If value matches main source list → lead_source_c

2F. PRODUCT CATEGORY (product_category_c)
Trigger: user mentions a product name like: veridia, dream homes, eligo,
         eden, new plots, old plots, wave floor, prime floors, armonia villa,
         wave galleria, ews, lig, mayfair park, swamanorath, wave garden, etc.
OR if user says: "product", "product wise", "by product", "category wise"
→ product_category_c

NEVER use project_c in Product Mode.

2G. PROJECT (project_c)
Trigger: "wave city", "wmcc", "wave estate", "by project", "project wise"
→ project_c
Use LIKE matching (partial). NEVER use = for project_c.

2H. PROPERTY TYPE (property_type_c)
Triggers: "residential", "commercial", "renting", "buying","leasing"
- "residential" → "Residential"
- "commercial"  → "Commercial"
- "renting"     → "Renting"
- "buying"        → "Buying"
- "leasing"       → "Leasing"
NOTE: if user says "commercial" as a division type, use property_type_c.
      If it describes a product category, use product_category_c.

2I. PROPERTY SIZE (property_size_c)
Triggers: 1BHK, 2BHK, 3BHK, 4BHK, 5BHK, penthouse, skyvillas,
          commercial office space, commercial space, plots
- "1bhk" / "1 bhk" → "1BHK"
- "2bhk" / "2 bhk" → "2BHK"
- "3bhk" / "3 bhk" → "3BHK"
- "4bhk" / "4 bhk" → "4BHK"
- "5bhk" / "5 bhk" → "5BHK"
- "penthouse"       → "Penthouse"
- "skyvillas"       → "Skyvillas"

2J. APPOINTMENT (is_appointment_booked_c)
Triggers: "meeting booked", "appointment booked", "appointment"
→ is_appointment_booked_c: "1"

2K. OWNER NAME (ownername_c)
Triggers: a person's name mentioned in query
- Full name: exact value
- Partial name: use LIKE

2L. BUDGET (budget_range_c)
Triggers: "budget", "price", amount values like "3 cr", "50 lacs"
→ budget_range_c: value as written

2M. SAP CODE (sap_customer_code_c)
Triggers: "sap code", "customer code", "sap customer code"
→ sap_customer_code_c: the code value

2N. CITY (city_c)
Triggers: user mentions a city name (NOT a month name)
→ city_c: city name
IMPORTANT: NEVER extract month names (Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec)
           or month abbreviations (January, February, etc.) as city values.
           These belong in date_hint, not city_c filter.

2O. Disqualificatio reason
Trigger: user mentions a disqualification reason (e.g., "budget issue", "location issue","out of budget","not interested buying")
→ disqualification_reason_c: the reason value
IMPORTANT: If user ask lead disqualification reason, then set group_by: ["disqualification_reason_c"]


2O. VS COMPARISON (status column)
When user says "X vs Y" for two statuses (qualified vs unqualified):
→ set status: ["qualified", "unqualified"] as a list

=============================================================
SECTION 3 — DATE EXTRACTION
=============================================================
3A. date_column selection:
- Default: "created_date_c"
- Use "lastmodified_date_c" ONLY if user explicitly says "modified", "updated", "last modified"
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
  "month wise", "quarter wise", "project wise", "status wise",
  "source wise", "product wise", "category wise", "owner wise",
  "city wise", "medium wise", "rating wise", "budget wise", etc.
- NEVER infer group_by from filters.
- NEVER add a column to group_by just because it appears in filters.

=============================================================
SECTION 5 — WORKED EXAMPLES
=============================================================

Q: "Total leads this month"
A: {{"aggregation":["lead_count"],"group_by":[],"filters":{{}},"date_hint":"this month","date_column":"created_date_c"}}

Q: "Open leads by source last quarter"
A: {{"aggregation":["lead_count"],"group_by":["lead_source_c"],"filters":{{"customer_feedback_c":"Discussion Pending"}},"date_hint":"last quarter","date_column":"created_date_c"}}

Q: "Junk leads from facebook this year"
A: {{"aggregation":["lead_count"],"group_by":[],"filters":{{"customer_feedback_c":"Junk","lead_source_sub_category_c":"Facebook"}},"date_hint":"this year","date_column":"created_date_c"}}

Q: "Leads from veridia project wise last month"
A: {{"aggregation":["lead_count"],"group_by":["project_c"],"filters":{{"product_category_c":"veridia"}},"date_hint":"last month","date_column":"created_date_c"}}

Q: "Hot leads from wave city q1 month wise"
A: {{"aggregation":["lead_count"],"group_by":[],"filters":{{"rating_c":"Hot","project_c":"wave city"}},"date_hint":"q1 month wise","date_column":"created_date_c"}}

Q: "Lead count by status this quarter"
A: {{"aggregation":["lead_count"],"group_by":["status"],"filters":{{}},"date_hint":"this quarter","date_column":"created_date_c"}}

Q: "Qualified vs unqualified leads"
A: {{"aggregation":["lead_count"],"group_by":["status"],"filters":{{"status":["qualified","unqualified"]}},"date_hint":null,"date_column":"created_date_c"}}

Q: "Website leads month wise last 3 months"
A: {{"aggregation":["lead_count"],"group_by":[],"filters":{{"contact_medium_c":"website"}},"date_hint":"last 3 months","date_column":"created_date_c"}}

Q: "Total leads"
A: {{"aggregation":["lead_count"],"group_by":[],"filters":{{}},"date_hint":null,"date_column":"created_date_c"}}

Q: "Leads by owner year wise"
A: {{"aggregation":["lead_count"],"group_by":["ownername_c"],"filters":{{}},"date_hint":"year wise","date_column":"created_date_c"}}

Q: "Leads by nupoor aggarwal"
A: {{"aggregation":["lead_count"],"group_by":[],"filters":{{"ownername_c":"nupoor aggarwal"}},"date_hint":"","date_column":"created_date_c"}}

Q: "Show me leads in delhi"
A: {{"aggregation":["lead_count"],"group_by":[],"filters":{{"city_c":"delhi"}},"date_hint":"","date_column":"created_date_c"}}


=============================================================
NOW PROCESS THE FOLLOWING QUERY
=============================================================
User Query: "{question}"

Return ONLY the JSON object wrapped in <JSON_RESPONSE> tags. No other text.

<JSON_RESPONSE>
"""


# ============================================================================
# LLM INTENT DETECTOR  (same pattern as inventory)
# ============================================================================
class LLMIntentDetector:
    def extract_intent(self, question: str) -> Dict[str, Any]:
        prompt = build_lead_llm_prompt(question)
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
            "aggregation": ["lead_count"],
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

def mom_logic(q):
    today      = datetime.today()
    current_fy = DateParser.get_current_fy()

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

def last_n_year_mom_qoq_yoy(q):
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
# ============================================================================
# INTENT DETECTOR  — normalises LLM output + deterministic extraction
# ============================================================================
class LeadIntentDetector:
    def __init__(self):
        self.keywords    = LeadColumnMetadata.KEYWORD_MAPPING
        self.valid_values = LeadColumnMetadata.VALID_VALUES

    # ── Filter normalisation ──────────────────────────────────────────────────
    def normalize_filters(self, raw_filters: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        col_map = {k.lower(): k for k in LeadColumnMetadata.COLUMNS.keys()}
        alias_col_map = {
            "owner_name_c": "ownername_c",
            "owner_name": "ownername_c",
            "owner": "ownername_c",
            "city": "city_c",
            "city_name": "city_c",
            "city_name_c": "city_c",
        }

        for col, values in raw_filters.items():
            if not values:
                continue
            col_key = str(col).lower().strip()
            target_col = col_map.get(col_key, alias_col_map.get(col_key, col))

            # Redirection rules
            if target_col != "product_category_c":
                v_str = " ".join([str(v) for v in (values if isinstance(values, list) else [values])]).lower()
                for cat in LeadColumnMetadata.PRODUCT_CATEGORIES:
                    if cat in v_str:
                        target_col = "product_category_c"
                        break
            if target_col not in ("project_c", "product_category_c"):
                v_str = " ".join([str(v) for v in (values if isinstance(values, list) else [values])]).lower()
                for proj in LeadColumnMetadata.PROJECT_VALUES:
                    if proj in v_str:
                        target_col = "project_c"
                        break

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
                            normalized_values.append(possible)
                            found = True
                            break
                    if not found and len(val_str) > 3:
                        for possible in self.valid_values[target_col]:
                            if val_str in possible.lower() or possible.lower() in val_str:
                                normalized_values.append(possible)
                                found = True
                                break
                    if found:
                        continue

                # 3. Fallback — keep as-is
                normalized_values.append(val)

            if normalized_values:
                normalized.setdefault(target_col, [])
                normalized[target_col].extend(normalized_values)
                normalized[target_col] = list(set(normalized[target_col]))

        return normalized

    def is_status_requested(self,q: str):
            return bool(re.search(
                r"\b(status\s*wise|by\s+status|status\s+distribution|status\s+breakdown)\b",
                q.lower()
            ))
    # ── Full intent normalisation ─────────────────────────────────────────────
    def normalize_intent(self, raw_intent: Dict[str, Any], question: str) -> Dict[str, Any]:
        q = question.lower()

        # 1. Aggregation — always lead_count for leads
        normalized_agg = ["lead_count"]

        # 2. Filters
        raw_filters         = raw_intent.get("filters", {})
        normalized_filters  = self.normalize_filters(raw_filters)
        deterministic_filters = self.extract_filters(question)
        for col, values in deterministic_filters.items():
            if col not in normalized_filters:
                normalized_filters[col] = values
            else:
                normalized_filters[col] = list(set(normalized_filters[col] + values))

        if "ownername_c" in raw_filters and "ownername_c" in normalized_filters:
            # Convert both values to lists if they are strings
            raw_owner = raw_filters["ownername_c"]
            norm_owner = normalized_filters["ownername_c"]

            if not isinstance(raw_owner, list):
                raw_owner = [raw_owner]

            if not isinstance(norm_owner, list):
                norm_owner = [norm_owner]

            # Merge and remove duplicates
            normalized_filters["ownername_c"] = list(set(norm_owner + raw_owner))
        else:
            # Remove ownername_c if it is present in only one of them
            normalized_filters.pop("ownername_c", None)

        # 3. Group by
        normalized_groupby = self.extract_groupby(question)
        if not normalized_groupby:
            raw_groupby = raw_intent.get("group_by", [])
            if isinstance(raw_groupby, str):
                raw_groupby = [raw_groupby]
            col_map     = {k.lower(): k for k in LeadColumnMetadata.COLUMNS.keys()}
            date_cols   = {"created_date_c", "lastmodified_date_c"}
            for g in raw_groupby:
                g_lower   = str(g).lower().strip()
                found_col = None
                if g_lower in self.keywords:
                    found_col = self.keywords[g_lower].get("column")
                if not found_col:
                    found_col = col_map.get(g_lower)
                if found_col and found_col.lower() not in date_cols:
                               # 🚫 Block unwanted status grouping
                    if found_col.lower() == "status":
                        if not self.is_status_requested(question):
                            continue
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
        filters = {}
        question_lower = question.lower()

        for keyword, mapping in self.keywords.items():
            if "column" in mapping and "value" in mapping:
                if re.search(rf"\b{re.escape(keyword)}\b", question_lower):
                    col = mapping["column"]
                    val = mapping["value"]
                    filters.setdefault(col, []).append(val)

        # Owner name detection
        name_match = re.search(
            r'\b(?:by|from|of|handled by|created by|owner)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            question
        )
        if name_match:
            filters.setdefault("ownername_c", []).append(name_match.group(1))

        # Budget detection
        budget_match = re.search(
            r'\b(\d+(?:\.\d+)?\s*(?:cr|crore|lac|lacs|lakh|lakhs|k)[\s\-to]+\d*(?:\.\d+)?\s*(?:cr|crore|lac|lacs|lakh|lakhs|k)?)\b',
            question_lower
        )
        if budget_match:
            filters.setdefault("budget_range_c", []).append(budget_match.group(1))

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
        """Extract group by columns from question"""
        question_lower = question.lower()
        group_by = []

        # Lead Source
        if re.search(r'\b(by source|source wise|lead source wise)\b', question_lower):
            group_by.append("lead_source_c")

        # Sub Source
        if re.search(r'\b(by sub source|sub source wise|sub-source wise)\b', question_lower):
            group_by.append("lead_source_sub_category_c")

        # Project
        if re.search(r'\b(by project|project wise|per project|projectwise)\b', question_lower):
            group_by.append("project_c")

        # Product / Category
        if re.search(r'\b(by product|product wise|category wise|all product|per product|productwise)\b', question_lower):
            group_by.append("product_category_c")

        # Status
        if re.search(r'\b(by status|status wise|per status)\b', question_lower):
            group_by.append("status")

        # Feedback
        if re.search(r'\b(by feedback|feedback wise)\b', question_lower):
            group_by.append("customer_feedback_c")

        # Rating
        if re.search(r'\b(by rating|rating wise)\b', question_lower):
            group_by.append("rating_c")

        # City
        if re.search(r'\b(by city|city wise|per city)\b', question_lower):
            group_by.append("city_c")

        # Owner
        if re.search(r'\b(by owner|owner wise|per owner)\b', question_lower):
            group_by.append("ownername_c")

        # Medium
        if re.search(r'\b(by medium|medium wise|contact medium wise|per medium)\b', question_lower):
            group_by.append("contact_medium_c")

        # Property Type
        if re.search(r'\b(by property type|property type wise|per property type)\b', question_lower):
            group_by.append("property_type_c")

        # Property Size
        if re.search(r'\b(by property size|property size wise|per property size)\b', question_lower):
            group_by.append("property_size_c")

        # Budget
        if re.search(r'\b(by budget|budget wise|per budget)\b', question_lower):
            group_by.append("budget_range_c")

        # Customer Code
        if re.search(r'\b(by customer code|customer code wise|per customer code)\b', question_lower):
            group_by.append("sap_customer_code_c")

        return group_by

    # ── Date intent detection — identical cascade to inventory ────────────────
    def detect_date_intent(self, question: str):
        q = question.lower()
        logger.info(f"Detecting date intent: {q}")

        NUMBER_WORDS = {
            "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
            "seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,
            "thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
            "seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,
        }

        month_range_match     = re.search(
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*(?:to|till|through|-|–)\s*"
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*", q
        )
        last_n_year_match     = re.search(r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+years?\b", q)
        last_n_month_match    = re.search(r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b", q)
        last_n_quarter_match  = re.search(r"\blast\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+quarters?\b", q)
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

        # last quarter + mom
        mom_last_quarter = parse_quarter_mom(q)
        if mom_last_quarter:
            return mom_last_quarter

        # 2️⃣ QoQ
        if is_qoq and not last_n_quarter_match and not last_n_year_match:
            print("hello------------")
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
                if q_num == 1:   s, e = datetime(target_fy,4,1), datetime(target_fy,6,30)
                elif q_num == 2: s, e = datetime(target_fy,7,1), datetime(target_fy,9,30)
                elif q_num == 3: s, e = datetime(target_fy,10,1),datetime(target_fy,12,31)
                else:            s, e = datetime(target_fy+1,1,1),datetime(target_fy+1,3,31)
                quarters.append({
                    "quarter":    f"Q{q_num} FY{target_fy}",
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date":   DateParser.date_to_yyyymmdd(e.date()),
                })
            return {"type": QueryType.QUARTER_WISE, "fy": target_fy, "quarters": quarters,
                    "label": f"Quarter-wise FY{target_fy}"}

        # 3️⃣ Specific quarters ((Q1 to Q3, Q1 and Q4,) + mom,yoy,qoq etc.)
        quarter_intent = DateParser.parse_quarter_intent(q)
        if quarter_intent:
            return quarter_intent

        # 4️⃣ Month range (jan to march, month wise)
        month_logic = parse_month_range_logic(q)
        if month_logic:
            return month_logic

        # 5️⃣ This quarter vs last quarter comparison
        if "this quarter vs last quarter" in q or "this quarter compared to last quarter" in q:
            this_q     = DateParser.get_fy_quarter(today.month)
            target_fy  = current_fy
            last_q     = this_q - 1 if this_q > 1 else 4
            last_q_fy  = target_fy if this_q > 1 else target_fy - 1

            def gqd(q_num, fy):
                if q_num == 1:   return datetime(fy,4,1), datetime(fy,6,30)
                elif q_num == 2: return datetime(fy,7,1), datetime(fy,9,30)
                elif q_num == 3: return datetime(fy,10,1),datetime(fy,12,31)
                else:            return datetime(fy+1,1,1),datetime(fy+1,3,31)

            ts, te = gqd(this_q, target_fy)
            ls, le = gqd(last_q, last_q_fy)
            return {
                "type": QueryType.MULTI_DATE_RANGE,
                "ranges": [
                    {"start_date": DateParser.date_to_yyyymmdd(ts.date()), "end_date": DateParser.date_to_yyyymmdd(te.date()), "label": f"This Quarter FY{target_fy} Q{this_q}"},
                    {"start_date": DateParser.date_to_yyyymmdd(ls.date()), "end_date": DateParser.date_to_yyyymmdd(le.date()), "label": f"Last Quarter FY{last_q_fy} Q{last_q}"},
                ],
                "label": f"This Quarter vs Last Quarter",
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
                    "label": "This Week"}

        # 9️⃣ Last N days
        if ("last" in q or "previous" in q) and "day" in q:
            words = q.split(); n = None
            for i, word in enumerate(words):
                if word in ("last","previous") and i + 1 < len(words):
                    nxt = words[i+1]
                    if nxt == "day":   n = 1; break
                    if nxt.isdigit():  n = int(nxt); break
                    if nxt in NUMBER_WORDS: n = NUMBER_WORDS[nxt]; break
            if n:
                end   = today.date() - timedelta(days=1)
                start = end - timedelta(days=n - 1)
                return {"type": QueryType.LAST_N_DAYS,
                        "start_date": DateParser.date_to_yyyymmdd(start),
                        "end_date":   DateParser.date_to_yyyymmdd(end),
                        "label": "Yesterday" if n == 1 else f"Last {n} Days"}

        # 🔟 Last N weeks
        if ("last" in q or "previous" in q) and "week" in q :
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
                                "label": f"Last {n} Week{'s' if n>1 else ''}"}
                    except Exception:
                        pass

        # 1️⃣1️⃣ This month
        if "this month" in q or "current month" in q:
            start = today.date().replace(day=1)
            return {"type": QueryType.THIS_MONTH,
                    "start_date": DateParser.date_to_yyyymmdd(start),
                    "end_date":   DateParser.today_yyyymmdd(),
                    "label": f"This Month - {today.strftime('%B %Y')}"}

        # 1️⃣2️⃣ Last month
        if "last month" in q or "previous month" in q:
            first_of_this  = today.date().replace(day=1)
            last_day_prev  = first_of_this - timedelta(days=1)
            first_day_prev = last_day_prev.replace(day=1)
            return {"type": QueryType.LAST_MONTH,
                    "start_date": DateParser.date_to_yyyymmdd(first_day_prev),
                    "end_date":   DateParser.date_to_yyyymmdd(last_day_prev),
                    "label": f"Last Month - {first_day_prev.strftime('%B %Y')}"}

        # 1️⃣3️⃣ MoM full FY
        if is_mom and not last_n_month_match and not last_n_quarter_match and not last_n_year_match:
            return mom_logic(q)

        # 1️⃣4️⃣ Last N months (with optional MoM)
        if last_n_month_match and not is_qoq and not is_yoy:
            return last_n_mom_logic(q)

        # 1️⃣5️⃣ This quarter
        if "this quarter" in q or "current quarter" in q:
            cq = DateParser.get_fy_quarter(today.month)
            if cq == 1:   s = datetime(current_fy,4,1)
            elif cq == 2: s = datetime(current_fy,7,1)
            elif cq == 3: s = datetime(current_fy,10,1)
            else:         s = datetime(current_fy+1,1,1)
            return {"type": QueryType.THIS_QUARTER,
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date":   DateParser.today_yyyymmdd(),
                    "label": f"This Quarter FY{current_fy} Q{cq}"}

        # 1️⃣6️⃣ Last quarter
        if "last quarter" in q or "previous quarter" in q:
            cq   = DateParser.get_fy_quarter(today.month)
            lq   = cq - 1 if cq > 1 else 4
            lqfy = current_fy if cq > 1 else current_fy - 1
            if lq == 1:   s, e = datetime(lqfy,4,1), datetime(lqfy,6,30)
            elif lq == 2: s, e = datetime(lqfy,7,1), datetime(lqfy,9,30)
            elif lq == 3: s, e = datetime(lqfy,10,1),datetime(lqfy,12,31)
            else:         s, e = datetime(lqfy+1,1,1),datetime(lqfy+1,3,31)
            return {"type": QueryType.LAST_QUARTER,
                    "start_date": DateParser.date_to_yyyymmdd(s.date()),
                    "end_date":   DateParser.date_to_yyyymmdd(e.date()),
                    "label": f"Last Quarter FY{lqfy} Q{lq}"}

        # 1️⃣7️⃣ Last N quarters
        if last_n_quarter_match and not is_yoy:
            return last_n_quarte_mom_qoq(q)

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

        if last_n_year_match:
            return last_n_year_mom_qoq_yoy(q)

        # 2️⃣2️⃣ This year / this FY
        if re.search(r"\b(this|current|present)\s*(fy|fiscal\s*year|financial\s*year|year)?\b",q):
            s, _ = DateParser.get_fy_start_end(current_fy)
            return {"type": QueryType.THIS_YEAR, "start_date": s,
                    "end_date": DateParser.today_yyyymmdd(),
                    "label": f"FY{current_fy} (YTD)"}

        # 2️⃣3️⃣ Last year / previous year
        if re.search(r"\b(last|previous|prev)\s*(fy|fiscal\s*year|financial\s*year|year)?\b", q):
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

        found_year = None
        # year_match = re.search(r'\b(20\d{2})\b', q)
        year_match = extract_fy(q)
        if year_match:
            # found_year = int(year_match.group(1))
            found_year = int(year_match)

        if found_month:
            fy_shift = -1 if re.search(r"\b(last|previous|prev)\s*(fy|fiscal\s*year|year)?\b", q) else 0
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
# SQL GENERATOR  — programmatic SQL builder (no LLM involved)
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


class LeadSQLGenerator:

    @staticmethod
    def build_select_clause(agg_infos: List[Dict], group_by: List[str]) -> str:
        select_parts = []
        if group_by:
            select_parts.extend(group_by)
        for agg in agg_infos:
            agg_type  = agg["type"]
            agg_col   = agg["column"]
            agg_alias = agg["alias"]
            if agg_type == AggregationType.COUNT:
                select_parts.append(f'COUNT(DISTINCT {agg_col}) AS "{agg_alias}"')
            elif agg_type == AggregationType.SUM:
                select_parts.append(f'SUM({_numeric_expr(agg_col)}) AS "{agg_alias}"')
            elif agg_type == AggregationType.AVG:
                select_parts.append(f'AVG({agg_col}) AS "{agg_alias}"')
            elif agg_type == AggregationType.MIN:
                select_parts.append(f'MIN({agg_col}) AS "{agg_alias}"')
            elif agg_type == AggregationType.MAX:
                select_parts.append(f'MAX({agg_col}) AS "{agg_alias}"')
        return "SELECT " + ", ".join(select_parts)
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
            # Convert YYYYMMDD date range to date_parse format
            # date_filter format: "created_date_c BETWEEN 20260401 AND 20260331"
            # Convert to: "TRY(date_parse(created_date_c, '%d-%m-%Y')) BETWEEN date '2026-04-01' AND date '2026-03-31'"
            
            converted_filter = LeadSQLGenerator._convert_date_filter(date_filter)
            conditions.append(converted_filter)


        for col, values in filters.items():
            col_meta  = LeadColumnMetadata.COLUMNS.get(col, {})
            col_type  = col_meta.get("type", "VARCHAR")
            col_conditions = []
            value_list = values if isinstance(values, list) else [values]

            for v in value_list:
                v_str = str(v).lower().strip()

                # Special token: valid leads = NOT junk
                if v_str == "__valid__":
                    col_conditions.append(
                        f"(lower(COALESCE({col}, '')) != 'junk')"
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

                # is_appointment_booked_c — exact match
                if col == "is_appointment_booked_c":
                    col_conditions.append(f"{col} = '{v}'")
                    continue

                # customer_feedback_c — EXACT MATCH (not LIKE)
                # Use exact match to avoid "Interested" matching "Not Interested"
                if col == "customer_feedback_c":
                    col_conditions.append(f"LOWER({col}) = '{v_str}'")
                    continue

                # Owner name — exact for full name, LIKE for partial
                if col == "ownername_c":
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

                # budget_range_c — LIKE with both no-space and space variants
                if col == "budget_range_c":
                    col_conditions.append(
                        f"(LOWER({col}) LIKE '%{v_str}%' OR LOWER({col}) LIKE '%{v_str.replace(' ', '')}%')"
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
    def build_group_by_clause(group_by: List[str]) -> str:
        return "GROUP BY " + ", ".join(group_by) if group_by else ""

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
    ) -> str:
        select_parts  = []
        group_by_parts = []

        # Period column (month / quarter / year)
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

        # User-requested group_by columns
        select_parts.extend(group_by)
        group_by_parts.extend(group_by)

        # Aggregation
        for agg in agg_infos:
            if agg["type"] == AggregationType.COUNT:
                agg_expr = f'COUNT(DISTINCT {agg["column"]}) AS "{agg["alias"]}"'
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

        select_clause   = "SELECT " + ", ".join(select_parts)
        from_clause     = f'FROM "{catalog}"."{schema}"."{table}"'

        # Date filter — YYYYMMDD integer BETWEEN (no date_parse)
        date_filter = None
        if date_range:
            start, end = date_range
            if start and end:
                date_filter = f"{date_column} BETWEEN {start} AND {end}"
            elif start:
                date_filter = f"{date_column} >= {start}"
            elif end:
                date_filter = f"{date_column} <= {end}"

        where_clause    = LeadSQLGenerator.build_where_clause(filters=filters, date_filter=date_filter)
        group_by_clause = ("GROUP BY " + ", ".join(group_by_parts)) if group_by_parts else ""

        sql = "\n".join(filter(None, [select_clause, from_clause, where_clause, group_by_clause]))
        return sql.strip()


# ============================================================================
# SQL VALIDATOR
# ============================================================================
class LeadSQLValidator:
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
    TEMPORAL_COLS = {"period","fy_year","month","quarter","year"}
    sql = sql.rstrip(";").strip()
    if re.search(r'\bORDER BY\b', sql, re.IGNORECASE):
        def fix_order(m):
            parts    = [p.strip() for p in m.group(1).split(",")]
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
            r'\bAS\s+"?(lead_count|total_lead|count|leads)\b"?',
            sql, re.IGNORECASE
        )
        if alias_match:
            sql += f' ORDER BY "{alias_match.group(1)}" DESC'
    return sql


# ============================================================================
# POST-PROCESSING: add totals row
# ============================================================================
NON_ADDITIVE_MARKERS   = ["%", ":"]
NON_ADDITIVE_KEYS      = {"fy_year","month","quarter","days","year","financial year","fiscal year"}
_TIME_DIMENSION_WORDS  = {"year","month","quarter","week","day","fiscal"}

def _is_additive_key(key: str) -> bool:
    key_lower = key.lower()
    if key_lower in NON_ADDITIVE_KEYS:
        return False
    if any(w in key_lower for w in _TIME_DIMENSION_WORDS):
        return False
    return not any(m in key for m in NON_ADDITIVE_MARKERS)

def add_total_row(data: list) -> list:
    SKIP_COLS = set()
    if not data or len(data) <= 1:
        return data
    total_row       = {}
    first_str_done  = False
    for key in data[0].keys():
        if key in SKIP_COLS:
            total_row[key] = None
            continue
        col_values  = [row.get(key) for row in data]
        non_null    = [v for v in col_values if v is not None]
        if not _is_additive_key(key):
            total_row[key] = "-" if first_str_done else ("Total" if not first_str_done else "-")
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
# MAIN ENGINE  — mirrors inventory's NLToSQLEngine exactly
# ============================================================================
class LeadNLToSQLEngine:
    def __init__(self):
        self.llm_detector   = LLMIntentDetector()
        self.intent_detector = LeadIntentDetector()
        self.sql_generator  = LeadSQLGenerator()
        self.sql_validator  = LeadSQLValidator()

    @staticmethod
    def _escape_sql_literal(v: str) -> str:
        return str(v).replace("'", "''")

    @staticmethod
    def _is_month_token(v: str) -> bool:
        token = re.sub(r"[^a-z]", "", str(v).lower())
        return token in DateParser.MONTH_MAP

    def _value_exists_for_owner_or_city(
        self,
        request: LeadSQLRequest,
        col: str,
        raw_value: Any,
        cache: Dict[Tuple[str, str], Optional[bool]],
    ) -> Optional[bool]:
        val = str(raw_value).strip()
        if not val:
            return False

        cache_key = (col, val.lower())
        if cache_key in cache:
            return cache[cache_key]

        esc = self._escape_sql_literal(val.lower())
        if col == "ownername_c" and " " in val:
            cond = f"LOWER(TRIM({col})) = '{esc}'"
        else:
            like_val = self._escape_sql_literal(val.lower().replace(" ", "%"))
            cond = f"LOWER(TRIM({col})) LIKE '%{like_val}%'"

        sql = (
            f'SELECT 1 AS ok FROM "{request.catalog}"."{request.db_schema}"."{request.table}" '
            f"WHERE {col} IS NOT NULL AND {cond} LIMIT 1"
        )
        rows, _, err = run_presto_query(sql)
        if err:
            logger.warning(f"Value validation failed for {col}='{val}': {err}")
            cache[cache_key] = None
            return None

        exists = bool(rows)
        cache[cache_key] = exists
        return exists

    def _sanitize_owner_city_filters(
        self,
        filters: Dict[str, Any],
        question: str,
        date_intent: Optional[Dict[str, Any]],
        request: LeadSQLRequest,
    ) -> Dict[str, Any]:
        if not filters:
            return filters

        q_lower = question.lower()
        has_date_context = bool(date_intent) or bool(
            re.search(r"\b(today|yesterday|week|month|quarter|year|fy|last|this|current)\b", q_lower)
        )
        target_cols = {"ownername_c", "city_c"}
        cache: Dict[Tuple[str, str], Optional[bool]] = {}
        sanitized: Dict[str, Any] = {}
        
        # Month names to filter out from city_c only
        month_names = {
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        }

        for col, values in filters.items():
            if col not in target_cols:
                sanitized[col] = values
                continue

            value_list = values if isinstance(values, list) else [values]
            kept = []
            for v in value_list:
                v_str = str(v).strip()
                if not v_str:
                    continue

                # ✅ STRATEGY:
                # For ownername_c: Trust LLM-extracted values (don't validate)
                # For city_c: Only drop obvious month name hallucinations; keep other values
                
                if col == "ownername_c":
                    # Keep all explicitly extracted owner names — don't validate
                    logger.info(f"Keeping LLM-extracted owner: {v_str}")
                    kept.append(v)
                    continue

                # For city_c: Check for month name hallucinations only
                if col == "city_c" and v_str.lower() in month_names:
                    logger.info(f"Dropping month name from {col} filter: {v_str}")
                    continue

                # For city_c: Try to validate, but be lenient
                exists = self._value_exists_for_owner_or_city(request, col, v_str, cache)
                if exists is True:
                    kept.append(v)
                    continue
                if exists is False:
                    logger.info(f"Warning: {col} value may not exist: {v_str} — keeping it anyway for user intent")
                    kept.append(v)  # Keep it anyway
                    continue

                # DB validation failed: drop only obvious date/month hallucinations.
                if has_date_context and self._is_month_token(v_str):
                    logger.info(f"Dropping probable date-token hallucination in {col}: {v_str}")
                    continue
                kept.append(v)

            if kept:
                sanitized[col] = list(dict.fromkeys(kept))

        return sanitized

    def process(self, request: LeadSQLRequest) -> LeadSQLResponse:
        question = request.question.strip()
        logger.info(f"Processing question: {question}")

        # ── Step 1: LLM extracts JSON intent (filters, group_by, date_hint) ──
        llm_intent = self.llm_detector.extract_intent(question)
        # ── Step 2: Deterministic normalisation locks hallucinations ───────────
        llm_intent = self.intent_detector.normalize_intent(llm_intent, question)
        logger.info(f"Normalized Intent: {llm_intent}")

        # ── Step 3: Python deterministically parses date intent ────────────────
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
        agg_map = {
            "lead_count": {"type": AggregationType.COUNT, "column": "lead_id_c", "alias": "Lead Count"},
        }
        aggregations = llm_intent.get("aggregation", ["lead_count"])
        agg_infos    = []
        for agg in aggregations:
            key = agg.strip().lower().replace(" ", "_")
            if key in agg_map:
                agg_infos.append(agg_map[key])
            else:
                logger.warning(f"Unknown aggregation: {agg}")
        if not agg_infos:
            agg_infos.append(agg_map["lead_count"])

        # ── Step 5: Filters and group_by from normalised intent ────────────────
        filters  = llm_intent.get("filters", {})
        group_by = llm_intent.get("group_by", [])
        filters  = self._sanitize_owner_city_filters(filters, question, date_intent, request)
        logger.info(f"Resolved filters: {filters}")
        logger.info(f"Resolved group_by: {group_by}")

        # ── Step 6: Date column validation ────────────────────────────────────
        date_column = llm_intent.get("date_column", "created_date_c")
        if date_column not in LeadColumnMetadata.DATE_COLUMNS:
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

                sql = LeadSQLGenerator.generate_sql(
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

            final_sql = "\n\nUNION ALL\n\n".join(sqls)
            metric_alias = agg_infos[0]["alias"] if agg_infos else "Lead Count"
            final_sql = f"(\n{final_sql}\n) ORDER BY \"{metric_alias}\" DESC"
            if group_by:
                final_sql += ", " + ", ".join(group_by)

        else:
            # Single period
            final_sql = LeadSQLGenerator.generate_sql(
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
            metric_alias = agg_infos[0]["alias"] if agg_infos else "Lead Count"
            if group_by:
                final_sql += f"\nORDER BY \"{metric_alias}\" DESC"

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

        return LeadSQLResponse(
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
app    = FastAPI(title="Lead NL-to-SQL Engine (Inventory Architecture)", version="1.0.0")
engine = LeadNLToSQLEngine()


@app.post("/generate-sql", response_model=LeadSQLResponse)
async def generate_sql(request: LeadSQLRequest):
    try:
        return engine.process(request)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", "date_format": "YYYYMMDD integer"}


if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 70)
    logger.info("LEAD NL-TO-SQL ENGINE — INVENTORY ARCHITECTURE")
    logger.info("LLM: intent JSON only | Python: date parsing + SQL generation")
    logger.info("Date format: YYYYMMDD integer (no date_parse)")
    logger.info("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8001)

