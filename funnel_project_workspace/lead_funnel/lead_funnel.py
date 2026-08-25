from fastapi import FastAPI, Query
import pandas as pd
import prestodb
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from typing import Optional
import os
from dotenv import load_dotenv
import numpy as np
import logging
from datetime import datetime, timedelta,date
from fastapi import Body
import re
from dateutil import parser as date_parser
from calendar import monthrange
from typing import List, Tuple, Any, Dict
from typing import Union

# Fields containing these markers will NOT be summed
NON_ADDITIVE_MARKERS = ["%", ":"]


def _is_additive_key(key: str) -> bool:
    """
    Decide whether a column/metric should be summed.
    """
    return not any(marker in key for marker in NON_ADDITIVE_MARKERS)


def _normalize_to_rows(data: Any) -> List[Dict[str, Any]]:
    """
    Normalize different response shapes into a list of rows.
    Supported:
    - List[Dict]  -> SQL-style output
    - Dict[str, Dict] -> Funnel-style output (nested)
    - Dict[str, Any] -> Single flat dict (convert to list)
    """
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if isinstance(data, dict):
        # Handle empty dict
        if not data:
            return []
        
        # Check if it's a nested dict (all values are dicts)
        values = list(data.values())
        if values and all(isinstance(v, dict) for v in values):
            # Nested format: { "User1": {...}, "User2": {...} }
            return values
        else:
            # Flat format: { "Total Leads": 100, "Valid Leads": 80, ... }
            # Treat it as a single row
            return [data]

    return []


def calculate_master_totals(data: Any) -> Dict[str, Union[int, float]]:
    """
    MASTER aggregation logic:
    - Column-wise totals
    - Sums ALL additive numeric fields
    - Ignores % and ratio fields
    - Works for ALL result shapes you showed
    """
    # Handle None or empty data
    if data is None or (isinstance(data, (dict, list)) and not data):
        return {}
    
    rows = _normalize_to_rows(data)
    totals: Dict[str, float] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if not _is_additive_key(key):
                continue

            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value

    # Clean numeric formatting
    final_totals = {}
    for k, v in totals.items():
        if float(v).is_integer():
            final_totals[k] = int(v)
        else:
            final_totals[k] = round(v, 2)

    return final_totals

# --------------------------------------------
# Logging Configuration
# --------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("lead_funnel_log.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------------------------
# Load Environment Variables
# --------------------------------------------
load_dotenv()

# --------------------------------------------
# Watsonx + Presto Configuration
# --------------------------------------------
CATALOG = os.getenv("CATALOG","salesforcereport")
FUNNEL_SCHEMA = os.getenv("FUNNEL_SCHEMA","funnel")
FUNNEL_TABLE = os.getenv("FUNNEL_TABLE","sales_funnel_monthly")

hostname = os.getenv("PRESTO_HOST")
portnumber = int(os.getenv("PRESTO_PORT"))
username = os.getenv("PRESTO_USERNAME")
password = os.getenv("PRESTO_PASSWORD")

# Watsonx Foundation Model credentials (optional)
creds = Credentials(
    url=os.getenv("WATSONX_URL"),
    api_key=password
)

model = ModelInference(
    model_id=os.getenv("MODEL_ID"),
    credentials=creds,
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params={"temperature": 0, "max_new_tokens": 300}
)

# --------------------------------------------
# FastAPI Setup
# --------------------------------------------
app = FastAPI(title="Watsonx Source-Wise Funnel Analytics Tool")

# --------------------------------------------
# Presto Query Helper with Logging
# --------------------------------------------
def query_presto(catalog: str, schema: str, sql: str) -> pd.DataFrame:
    """Run query on Watsonx.data Presto with detailed logging."""
    logger.info(f"Executing Presto query on catalog '{catalog}' and schema '{schema}'...")
    logger.debug(f"SQL Query:\n{sql}")

    try:
        conn = prestodb.dbapi.connect(
            host=hostname,
            port=portnumber,
            user=username,
            catalog=catalog,
            schema=schema,
            http_scheme="https",
            auth=prestodb.auth.BasicAuthentication(username, password)
        )
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        logger.info(f"Query executed successfully — {len(df)} rows fetched from '{schema}'.")
        return df

    except Exception as e:
        logger.error(f"Error executing query on catalog '{catalog}' schema '{schema}': {e}", exc_info=True)
        raise

# --------------------------------------------
# Source-Wise Funnel (Dynamic with Events)
# --------------------------------------------
def compute_source_wise_funnel(leads: pd.DataFrame):
    """
    Compute funnel metrics from the monthly funnel table and return a clean dictionary.
    This works with your existing CSV/data structure.
    """
    print("Starting dynamic source-wise funnel computation...")
    print(f"Leads DataFrame shape: {leads.shape}")
    
    if leads is None or leads.empty:
        logger.error("Leads DataFrame is None or empty.")
        return {"error": "leads data is missing"}
    
    # We'll compute overall funnel (since your data is already aggregated monthly)
    # Sum up if multiple rows (e.g., multiple months), or just take values if single row
    total_leads = leads["total_leads"].sum()
    valid_leads = leads["valid_leads"].sum()
    junk_leads = leads["junk_leads"].sum()
    sol_leads = leads["sol_leads_interested"].sum()
    meeting_booked = leads["meeting_booked"].sum()
    meeting_done = leads["meeting_done"].sum()
    sales_done = leads["sales_done"].sum()

    # Junk %
    junk_percent = round((junk_leads / total_leads * 100), 1) if total_leads > 0 else 0.0
    # Conversion ratios (handle division by zero)
    def safe_div(numerator, denominator):
        return round(numerator / denominator, 2) if denominator > 0 else 0.0

    tl_vl = safe_div(total_leads, valid_leads)
    # vl_sol = safe_div(sol_leads, valid_leads)
    vl_sol = safe_div(valid_leads, sol_leads)
    sol_mb = safe_div(sol_leads, meeting_booked)
    mb_md = safe_div(meeting_booked, meeting_done)
    md_sd = safe_div(meeting_done, sales_done)
    tl_sd = safe_div(total_leads, sales_done)
    vl_sd = safe_div(valid_leads, sales_done)
    sol_sd = safe_div(sol_leads, sales_done)
    mb_sd = safe_div(meeting_booked, sales_done)

    # Final clean dictionary output
    funnel_data = {
        "Total Leads": int(total_leads),
        "Valid Leads": int(valid_leads),
        "Junk Leads": int(junk_leads),
        "SOL Leads (Interested)": int(sol_leads),
        "Meeting Booked": int(meeting_booked),
        "Meeting Done": int(meeting_done),
        "Sales Done": int(sales_done),
        "Junk %": f"{junk_percent}%",
        "TL:VL": tl_vl,
        "VL:SOL": vl_sol,
        "SOL:MB": sol_mb,
        "MB:MD": mb_md,
        "MD:SD": md_sd,
        "TL:SD": tl_sd,
        "VL:SD": vl_sd,
        "SOL:SD": sol_sd,
        "MB:SD": mb_sd
    }

    logger.info("Funnel computation completed successfully.")
    return funnel_data

def sort_funnel_by_numeric_desc(data: Any, return_as_list: bool = False) -> Any:
    """
    Sort nested funnel dictionaries by numeric values in descending order.
    Works with various funnel output structures:
    - Dict[user, Dict[metric, value]] -> Sorts users by total/first numeric metric
    - Dict[project, Dict[user, Dict[metric, value]]] -> Sorts projects and users
    
    Args:
        data: The funnel data to sort
        return_as_list: If True, returns a list of objects with 'name' key instead of dict
                       This ensures order is preserved when transmitted through JSON APIs
    """
    if not isinstance(data, dict) or not data:
        return data
    
    # Check if this is a nested structure (all values are dicts)
    first_value = next(iter(data.values()))
    
    if isinstance(first_value, dict):
        # Check if it's doubly nested (project -> user -> metrics)
        first_inner_value = next(iter(first_value.values()), None) if first_value else None
        
        if isinstance(first_inner_value, dict):
            # Doubly nested: Sort projects, then sort users within each project
            sorted_data = {}
            for key in sorted(data.keys()):
                sorted_data[key] = sort_funnel_by_numeric_desc(data[key], return_as_list=return_as_list)
            return sorted_data
        else:
            # Single nested: Sort by first numeric metric value
            def get_sort_key(item):
                key, metrics = item
                # Find first numeric value in metrics dict
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)) and not any(marker in metric_name for marker in ["%", ":"]):
                        return -metric_value  # Negative for descending order
                return 0
            
            sorted_items = sorted(data.items(), key=get_sort_key)
            
            # Return as list if requested (preserves order in JSON)
            if return_as_list:
                return [
                    {"name": name, **metrics}
                    for name, metrics in sorted_items
                ]
            
            return dict(sorted_items)
    
    return data



def normalize(text: str):
    return text.lower().replace(",", " ").replace("-", " ").replace("  ", " ").strip()

def extract_year_from_text(text):
    for part in text.split():
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None

def extract_month_by_name(text):
    months = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }
    for word in text.split():
        if word in months:
            return months[word]
    return None

def parse_single_date(q: str | None) -> date | None:
    """
    Parse single date forms like:
      - '15 april 2024'
      - '15 april'
      - '5th june 23'
      - '15/04/2024' or '15-04-2024'
    Returns a datetime.date or None.
    """
    if not q or not isinstance(q, str):
        return None

    original_q = q
    q = q.strip().lower()

    # First try: DD/MM/YYYY or DD-MM-YYYY
    slash_match = re.match(r'^(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})$', q)
    if slash_match:
        day, month, year = map(int, slash_match.groups())
        if year < 100:
            year += 2000
        try:
            return datetime(year, month, day).date()
        except ValueError:
            return None

    # Natural language: 15 april 2024, 5th june 23, etc.
    m = re.search(
        r'\b([0-3]?\d)(?:st|nd|rd|th)?\s+'
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|'
        r'nov(?:ember)?|dec(?:ember)?)'
        r'(?:[,\s]+(20\d{2}|\d{2}))?\b',
        q
    )
    if not m:
        return None

    day = int(m.group(1))
    month_name = m.group(2)
    year_part = m.group(3)

    month = extract_month_by_name(month_name)
    if not month:
        return None

    if year_part:
        year = int(year_part)
        if len(year_part) == 2:
            year += 2000
    else:
        fy = get_current_fy()
        year = fy if month >= 4 else fy + 1

    try:
        return datetime(year, month, day).date()
    except ValueError:
        return None

def get_current_fy():
    today = datetime.today()
    fy_start = today.year if today.month >= 4 else today.year - 1
    return fy_start

def get_fy_quarter(m):
    if 4 <= m <= 6:   return 1
    if 7 <= m <= 9:   return 2
    if 10 <= m <= 12: return 3
    return 4

def detect_last_n_days(question: str):
    text = question.lower().strip()

    # --------------------------------------------
    # 1. LAST YEAR / PREVIOUS FY (Highest Priority)
    # --------------------------------------------
    last_year_patterns = [
        r"last\s+year\b",
        r"previous\s+year\b",
        r"last\s+financial\s+year\b",
        r"last\s+fy\b",
        r"previous\s+fy\b",
        r"last\s+f\.?y\.?\b"
    ]

    multi_year_patterns = [
        (r"last\s+(\d+)\s+years?", "years_number"),
        (r"last\s+(one|two|three|four|five)\s+years?", "years_word"),
    ]

    today = datetime.today()
    current_fy = today.year if today.month >= 4 else today.year - 1
    current_q = get_fy_quarter(today.month)

    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5
    }

    for pattern in last_year_patterns:

        if re.search(pattern, text):
            last_fy = current_fy - 1
            start_date = f"01-04-{last_fy}"
            end_date = f"31-03-{last_fy + 1}"

            return {
                "type": "last_financial_year",
                "fy": last_fy,
                "label": f"Last Financial Year (FY{last_fy})",
                "start_date": start_date,
                "end_date": end_date
            }
    
    for pattern, ptype in multi_year_patterns:
        match = re.search(pattern, text)
        if match:
            # Convert word to number
            if ptype == "years_number":
                n = int(match.group(1))
            else:
                n = word_to_num[match.group(1)]

            # Current FY
            today = datetime.today()
            current_fy = today.year if today.month >= 4 else today.year - 1

            # Last N full FY ranges
            start_fy = current_fy - n   # N financial years back
            end_fy = current_fy - 1     # Last completed FY

            start_date = f"01-04-{start_fy}"
            end_date = f"31-03-{end_fy + 1}"

            return {
                "type": "last_n_financial_years",
                "years": n,
                "label": f"Last {n} Financial Years (FY{start_fy}–FY{end_fy})",
                "start_date": start_date,
                "end_date": end_date
            }

    # --------------------------------------------
    # 2. Keywords Patterns
    # --------------------------------------------
    patterns = [
        (r"last\s+(\d+)\s+days?", "days"),
        (r"past\s+(\d+)\s+days?", "days"),

        (r"last\s+(\d+)\s+months?", "months"),
        (r"last\s+(one|two|three|four|five|six)\s+months?", "months_word"),

        (r"last\s+1\s+month\b", "month_single"),
        (r"last\s+one\s+month\b", "month_single"),
        (r"last\s+month\b", "month_single"),

        (r"last\s+week\b", "week"),
        (r"past\s+week\b", "week"),

        (r"last\s+quarter\b", "quarter"),
        (r"past\s+quarter\b", "quarter"),

        (r"last\s+(\d+)\s+quarters?", "quarters_number"),
        (r"last\s+(one|two|three|four)\s+quarters?", "quarters_word"),
    ]

    word_to_num = {
        "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6
    }

    # Helper: compute last N full quarters
    def compute_last_n_quarters(n):
        end_q = current_q - 1
        end_fy = current_fy

        if end_q <= 0:
            end_q = 4
            end_fy -= 1

        start_q = end_q - (n - 1)
        start_fy = end_fy

        while start_q <= 0:
            start_q += 4
            start_fy -= 1

        q_dates = {
            1: ("01-04", "30-06"),
            2: ("01-07", "30-09"),
            3: ("01-10", "31-12"),
            4: ("01-01", "31-03")
        }

        start_day, start_month = q_dates[start_q][0].split("-")
        end_day, end_month = q_dates[end_q][1].split("-")

        start_date = f"{start_day}-{start_month}-{start_fy}"
        end_date = f"{end_day}-{end_month}-{end_fy}"

        return {
            "type": "last_n_quarters",
            "label": f"Last {n} Quarters (Q{start_q} FY{start_fy} → Q{end_q} FY{end_fy})",
            "start_date": start_date,
            "end_date": end_date
        }

    # --------------------------------------------
    # 3. PATTERN PROCESS LOOP
    # --------------------------------------------
    for pattern, ptype in patterns:
        match = re.search(pattern, text)
        if not match:
            continue

        # --- Last Month (full previous month) ---
        if ptype == "month_single":
            first_of_this_month = today.replace(day=1)
            last_day_prev = first_of_this_month - timedelta(days=1)
            first_day_prev = last_day_prev.replace(day=1)

            return {
                "type": "last_full_month",
                "label": f"Last Month ({first_day_prev.strftime('%b %Y')})",
                "start_date": first_day_prev.strftime("%d-%m-%Y"),
                "end_date": last_day_prev.strftime("%d-%m-%Y")
            }

        # --- Last Week (Mon–Sun) ---
        if ptype == "week":
            days_back = today.weekday() + 7
            last_monday = today - timedelta(days=days_back)
            last_sunday = last_monday + timedelta(days=6)

            return {
                "type": "last_full_week",
                "label": "Last Week (Mon–Sun)",
                "start_date": last_monday.strftime("%d-%m-%Y"),
                "end_date": last_sunday.strftime("%d-%m-%Y")
            }

        # --- Last N Quarters (number) ---
        if ptype == "quarters_number":
            return compute_last_n_quarters(int(match.group(1)))

        # --- Last N Quarters (word) ---
        if ptype == "quarters_word":
            return compute_last_n_quarters(word_to_num.get(match.group(1), 1))

        # --- Last Quarter ---
        if ptype == "quarter":
            return compute_last_n_quarters(1)

        # --- Last N Full Months (word) ---
        if ptype == "months_word":
            n = word_to_num.get(match.group(1), 1)

            first_of_this_month = today.replace(day=1)
            end_date_dt = first_of_this_month - timedelta(days=1)
            start_date_dt = end_date_dt.replace(day=1)

            for _ in range(n - 1):
                start_date_dt = (start_date_dt.replace(day=1) - timedelta(days=1)).replace(day=1)

            return {
                "type": "last_n_full_months",
                "months": n,
                "label": f"Last {n} Months (Full Months)",
                "start_date": start_date_dt.strftime("%d-%m-%Y"),
                "end_date": end_date_dt.strftime("%d-%m-%Y")
            }

        # --- Last N Full Months (numeric) ---
        if ptype == "months":
            n = int(match.group(1))

            first_of_this_month = today.replace(day=1)
            end_date_dt = first_of_this_month - timedelta(days=1)
            start_date_dt = end_date_dt.replace(day=1)

            for _ in range(n - 1):
                start_date_dt = (start_date_dt.replace(day=1) - timedelta(days=1)).replace(day=1)

            return {
                "type": "last_n_full_months",
                "months": n,
                "label": f"Last {n} Months (Full Months)",
                "start_date": start_date_dt.strftime("%d-%m-%Y"),
                "end_date": end_date_dt.strftime("%d-%m-%Y")
            }

        # --- Last N Days ---
        if ptype == "days":
            days = int(match.group(1))
            start_date = (today - timedelta(days=days - 1)).strftime("%d-%m-%Y")
            end_date = today.strftime("%d-%m-%Y")

            return {
                "type": "last_n_days",
                "days": days,
                "label": f"Last {days} Days",
                "start_date": start_date,
                "end_date": end_date
            }

    return None

def detect_this_period(question: str):
    text = question.lower().strip()
    today = datetime.today()
    # Determine current FY
    current_fy = today.year if today.month >= 4 else today.year - 1

    # Helper: Get current quarter
    current_q = get_fy_quarter(today.month)

    # ===================================================================
    # 1. THIS MONTH
    # ===================================================================
    if re.search(r"\bthis\s+month\b|\bcurrent\s+month\b", text):
        start = today.replace(day=1).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        return {
            "type": "this_month",
            "label": f"This Month (MTD) – {today.strftime('%b %Y')}",
            "start_date": start,
            "end_date": end
        }

    # ===================================================================
    # 2. THIS QUARTER
    # ===================================================================
    if re.search(r"\bthis\s+quarter\b|\bcurrent\s+quarter\b|\bqtd\b", text):
        print(current_q, current_fy, "===============================")
        if current_q == 1:
            start = f"{current_fy}-04-01"
            end   = f"{current_fy}-06-30"
        elif current_q == 2:
            start = f"{current_fy}-07-01"
            end   = f"{current_fy}-09-30"
        elif current_q == 3:
            start = f"{current_fy}-10-01"
            end   = f"{current_fy}-12-31"
        else:  # current_q == 4
            start = f"{current_fy + 1}-01-01"
            end   = f"{current_fy + 1}-03-31"

        return {
            "type": "this_quarter",
            "label": f"This Quarter (Q{current_q} FY{current_fy})",
            "start_date": start,
            "end_date": end
        }
    # ===================================================================
    # 3. THIS YEAR / THIS FY
    # ===================================================================
    if re.search(r"\bthis\s+year\b|\bthis\s+fy\b|\bcurrent\s+year\b|\bcurrent\s+fy\b", text):
        start = f"{current_fy}-04-01"
        end = today.strftime("%Y-%m-%d")
        return {
            "type": "this_fy",
            "label": f"This Financial Year (FY{current_fy} YTD)",
            "start_date": start,
            "end_date": end
        }

    # ===================================================================
    # 4. THIS WEEK (Monday to Today)
    # ===================================================================
    if re.search(r"\bthis\s+week\b|\bcurrent\s+week\b", text):
        monday = today - timedelta(days=today.weekday())
        start = monday.strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        return {
            "type": "this_week",
            "label": "This Week (Mon–Today)",
            "start_date": start,
            "end_date": end
        }
    return None

def detect_yoy(question: str):
    text = question.lower().strip()

    # Trigger keywords for YoY
    trigger_keywords = [
        "yoy", "year on year", "year-on-year", "year over year",
        "last 3 years", "last three years", "past 3 years",
        "yoy performance", "yearly comparison"
    ]

    if not any(kw in text for kw in trigger_keywords):
        return None

    from datetime import datetime
    today = datetime.today()
    current_fy = today.year if today.month >= 4 else today.year - 1

    # We want last 3 COMPLETED financial years
    # Example: Today = Nov 2025 → Current FY = 2025 → Completed = FY22, FY23, FY24
    latest_completed_fy = current_fy - 1
    years = [
        latest_completed_fy - 2,  # e.g., FY22
        latest_completed_fy - 1,  # e.g., FY23
        latest_completed_fy,      # e.g., FY24
        latest_completed_fy + 1   # e.g., FY25 (current FY, optional)
    ]

    logger.info(f"YoY detected → Comparing last 3 FYs: {years}")

    def fy_dates(fy_year: int):
        return f"{fy_year}-04-01", f"{fy_year + 1}-03-31"

    yoy_periods = []
    for fy in years:
        start, end = fy_dates(fy)
        yoy_periods.append({
            "year": f"FY{fy}",
            "start_date": start,
            "end_date": end
        })

    return {
        "type": "yoy",
        "years": years,
        "periods": yoy_periods
    }

def detect_qoq(question: str):
    import re
    from datetime import datetime

    text = question.lower().strip()

    # ----------------------------------------
    # 1) Detect user intent (QOQ / Quarterly)
    # ----------------------------------------
    trigger_keywords = [
        "qoq", "quarter on quarter", "quarter-wise", "quarter wise",
        "quater wise", "quarterly", "quarterwise"
    ]
    if not any(kw in text for kw in trigger_keywords):
        return None

    # ----------------------------------------
    # 2) Determine current FY
    # ----------------------------------------
    today = datetime.today()
    current_fy = today.year if today.month >= 4 else today.year - 1

    # ----------------------------------------
    # 3) Extract explicit year (2023, 2024…)
    # ----------------------------------------
    year_match = re.search(r"\b(20\d{2})\b", text)
    explicit_year = int(year_match.group(1)) if year_match else None

    # ----------------------------------------
    # 4) Extract FY format like "FY24" or "fy2025"
    # ----------------------------------------
    fy_match = re.search(r"fy\s?(\d{2,4})", text)
    explicit_fy = None
    if fy_match:
        fy_value = fy_match.group(1)
        if len(fy_value) == 2:
            explicit_fy = int("20" + fy_value)      # fy24 → 2024
        else:
            explicit_fy = int(fy_value)             # fy2024 → 2024

    # ----------------------------------------
    # 5) Detect LAST YEAR / PREVIOUS FY logic
    # ----------------------------------------
    if "last year" in text or "previous year" in text or "last fy" in text or "previous fy" in text:
        target_fy = current_fy - 1

    elif explicit_year:
        target_fy = explicit_year

    elif explicit_fy:
        target_fy = explicit_fy

    else:
        # default → current FY
        target_fy = current_fy

    logger.info(f"QOQ → Using FY{target_fy}")

    # ----------------------------------------
    # 6) Build Q1–Q4 of FY
    # ----------------------------------------
    def quarter_dates(q, fy_year):
        if q == 1:
            return f"{fy_year}-04-01", f"{fy_year}-06-30"
        elif q == 2:
            return f"{fy_year}-07-01", f"{fy_year}-09-30"
        elif q == 3:
            return f"{fy_year}-10-01", f"{fy_year}-12-31"
        elif q == 4:
            return f"{fy_year + 1}-01-01", f"{fy_year + 1}-03-31"

    quarters = []
    for q in range(1, 5):
        start, end = quarter_dates(q, target_fy)
        quarters.append({
            "quarter": f"Q{q} FY{target_fy}",
            "start_date": start,
            "end_date": end
        })

    return {
        "type": "quarter_wise",
        "fy": target_fy,
        "quarters": quarters
    }

def detect_mom(question: str):
    import re
    from datetime import datetime
    from calendar import monthrange

    q = question.lower().strip()

    # --------------------------------------------------
    # 1) MOM INTENT CHECK
    # --------------------------------------------------
    mom_keywords = [
        "mom",
        "month on month",
        "month-on-month",
        "monthly",
        "month wise",
        "month over month"
    ]

    if not any(k in q for k in mom_keywords):
        return None

    today = datetime.today()

    # --------------------------------------------------
    # 2) FINANCIAL YEAR (APR–MAR)
    # --------------------------------------------------
    current_fy = today.year if today.month >= 4 else today.year - 1

    FY_QUARTERS = {
        1: (4, 6),    # Apr–Jun
        2: (7, 9),    # Jul–Sep
        3: (10, 12),  # Oct–Dec
        4: (1, 3)     # Jan–Mar
    }

    # --------------------------------------------------
    # 3) MONTH NORMALIZATION
    # --------------------------------------------------
    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12
    }

    # --------------------------------------------------
    # 4) EXTRACT YEAR / FY
    # --------------------------------------------------
    year_match = re.search(r"\b(20\d{2})\b", q)
    specified_year = int(year_match.group(1)) if year_match else None

    fy_match = re.search(r"\bfy\s?(\d{2})\b", q)
    specified_fy = int("20" + fy_match.group(1)) if fy_match else None

    # --------------------------------------------------
    # 5) EXPLICIT MONTH RANGE
    # --------------------------------------------------
    month_range_regex = (
        r"\b(" + "|".join(month_map.keys()) + r")\b\s*"
        r"(to|till|-|and)\s*"
        r"\b(" + "|".join(month_map.keys()) + r")\b"
    )

    m_range = re.search(month_range_regex, q)

    # --------------------------------------------------
    # 6) QUARTER DETECTION
    # --------------------------------------------------
    q_match = re.search(r"\bq([1-4])\b", q)
    quarter = int(q_match.group(1)) if q_match else None

    print(f"Detected → specified_year: {specified_year}, specified_fy: {specified_fy}, month_range: {m_range.groups() if m_range else None}, quarter: {quarter}")

    is_last_quarter = (
        "last quarter" in q or
        "previous quarter" in q
    )

    # --------------------------------------------------
    # 7) RESOLVE TARGET MONTH RANGE
    # --------------------------------------------------
    if m_range:
        sm = month_map[m_range.group(1)]
        em = month_map[m_range.group(3)]
        fy = specified_fy or specified_year or current_fy
        year = fy if sm >= 4 else fy + 1

    elif is_last_quarter:
        # Determine current FY start year
        current_fy = today.year if today.month >= 4 else today.year - 1

        # Determine current quarter inside FY (Apr–Mar)
        if 4 <= today.month <= 6:
            curr_q = 1
        elif 7 <= today.month <= 9:
            curr_q = 2
        elif 10 <= today.month <= 12:
            curr_q = 3
        else:
            curr_q = 4  # Jan–Mar

        # Determine last quarter
        if curr_q == 1:
            last_q = 4
            fy = current_fy - 1
        else:
            last_q = curr_q - 1
            fy = current_fy

        FY_QUARTERS = {
            1: (4, 6),
            2: (7, 9),
            3: (10, 12),
            4: (1, 3)
        }

        sm, em = FY_QUARTERS[last_q]

        # Year handling
        if last_q == 4:
            year = fy + 1   # Jan–Mar belongs to next calendar year
        else:
            year = fy

    elif quarter:
        print(q)
        fy = current_fy
        if "last year" in q or "previous year" in q:
            fy = current_fy - 1
        # If user gives FY explicitly (e.g. fy24)
        
        print(f"Explicit FY detected: {specified_fy}, Explicit Year detected: {specified_year} , {fy}")
        if specified_fy:
            fy = specified_fy

        # If user gives calendar year (e.g. 2024)
        elif specified_year:
            fy = specified_year  # treat as FY start year


        sm, em = FY_QUARTERS[quarter]

        # Year handling for calendar year mapping
        if quarter == 4:
            year = fy + 1  # Jan–Mar belongs to next calendar year
        else:
            year = fy

    elif (
        "last year" in q or
        "previous year" in q or
        "previous fy" in q
    ):
        fy = current_fy - 1
        sm, em = 4, 3
        year = fy

    elif specified_fy or specified_year:
        fy = specified_fy or specified_year
        sm, em = 4, 3
        year = fy

    else:
        # Default → current FY till today
        fy = current_fy
        sm = 4
        em = today.month
        year = fy

    # --------------------------------------------------
    # 8) GENERATE MONTH-WISE PERIODS
    # --------------------------------------------------
    periods = []

    y = year
    m = sm

    while True:
        _, last_day = monthrange(y, m)

        start_date = f"01-{m:02d}-{y}"
        end_date = f"{last_day:02d}-{m:02d}-{y}"

        label = datetime(y, m, 1).strftime("%b %Y")

        # Apply MTD only for current FY current month
        if fy == current_fy and y == today.year and m == today.month:
            end_date = today.strftime("%d-%m-%Y")
            label += " (MTD)"

        periods.append({
            "label": label,
            "start_date": start_date,
            "end_date": end_date,
            "period": f"{start_date} to {end_date}"
        })

        if m == em:
            break

        m += 1
        if m > 12:
            m = 1
            if y is None:
                y = datetime.today().year
            y += 1

    return {
        "type": "mom",
        "fy": f"FY{fy}",
        "periods": periods
    }
    
def parse_multi_year_date(q: str) -> List[Tuple[str, str]] | None:
    """
    Rules:
    - 'and', ','  → discrete years
    - 'to'        → continuous range
    Financial Year: April–March
    """

    print("parse_multi_year_date=============================in function=====")

    if not q or not isinstance(q, str):
        return None

    q_lower = q.lower().strip()

    # -------------------------------------------------
    # 🚫 HARD BLOCKS
    # -------------------------------------------------

    # If month present → handled by month parser
    if re.search(
        r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec',
        q_lower
    ):
        return None

    # Quarter intent
    if re.search(r'\bq[1-4]\b|quarter', q_lower):
        return None

    # -------------------------------------------------
    # Detect continuous vs discrete
    # -------------------------------------------------
    is_continuous = ' to ' in q_lower

    # -------------------------------------------------
    # Extract Years
    # -------------------------------------------------
    year_pattern = r'\b(19\d{2}|20\d{2})\b'
    years = [int(y) for y in re.findall(year_pattern, q_lower)]

    if not years:
        return None

    years = sorted(set(years))

    ranges: List[Tuple[str, str]] = []

    # -------------------------------------------------
    # Build Result
    # -------------------------------------------------

    if is_continuous:
        start_year = years[0]
        end_year = years[-1]

        start_date = f"01-04-{start_year}"
        end_date = f"31-03-{end_year + 1}"

        ranges.append((start_date, end_date))

    else:
        for y in years:
            start_date = f"01-04-{y}"
            end_date = f"31-03-{y + 1}"
            ranges.append((start_date, end_date))

    return ranges


def detect_year_range_logic(question: str):
    import re

    text = question.lower().strip()

    year_range_patterns = [
        r"(20\d{2})\s*(to|and|-|–)\s*(20\d{2})",            # 2022 to 2024, 2022-2024
        r"fy\s*(20\d{2})\s*(to|-|–)\s*fy\s*(20\d{2})",      # FY 2022 to FY 2024
        r"fy\s*(\d{2})\s*(to|-|–)\s*fy\s*(\d{2})",          # FY22 to FY24
        r"fy\s*(\d{2})\s*(to|-|–)\s*(\d{2})",               # FY22-24 (common)
    ]

    print(year_range_patterns,'----------------------------')
    for pattern in year_range_patterns:
        match = re.search(pattern, text)
        if match:
            y1, _, y2 = match.groups()

            # Convert 2-digit year → 4-digit
            if len(y1) == 2:
                y1 = "20" + y1
            if len(y2) == 2:
                y2 = "20" + y2

            y1 = int(y1)
            y2 = int(y2)

            start_year = min(y1, y2)
            end_year = max(y1, y2)

            start_date = f"01-04-{start_year}"
            end_date = f"31-03-{end_year + 1}"

            return {
                "type": "year_range",
                "label": f"FY {start_year} to FY {end_year}",
                "start_date": start_date,
                "end_date": end_date
            }

    return None
                
def parse_single_or_range_date(q: str | None):
    """
    Supports:
      - '15 april'
      - '15 april 2024'
      - '15 to 30 april'
      - '15 apr to 30 apr 2024'
      - '5th jun to 10th jul'
      - '15/04/2024 to 20/04/2024'
      - '15-04-2024 to 20-04-2024'
      - '15 september to 30 september'  ← your failing case
    Returns: (start_date, end_date) or (single_date, single_date) or None
    """
    if not q or not isinstance(q, str):
        return None

    q = q.strip()
    original_q = q
    q_lower = q.lower()

    # ------------------------------------------------------
    # 1️⃣ Same month range: "15 to 30 april" or "15 sep to 30 september 2024"
    # ------------------------------------------------------
    m = re.search(
        r'(\d{1,2}(?:st|nd|rd|th)?)\s+to\s+(\d{1,2}(?:st|nd|rd|th)?)\s+'
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|'
        r'nov(?:ember)?|dec(?:ember)?)'
        r'(?:[,\s]+(20\d{2}|\d{2}))?',
        q_lower
    )
    if m:
        day1_str = m.group(1)
        day2_str = m.group(2)
        month_name = m.group(3)
        year_part = m.group(4)

        base = f"{day1_str} {month_name}"
        if year_part:
            base += f" {year_part}"

        d1 = parse_single_date(base.replace(' to ', ' ') + " placeholder")  # just use base for day1
        raw1 = base
        raw2 = base.replace(day1_str, day2_str, 1)  # replace only first day

        d1 = parse_single_date(raw1)
        d2 = parse_single_date(raw2)

        if d1 and d2 and d1 <= d2:
            return d1, d2

    # ------------------------------------------------------
    # 2️⃣ Full natural language range: "15 apr 2024 to 20 may 2024"
    # ------------------------------------------------------
    full_nat_pat = re.compile(
        r'([0-3]?\d(?:st|nd|rd|th)?\s+[a-z]+\b(?:\s+(?:19|20)?\d{2})?)\s+to\s+'
        r'([0-3]?\d(?:st|nd|rd|th)?\s+[a-z]+\b(?:\s+(?:19|20)?\d{2})?)',
        re.IGNORECASE
    )
    m = full_nat_pat.search(q)
    if m:
        raw1 = m.group(1).strip()
        raw2 = m.group(2).strip()
        d1 = parse_single_date(raw1)
        d2 = parse_single_date(raw2)
        if d1 and d2 and d1 <= d2:
            return d1, d2

    # ------------------------------------------------------
    # 3️⃣ Slash/Hyphen date range: "15/04/2024 to 20/04/2024"
    # ------------------------------------------------------
    slash_pat = re.compile(
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\s+to\s+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        re.IGNORECASE
    )
    m = slash_pat.search(q_lower)
    if m:
        d1 = parse_single_date(m.group(1))
        d2 = parse_single_date(m.group(2))
        if d1 and d2 and d1 <= d2:
            return d1, d2

    # ------------------------------------------------------
    # 4️⃣ Fallback: single date
    # ------------------------------------------------------
    single = parse_single_date(original_q)
    if single:
        return single, single

    return None

def month_year_range_parser(q: str):
    """
    Supports:
        - april 2022 - april 2025
        - april 2022 to april 2025
        - april 2022 & april 2025
        - april 2022 through april 2025
    
    Returns:
        {
            "start_date": "DD-MM-YYYY",
            "end_date": "DD-MM-YYYY"
        }
    """

    if not q or not isinstance(q, str):
        return None

    q = q.lower().strip()

    month_regex = (
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    )

    year_regex = r"(19\d{2}|20\d{2})"
    # range_regex = r"(?:to|till|until|thru|through|and|[-–—&])"

    range_regex = r"(?:\s*(?:to|till|until|thru|through|and|-|–|—|&)\s*)"

    # pattern = rf"\b{month_regex}\s+{year_regex}\b\s*{range_regex}\s*\b{month_regex}\s+{year_regex}\b"
    pattern = rf"\b{month_regex}\s+{year_regex}{range_regex}{month_regex}\s+{year_regex}\b"
    match = re.search(pattern, q)

    if not match:
        return None

    start_month_str = match.group(1)
    start_year_str  = match.group(2)
    end_month_str   = match.group(3)
    end_year_str    = match.group(4)

    month_map = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct':10, 'october':10,
        'nov':11, 'november':11,
        'dec':12, 'december':12,
    }

    m1 = month_map.get(start_month_str[:3])
    m2 = month_map.get(end_month_str[:3])

    if not m1 or not m2:
        return None

    y1 = int(start_year_str)
    y2 = int(end_year_str)

    start_date = f"01-{m1:02d}-{y1}"
    end_day = monthrange(y2, m2)[1]
    end_date = f"{end_day:02d}-{m2:02d}-{y2}"

    return {
        "start_date": start_date,
        "end_date": end_date
    }

        
def month_range_with_year(q: str):
    """
    Supports:
      - april to june
      - april to june 2024
      - april to june last year
      - jan to mar previous year
      - oct to feb last year
    Financial Year: April–March
    """
    print("month_range_with_year---------------------")

    if not q or not isinstance(q, str):
        return None

    q = q.lower().strip()

    RANGE_WORDS = r'(?:to|till|until|thru|through|-|–|—|and)'
    MONTH_PATTERN = (
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|'
        r'jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|'
        r'oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    )
    YEAR_PATTERN = r'(19\d{2}|20\d{2}|\d{2})'

    # -----------------------------------
    # Detect relative year phrases
    # -----------------------------------
    is_last_year = bool(re.search(r'\b(last|previous)\s+year\b', q))
    is_this_year = bool(re.search(r'\b(this|current)\s+year\b', q))

    # -----------------------------------
    # Month range regex
    # -----------------------------------
    m = re.search(
        r'(' + MONTH_PATTERN + r')\s*' +
        RANGE_WORDS +
        r'\s*(' + MONTH_PATTERN + r')' +
        r'(?:[,\s]+(' + YEAR_PATTERN + r'))?',
        q
    )

    if not m:
        return None
    start_month_str, end_month_str, year_str = m.group(1), m.group(3), m.group(5)

    # -----------------------------------
    # Normalize months
    # -----------------------------------
    month_map = {
        'jan':1, 'january':1,
        'feb':2, 'february':2,
        'mar':3, 'march':3,
        'apr':4, 'april':4,
        'may':5,
        'jun':6, 'june':6,
        'jul':7, 'july':7,
        'aug':8, 'august':8,
        'sep':9, 'sept':9, 'september':9,
        'oct':10, 'october':10,
        'nov':11, 'november':11,
        'dec':12, 'december':12,
    }

    def norm_month(m):
        return month_map.get(m[:3])

    m1 = norm_month(start_month_str)
    m2 = norm_month(end_month_str)
    if not m1 or not m2:
        return None

    # -----------------------------------
    # Resolve Financial Year
    # -----------------------------------
    today = datetime.today()
    current_fy = today.year if today.month >= 4 else today.year - 1

    if year_str:
        y = int(year_str)
        if y < 100:
            y += 2000
        fy = y
    elif is_last_year:
        fy = current_fy - 1
    else:
        fy = current_fy

    # -----------------------------------
    # Assign years per month (FY-aware)
    # -----------------------------------
    y1 = fy if m1 >= 4 else fy + 1
    y2 = fy if m2 >= 4 else fy + 1

    # Cross-year correction (e.g., Oct → Feb)
    if m2 < m1:
        y2 += 1

    # -----------------------------------
    # Build dates
    # -----------------------------------
    start_date = f"01-{m1:02d}-{y1}"
    end_day = monthrange(y2, m2)[1]
    end_date = f"{end_day:02d}-{m2:02d}-{y2}"

    return {
        "start_date": start_date,
        "end_date": end_date
    }

def month_rang_without_year(q:str):
    month_regex = (
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)"
    )

    m = re.search(
        rf"{month_regex}\s*(?:to|and|or|\-|&|\s)\s*{month_regex}(?:\s+(20\d{{2}}|\d{{2}}))?",
        q
    )   

    if m:
        m1_name = m.group(1)
        m2_name = m.group(2)
        year_part = m.group(3)   # optional year after second month

        m1 = extract_month_by_name(m1_name)
        m2 = extract_month_by_name(m2_name)
        print(m1,m2,year_part)

        # If a year is provided once, apply same year to both months
        if year_part:
            year = int(year_part)
            if len(year_part) == 2:
                year = 2000 + year
            y1 = y2 = year
        else:
            # Fiscal year handling
            fy = get_current_fy()
            y1 = fy if m1 >= 4 else fy + 1
            y2 = fy if m2 >= 4 else fy + 1

        # Build start date (1st of first month)
        start_date = f"01-{m1:02d}-{y1}"
        print(start_date)

        # Build end date (last day of second month)
        last_day = monthrange(y2, m2)[1]
        end_date = f"{last_day:02d}-{m2:02d}-{y2}"
        print(end_date)


        return {
            "start_date": start_date, 
            "end_date": end_date
        }

    return None

def get_last_day_of_month(year: int, month: int) -> int:
    """Safely return the last day of the given month/year."""
    # List of days in each month (index 0 unused)
    month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    if month == 2:
        # Check for leap year
        if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            return 29
        else:
            return 28
    else:
        return month_days[month]
    
def get_current_fy_year() -> int:
    today = datetime.today()
    return today.year if today.month >= 4 else today.year - 1


def parse_multi_month_date(q: str) -> List[Tuple[str, str]] | None:
    """
    Rules:
    - 'and', ','  → discrete months (ONLY those months)
    - 'to'        → continuous range (fill months in between)
    Financial Year: April–March
    """

    if not q or not isinstance(q, str):
        return None

    q_lower = q.lower().strip()

    # -------------------------------------------------
    # 🚫 HARD BLOCKS (wrong intent)
    # -------------------------------------------------

    # Exact date present → handled by date parser
    if (
        re.search(r'\b\d{1,2}(st|nd|rd|th)?\b', q_lower)
        and re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', q_lower)
    ):
        return None

    # Quarter intent → handled by quarter parser
    if re.search(r'\bq[1-4]\b|quarter', q_lower):
        return None

    # Exact date range like "15 sep to 30 oct"
    if re.search(
        r'\b\d{1,2}\b.*\b(to|till|until|through|-|–|—)\b.*\b\d{1,2}\b',
        q_lower
    ):
        return None

    # Continuous month range with relative year → handled elsewhere
    if (
        ' to ' in q_lower
        and re.search(r'\b(last|previous|this|current|next)\s+year\b', q_lower)
    ):
        return None

    # Non-"to" range words should not enter here
    if re.search(r'\b(till|until|through|-|–|—)\b', q_lower) and ' to ' not in q_lower:
        return None

    # -------------------------------------------------
    # Detect continuous vs discrete
    # -------------------------------------------------
    is_continuous = ' to ' in q_lower

    # -------------------------------------------------
    # Relative year handling (ONLY for discrete months)
    # -------------------------------------------------
    fy_shift = 0
    if re.search(r'\b(last|previous)\s+year\b', q_lower):
        fy_shift = -1
    elif re.search(r'\b(next)\s+year\b', q_lower):
        fy_shift = 1

    # -------------------------------------------------
    # 1️⃣ YEAR-ONLY DETECTION
    # -------------------------------------------------
    year_pattern = r'\b(19\d{2}|20\d{2})\b'
    years = [int(y) for y in re.findall(year_pattern, q_lower)]

    month_check = re.search(
        r'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec',
        q_lower
    )

    if years and not month_check:
        years = sorted(set(years))
        ranges = []

        if is_continuous:
            start_year = years[0]
            end_year = years[-1]
            ranges.append(
                (f"01-04-{start_year}", f"31-03-{end_year + 1}")
            )
        else:
            for y in years:
                ranges.append(
                    (f"01-04-{y}", f"31-03-{y + 1}")
                )

        return ranges

    # -------------------------------------------------
    # 2️⃣ MONTH LOGIC
    # -------------------------------------------------

    q_normalized = re.sub(r'\s+and\s+|\s*,\s*', ' ', q_lower)

    month_pattern = (
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
        r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|'
        r'nov(?:ember)?|dec(?:ember)?)'
    )

    month_matches = re.findall(month_pattern, q_normalized)
    if not month_matches:
        return None

    # Explicit year mentions
    year_pattern = r'\b(20\d{2}|19\d{2}|\d{2})\b'
    year_mentions = re.findall(year_pattern, q_lower)

    explicit_years = []
    for y in year_mentions:
        yi = int(y)
        if len(y) == 2:
            yi += 2000
        explicit_years.append(yi)

    default_year = get_current_fy_year() + fy_shift

    month_years = []

    for i, month_name in enumerate(month_matches):
        month_num = extract_month_by_name(month_name)
        if not month_num:
            continue

        if explicit_years:
            year = explicit_years[min(i, len(explicit_years) - 1)]
        else:
            year = default_year + 1 if month_num < 4 else default_year

        month_years.append((year, month_num))

    if not month_years:
        return None

    month_years = sorted(set(month_years))

    # -------------------------------------------------
    # 3️⃣ BUILD RESULT
    # -------------------------------------------------
    ranges: List[Tuple[str, str]] = []

    if is_continuous:
        start_year, start_month = month_years[0]
        end_year, end_month = month_years[-1]

        start_date = f"01-{start_month:02d}-{start_year}"
        last_day = get_last_day_of_month(end_year, end_month)
        end_date = f"{last_day:02d}-{end_month:02d}-{end_year}"

        ranges.append((start_date, end_date))
    else:
        for year, month in month_years:
            last_day = get_last_day_of_month(year, month)
            start_date = f"01-{month:02d}-{year}"
            end_date = f"{last_day:02d}-{month:02d}-{year}"
            ranges.append((start_date, end_date))

    return ranges


def multi_quater_parser(q: str):
    quarter_matches = re.findall(r'\b(?:q|quarter|qtr\s*)([1-4])\b', q)
    fy_shift = 0
    if re.search(r'\b(last|previous)\s+year\b', q):
        fy_shift = -1
    elif re.search(r'\b(next)\s+year\b', q):
        fy_shift = 1

    is_continuous = ' to ' in q
    if quarter_matches:
        quarter_nums = [int(q) for q in quarter_matches]

        # Extract explicit year if present, otherwise use current FY start year
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', q)
        if year_match:
            base_fy_year = int(year_match.group(1))
        else:
            today = date.today()
            # FY starts in April
            base_fy_year = today.year if today.month >= 4 else today.year - 1
            base_fy_year += fy_shift

        def get_quarter_dates(q_num: int, fy_year: int):
            quarter_map = {
                1: (f"01-04-{fy_year}",     f"30-06-{fy_year}"),
                2: (f"01-07-{fy_year}",     f"30-09-{fy_year}"),
                3: (f"01-10-{fy_year}",     f"31-12-{fy_year}"),
                4: (f"01-01-{fy_year + 1}", f"31-03-{fy_year + 1}"),
            }
            return quarter_map[q_num]

        ranges = []

        if is_continuous and len(quarter_nums) >= 2:
            # q1 to q3 → expand all quarters in between
            start_q = min(quarter_nums)
            end_q = max(quarter_nums)

            for q_num in range(start_q, end_q + 1):
                ranges.append(get_quarter_dates(q_num, base_fy_year))
        else:
            # q1 and q3 → only selected quarters
            for q_num in sorted(set(quarter_nums)):
                ranges.append(get_quarter_dates(q_num, base_fy_year))

        return ranges

def parse_date_range(question: str):
    q = normalize(question)

    start_date, end_date = None, None

    # -------------------------------------------------------
    # 1️⃣ Detect Quarter (q1, q2, q3, q4)
    # -------------------------------------------------------
    if "q1" in q or "quarter 1" in q:
        year = extract_year_from_text(q) or get_current_fy()
        return f"01-04-{year}", f"30-06-{year}"

    if "q2" in q or "quarter 2" in q:
        year = extract_year_from_text(q) or get_current_fy()
        return f"01-07-{year}", f"30-09-{year}"

    if "q3" in q or "quarter 3" in q:
        year = extract_year_from_text(q) or get_current_fy()
        return f"01-10-{year}", f"31-12-{year}"

    if "q4" in q or "quarter 4" in q:
        year = extract_year_from_text(q) or get_current_fy()
        year = year + 1
        return f"01-01-{year}", f"31-03-{year}"
    
    # -------------------------------------------------------
    # 3️⃣ between date like "5 june  to 10 june 2024"
    # -------------------------------------------------------
    date_pair = parse_single_or_range_date(q)
    if date_pair:
        d1, d2 = date_pair
        return d1.strftime("%d-%m-%Y"), d2.strftime("%d-%m-%Y")

    # -------------------------------------------------------
    # 3️⃣ Single full date like "5 june 2024"
    # -------------------------------------------------------

    # d = parse_single_date(q)
    # if d:
    #     return d.strftime("%d-%m-%Y"), d.strftime("%d-%m-%Y")

    # # First try multi-month full months
    # multi = parse_multi_month_date(q)
    # if multi:
    #     return {"ranges": multi}  # list of full month ranges
    
    range_data  = month_year_range_parser(q)
    if range_data:
        return range_data
    
    # -------------------------------------------------------
    # 2️⃣ Detect explicit date range: "april 2024 to june 2024"
    # -------------------------------------------------------
    month_range = month_range_with_year(q)
    if month_range:
        return month_range['start_date'],month_range['end_date']

    # -------------------------------------------------------
    # 6️⃣ Month range WITHOUT year (april to september)
    # -------------------------------------------------------
    month_range_wy = month_rang_without_year(q)
    if month_range_wy:
        return month_range_wy['start_date'],month_range_wy['end_date']
    
    # -------------------------------------------------------
    # 4️⃣ Single Month WITH year
    # -------------------------------------------------------
    month = extract_month_by_name(q)
    year = extract_year_from_text(q)

    if month and year:
        start_date = f"01-{month:02d}-{year}"
        last_day = monthrange(year, month)[1]
        end_date = f"{last_day:02d}-{month:02d}-{year}"
        return start_date, end_date

    # -------------------------------------------------------
    # 5️⃣ Single Month WITHOUT year (FY logic)
    # -------------------------------------------------------
    if month and not year:
        fy = get_current_fy()
        year = fy if month >= 4 else fy + 1
        start_date = f"01-{month:02d}-{year}"
        last_day = monthrange(year, month)[1]
        end_date = f"{last_day:02d}-{month:02d}-{year}"
        return start_date, end_date
    
    
    # -------------------------------------------------------
    # Additional check for "last n days"
    # -------------------------------------------------------
    last_n_days = detect_last_n_days(question)
    if last_n_days:
        return last_n_days["start_date"], last_n_days["end_date"]
    
    # -------------------------------------------------------
    # Additional check for "this period"
    # -------------------------------------------------------
    this_period = detect_this_period(question)
    if this_period:
        return this_period["start_date"], this_period["end_date"]


    # -------------------------------------------------------
    # 7️⃣ FY detection
    # -------------------------------------------------------
    if "fy" in q or "financial year" in q or "f y" in q:
        year = extract_year_from_text(q)
        if year:
            return f"01-04-{year}", f"31-03-{year+1}"
        
    print("-----------------------------------------------------------")
        
    # year range logic       
    year_range = detect_year_range_logic(question)
    if year_range:
        print("===========================================")
        return year_range["start_date"], year_range["end_date"]

    # Standalone year → interpret as FY
    year = extract_year_from_text(q)
    if year:
        return f"01-04-{year}", f"31-03-{year+1}"

    # -------------------------------------------------------
    # 8️⃣ Default → current FY
    # -------------------------------------------------------
    fy = get_current_fy()
    return f"01-04-{fy}", f"31-03-{fy+1}"

# def normalize_to_date(d: Union[str, date]) -> date:
#     """
#     Normalize date input to python date object.
#     Accepts:
#       - 'DD-MM-YYYY'
#       - 'YYYY-MM-DD'
#       - datetime.date
#     """

#     print(f"Normalizing date: {d}")
#     if isinstance(d, date):
#         return d
#     print(f"Normalizing date: {d}")
#     d = d.strip()
#     print(f"Failed to parse date: {d}")
#     for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
#         try:
#             return datetime.strptime(d, fmt).date()
#         except ValueError:
#             continue

    
#     raise ValueError(f"Unsupported date format: {d}")



def normalize_to_date(d: Union[str, date]) -> date:
    print(f"Normalizing date: {d}")

    if isinstance(d, date):
        return d

    if d is None:
        raise ValueError("Date cannot be None")

    d = str(d).strip()

    if d.lower() in ("start_date", "end_date"):
        raise ValueError(
            f"Received placeholder '{d}' instead of an actual date value."
        )

    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(d, fmt).date()
        except ValueError:
            continue

    raise ValueError(f"Unsupported date format: {d}")

def build_sql_date_filter(date_result, column_name="period_month"):
    """
    Builds SQL WHERE clause safely from date_result
    """

    if isinstance(date_result, dict) and "ranges" in date_result:
        where_clauses = []

        
        for start_date, end_date in date_result["ranges"]:
            start_dt = normalize_to_date(start_date)
            end_dt = normalize_to_date(end_date)

            where_clauses.append(
                f"DATE({column_name}) BETWEEN DATE '{start_dt.isoformat()}' AND DATE '{end_dt.isoformat()}'"
            )

        date_filter = " WHERE (" + " OR ".join(where_clauses) + ")"
        filters_label = ", ".join(
            [f"{normalize_to_date(s)} to {normalize_to_date(e)}" for s, e in date_result["ranges"]]
        )

        # global min/max for downstream functions
        global_start = min(normalize_to_date(s) for s, _ in date_result["ranges"])
        global_end = max(normalize_to_date(e) for _, e in date_result["ranges"])

    else:
        start_date, end_date = date_result
        start_dt = normalize_to_date(start_date)
        end_dt = normalize_to_date(end_date)

        date_filter = (
            f" WHERE DATE({column_name}) BETWEEN "
            f"DATE '{start_dt.isoformat()}' AND DATE '{end_dt.isoformat()}'"
        )

        filters_label = f"{start_dt} to {end_dt}"
        global_start = start_dt
        global_end = end_dt

    return date_filter, filters_label, global_start, global_end

# --------------------------------------------
# Source-Wise Funnel API Endpoint
# --------------------------------------------
@app.post("/funnel/leadfunnel/question")
async def source_funnel_from_question(payload: dict = Body(...)):
    question = payload.get("question", "").lower()
    logger.info(f"Incoming /funnel/leadfunnel/question request: {question}")
    # ------------------------------------
    # 1. CHECK FOR QOQ FIRST
    # ------------------------------------
    qoq_result = detect_qoq(question)
    if qoq_result:
        quarters = qoq_result["quarters"]
        analysis_type = qoq_result["type"]
        fy = qoq_result["fy"]
        all_quarter_results = []
        all_quarter_funnels = []

        logger.info(f"Detected {analysis_type.upper()} for FY{fy} → Running {len(quarters)} quarters")

        for qtr in quarters:
            start_date = qtr["start_date"]
            end_date = qtr["end_date"]
            label = qtr["quarter"]

            logger.info(f"Processing {label}: {start_date} to {end_date}")
            print(f"Processing {label}: {start_date} to {end_date}")

            # Build date filter for this quarter
            date_filter = f""" 
                        WHERE DATE(period_month) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                            """

            # SQL Queries
            lead_sql = f"""
                SELECT *
                FROM {CATALOG}.{FUNNEL_SCHEMA}.{FUNNEL_TABLE}
                {date_filter}
            """

            # Execute Presto queries (including event query)
            try:
                funnel_data = query_presto(CATALOG, FUNNEL_SCHEMA, lead_sql)
            except Exception as e:
                logger.error(f"Presto query failed: {e}", exc_info=True)
                return {"status":"error","message":str(e)}

            if funnel_data.empty:
                logger.warning("No leads found for selected period.")

            # Compute source-wise funnel with events
            source_funnel = compute_source_wise_funnel(funnel_data)

            result = {
                    "quarter": label,
                    "period": f"{start_date} to {end_date}",
                    "funnel": sort_funnel_by_numeric_desc(source_funnel, return_as_list=True),
                }
            all_quarter_results.append(result)
            all_quarter_funnels.append(source_funnel)

            # Return structured QoQ or Quarter-wise response
        all_quarter_results.sort(
            key=lambda x: x["funnel"].get("Total Leads", 0) if isinstance(x.get("funnel"), dict) else 0,
            reverse=True
        )
        return {
            "status": "success",
            "analysis_type": analysis_type,
            "fy": f"FY{fy}",
            "data": all_quarter_results,
            "totals": calculate_master_totals(all_quarter_funnels)
        }

    elif (yoy_result := detect_yoy(question)):
        periods = yoy_result["periods"]
        all_year_results = []
        all_year_funnels = []

        logger.info(f"Running YoY analysis → {len(periods)} years")

        for period in periods:
            start_date = period["start_date"]
            end_date = period["end_date"]
            label = period["year"]

            date_filter = f""" 
                        WHERE DATE(period_month) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                            """

            # SQL Queries
            lead_sql = f"""
                SELECT *
                FROM {CATALOG}.{FUNNEL_SCHEMA}.{FUNNEL_TABLE}
                {date_filter}
            """

            # Execute Presto queries (including event query)
            try:
                leads = query_presto(CATALOG, FUNNEL_SCHEMA, lead_sql)
            except Exception as e:
                logger.error(f"Presto query failed: {e}", exc_info=True)
                return {"status":"error","message":str(e)}

            if leads.empty:
                logger.warning("No leads found for selected period.")
                # return {"status":"no_data","message":"No leads found for selected period"}

            # Compute source-wise funnel with events
            source_funnel = compute_source_wise_funnel(leads)

            result = {
                    "quarter": label,
                    "period": f"{start_date} to {end_date}",
                    "funnel": sort_funnel_by_numeric_desc(source_funnel, return_as_list=True),
                }
            all_year_results.append(result)
            all_year_funnels.append(source_funnel)

            # Return structured QoQ or Quarter-wise response
        all_year_results.sort(
            key=lambda x: x["funnel"].get("Total Leads", 0) if isinstance(x.get("funnel"), dict) else 0,
            reverse=True
        )
        return {
            "status": "success",
            "analysis_type": "year_on_year",
            "fy": "Last 3 completed financial years",
            "data": all_year_results,
            "totals": calculate_master_totals(all_year_funnels)
        }
    elif (mom_result := detect_mom(question)):
        periods = mom_result["periods"]
        all_month_results = []
        all_month_funnels = []

        logger.info(f"Running MoM analysis → {len(periods)} Months")

        for period in periods:
            label = period["label"]
            start_date = datetime.strptime(period["start_date"], "%d-%m-%Y").strftime("%Y-%m-%d")
            end_date = datetime.strptime(period["end_date"], "%d-%m-%Y").strftime("%Y-%m-%d")

            print(f"Processing {label}: {start_date} to {end_date}")

            date_filter = f""" 
                        WHERE DATE(period_month) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                            """

            # SQL Queries
            lead_sql = f"""
                SELECT *
                FROM {CATALOG}.{FUNNEL_SCHEMA}.{FUNNEL_TABLE}
                {date_filter}
            """

            # Execute Presto queries (including event query)
            try:
                leads = query_presto(CATALOG, FUNNEL_SCHEMA, lead_sql)
               
            except Exception as e:
                logger.error(f"Presto query failed: {e}", exc_info=True)
                return {"status":"error","message":str(e)}

            if leads.empty:
                logger.warning("No leads found for selected period.")
                # return {"status":"no_data","message":"No leads found for selected period"}

            # Compute source-wise funnel with events
            source_funnel = compute_source_wise_funnel(leads)
            all_month_funnels.append(source_funnel)
            all_month_results.append({
                "month": label,
                "period": period["period"],
                "funnel": source_funnel,
            })

        all_month_results.sort(
            key=lambda x: x["funnel"].get("Total Leads", 0) if isinstance(x.get("funnel"), dict) else 0,
            reverse=True
        )
        return {
            "status": "success",
            "analysis_type": "month_on_month",
            "comparison": "Last 6 months + Current MTD",
            "data": all_month_results,
            "totals": calculate_master_totals(all_month_funnels)
        }
        
    elif (mul_year := parse_multi_year_date(question)):
            filters_label = ""
            yearly_results = []
            yearly_funnels = []
            for start_str, end_str in mul_year:

                start_date = datetime.strptime(start_str, "%d-%m-%Y").strftime("%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%d-%m-%Y").strftime("%Y-%m-%d")
                date_filter = f""" 
                        WHERE DATE(period_month) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                            """
                # SQL Queries
                lead_sql = f"""
                    SELECT *
                    FROM {CATALOG}.{FUNNEL_SCHEMA}.{FUNNEL_TABLE}
                    {date_filter}
                """

                # Execute Presto queries (including event query)
                try:
                    leads = query_presto(CATALOG, FUNNEL_SCHEMA, lead_sql)
                
                except Exception as e:
                    logger.error(f"Presto query failed: {e}", exc_info=True)
                    return {"status":"error","message":str(e)}

                if leads.empty:
                    logger.warning("No leads found for selected period.")
                    # return {"status":"no_data","message":"No leads found for selected period"}

                # Compute source-wise funnel with events
                source_funnel = compute_source_wise_funnel(leads)
                yearly_funnels.append(source_funnel)
                yearly_results.append({
                    "period": f"{start_date} to {end_date}",
                    "funnel": sort_funnel_by_numeric_desc(source_funnel, return_as_list=True),
                })
            yearly_results.sort(
                key=lambda x: x["funnel"].get("Total Leads", 0) if isinstance(x.get("funnel"), dict) else 0,
                reverse=True
            )
            return {
                "status": "success",
                "filter": filters_label,
                "filter_type": "discrete_years",
                "periods": yearly_results,
                "totals": calculate_master_totals(yearly_funnels)
            }
                
    elif (mul_quarter := multi_quater_parser(question)):
        quarter_results = []
        quarter_funnels = []
        filters_label = []

        for start_date, end_date in mul_quarter:
            # Convert DD-MM-YYYY -> YYYY-MM-DD
            start_date_sql = datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
            end_date_sql = datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")

            period_label = f"{start_date_sql} to {end_date_sql}"
            filters_label.append(period_label)

            date_filter = f"""
                WHERE DATE(period_month)
                    BETWEEN DATE '{start_date_sql}'
                        AND DATE '{end_date_sql}'
            """

            lead_sql = f"""
                SELECT *
                FROM {CATALOG}.{FUNNEL_SCHEMA}.{FUNNEL_TABLE}
                {date_filter}
            """

            try:
                leads = query_presto(CATALOG, FUNNEL_SCHEMA, lead_sql)
            except Exception as e:
                logger.error(f"Presto query failed: {e}", exc_info=True)
                return {
                    "status": "error",
                    "message": str(e)
                }

            if leads.empty:
                logger.warning(
                    f"No leads found for period {start_date_sql} to {end_date_sql}"
                )

            source_funnel = compute_source_wise_funnel(leads)

            quarter_funnels.append(source_funnel)
            quarter_results.append({
                "period": period_label,
                "funnel": sort_funnel_by_numeric_desc(
                    source_funnel,
                    return_as_list=True
                ),
            })

        # Sort by Total Leads (works because funnel is returned as list)
        def get_total_leads(item):
            funnel = item.get("funnel", [])
            if isinstance(funnel, list):
                for row in funnel:
                    if row.get("Stage") == "Total Leads":
                        return row.get("Count", 0)
            return 0

        quarter_results.sort(
            key=get_total_leads,
            reverse=True
        )

        return {
            "status": "success",
            "filter": ", ".join(filters_label),
            "filter_type": "multi_quarters",
            "periods": quarter_results,
            "totals": calculate_master_totals(quarter_funnels)
        }

    elif (mul_month := parse_multi_month_date(question)):
        period_results = []
        period_funnels = []
        filters_label = []

        for start_date, end_date in mul_month:

            start_date = datetime.strptime(start_date, "%d-%m-%Y").strftime("%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%d-%m-%Y").strftime("%Y-%m-%d")
            period_label = f"{start_date} to {end_date}"
            filters_label.append(period_label)
            date_filter = f"""
                        WHERE DATE(period_month) BETWEEN DATE '{start_date}' AND DATE '{end_date}'
                            """

            # SQL Queries
            lead_sql = f"""
                SELECT *
                FROM {CATALOG}.{FUNNEL_SCHEMA}.{FUNNEL_TABLE}
                {date_filter}
            """

            # Execute Presto queries (including event query)
            try:
                leads = query_presto(CATALOG, FUNNEL_SCHEMA, lead_sql)

            except Exception as e:
                logger.error(f"Presto query failed: {e}", exc_info=True)
                return {"status":"error","message":str(e)}

            if leads.empty:
                logger.warning("No leads found for selected period.")
                # return {"status":"no_data","message":"No leads found for selected period"}

            # Compute source-wise funnel with events
            source_funnel = compute_source_wise_funnel(leads)
            period_funnels.append(source_funnel)
            period_results.append({
                "period": period_label,
                "funnel": sort_funnel_by_numeric_desc(source_funnel, return_as_list=True),
            })

        period_results.sort(
            key=lambda x: x["funnel"].get("Total Leads", 0) if isinstance(x.get("funnel"), dict) else 0,
            reverse=True
        )
        return {
            "status": "success",
            "filter": ", ".join(filters_label),
            "filter_type": "discrete_months",
            "periods": period_results,
            "totals": calculate_master_totals(period_funnels)
        }
    else:

        # ------------------------------------
        # 2. NORMAL DATE PARSING (your logic)
        # ------------------------------------
        print('Normal date parsing flow --------------------')
        date_result = parse_date_range(question)

        date_filter, filters_label, start_date, end_date = build_sql_date_filter(
            date_result,
            column_name="period_month"
        )

        logger.info(f"Final SQL Date Filter: {date_filter.strip()}")

        # SQL queries - including event query with OwnerId
        lead_sql = f"""
            SELECT *
            FROM {CATALOG}.{FUNNEL_SCHEMA}.{FUNNEL_TABLE}
            {date_filter}
        """

        print(lead_sql,'lead sql --------------------')

        # Execute Presto queries (including event query)
        try:
            leads = query_presto(CATALOG, FUNNEL_SCHEMA, lead_sql)
        except Exception as e:
            logger.error(f"Presto query failed: {e}", exc_info=True)
            return {"status":"error","message":str(e)}

        if leads.empty:
            logger.warning("No leads found for selected period.")
            return {"status":"no_data","message":"No leads found for selected period"}

        # Compute source-wise funnel with events
        source_funnel = compute_source_wise_funnel(leads)

        return {
            "status": "success",
            "filters": filters_label,
            "lead funnel": source_funnel,
            "totals": calculate_master_totals(source_funnel)
        }

# --------------------------------------------
# Health Check Endpoint
# --------------------------------------------
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Source-Wise Funnel Analytics API is running"}