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
from typing import List, Dict, Any, Optional, Tuple
from typing import Any, Dict, List, Union

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
    - Dict[str, Dict] -> Funnel-style output
    """
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if isinstance(data, dict):
        # Funnel-style: { "Wave City": { ...metrics... }, ... }
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())

    return []


def calculate_master_totals(data: Any) -> Dict[str, Union[int, float]]:
    """
    MASTER aggregation logic:
    - Column-wise totals
    - Sums ALL additive numeric fields
    - Ignores % and ratio fields
    - Works for ALL result shapes you showed
    """
    rows = _normalize_to_rows(data)
    totals: Dict[str, float] = {}

    for row in rows:
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
        logging.FileHandler("funnel_source_tool.log", mode="a", encoding="utf-8"),
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
LEAD_SCHEMA = os.getenv("LEAD_SCHEMA","lead_fy_year")
OPP_SCHEMA = os.getenv("OPP_SCHEMA","opportunity_sf_report")
EVENT_SCHEMA = os.getenv("EVENT_SCHEMA","event_sf_report")

LEAD_TABLE = os.getenv("LEAD_TABLE","lead_fy_report")
OPP_TABLE = os.getenv("OPP_TABLE","opportunity_report")
EVENT_TABLE = os.getenv("EVENT_TABLE","event_report")

hostname = os.getenv("PRESTO_HOST")
portnumber = int(os.getenv("PRESTO_PORT"))
username = os.getenv("PRESTO_USERNAME")
password = os.getenv("PRESTO_PASSWORD")

# Watsonx Foundation Model credentials (optional)
creds = Credentials(
    url=os.getenv("WATSONX_URL"),
    api_key=os.getenv("WATSONX_API_KEY")
)

model = ModelInference(
    model_id=os.getenv("MODEL_ID"),
    credentials=creds,
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params={"temperature": 0, "max_new_tokens": 300}
)

PROJECT_ALIASES = {
    "wave city": "wave city",
    "wavecity": "wave city",
    "wave_city": "wave city",

    "wmcc sec 32": "wmcc sec 32",
    "wmcc": "wmcc sec 32",
    "wmcc sector 32": "wmcc sec 32",
    "wmcc sec32": "wmcc sec 32",
    "wmcc32": "wmcc sec 32",

    "wave estate": "wave estate",
    "estate": "wave estate",  # careful if ambiguous

    # Add more as needed...
}
PRODUCT_ALIASES = {
    # Full product list from your input (exact DB values)
    "amore": "AMORE",
    "armonia": "ARMONIA",
    "villa": "VILLA",
    "comm booth": "COMM BOOTH",
    "dream homes": "DREAM HOMES",
    "eden": "EDEN",
    "eligo": "ELIGO",
    "ews": "EWS",
    "ews 410": "EWS_001_(410)",
    "ews_001_(410)": "EWS_001_(410)",
    "executive floors": "EXECUTIVE FLOORS",
    "executive": "EXECUTIVE FLOORS",  # short form
    "fsi": "FSI",
    "golf range": "Golf Range",
    "harmony greens": "HARMONY GREENS",
    "hssc": "HSSC",
    "institutional": "INSTITUTIONAL",
    "lig": "LIG",
    "lig 310": "LIG_001_(310)",
    "lig_001_(310)": "LIG_001_(310)",
    "livork": "LIVORK",
    "mayfair park": "Mayfair Park",
    "new plots": "NEW PLOTS",
    "old plots": "OLD PLOTS",
    "plot res if": "PLOT-RES-IF",
    "plots comm": "PLOTS-COMM",
    "plots res": "PLOTS-RES",
    "prime floors": "PRIME FLOORS",
    "swamanorath": "SWAMANORATH",
    "vasilia": "VASILIA",
    "veridia": "VERIDIA",
    "veridia 3": "VERIDIA-3",
    "veridia 4": "VERIDIA-4",
    "veridia 5": "VERIDIA-5",
    "veridia 6": "VERIDIA-6",
    "veridia 7": "VERIDIA-7",
    "wave floor": "WAVE FLOOR",
    "wave floor 85": "WAVE FLOOR 85",
    "wave floor 99": "WAVE FLOOR 99",
    "wave galleria": "WAVE GALLERIA",
    "wave garden": "WAVE GARDEN",
    "wave garden gh2 ph2": "WAVE GARDEN GH2-Ph-2",
    "waved garden": "WAVED GARDEN",

    # Additional useful short forms (optional, customize as needed)
    "veridia-3": "VERIDIA-3",
    "veridia-4": "VERIDIA-4",
    "veridia-5": "VERIDIA-5",
    "veridia-6": "VERIDIA-6",
    "veridia-7": "VERIDIA-7",
    "wave floors": "WAVE FLOOR",
    "executive floor": "EXECUTIVE FLOORS",
    "prime floor": "PRIME FLOORS",
}

# Known lead sources (add all your actual lead_source_c values)
SOURCE_ALIASES = {
    # Canonical → itself
    "bulk sale": "Bulk Sale",
    "channel partner": "Channel Partner",
    "digital": "Digital",
    "direct": "Direct",
    "direct walkin": "Direct Walkin",
    "electronic media": "Electronic Media",
    "events / exhibitions": "Events / Exhibitions",
    "existing customer": "Existing customer",
    "lead reassigned": "Lead Reassigned",
    "outbound campaign": "Outbound Campaign",
    "outdoor": "Outdoor",
    "print media": "Print Media",
    "reference sale": "Reference Sale",
    "referral": "Referral",
    "referral sale": "Referral Sale",
    "sms campaign": "SMS Campaign",
    "transfered unit": "Transfered Unit",
    "shifting": "Shifting",
    "word of mouth": "Word of mouth",

    # Common short forms / variations → map to canonical
    "bulk": "Bulk Sale",
    "channel": "Channel Partner",
    "cp": "Channel Partner",
    "direct walk-in": "Direct Walkin",
    "walkin": "Direct Walkin",
    "walk-in": "Direct Walkin",
    "event": "Events / Exhibitions",
    "exhibition": "Events / Exhibitions",
    "existing": "Existing customer",
    "reassigned": "Lead Reassigned",
    "outbound": "Outbound Campaign",
    "hoarding": "Outdoor",
    "print": "Print Media",
    "reference": "Reference Sale",
    "refer": "Referral",
    "referral sale": "Referral Sale",
    "sms": "SMS Campaign",
    "transfer": "Transfered Unit",
    "transferred": "Transfered Unit",
    "unit shifting": "Shifting",

}

GENERIC_PROJECT_TRIGGERS = {"project", "projects"}
GENERIC_PRODUCT_TRIGGERS = {"product", "products", "category", "categories", "channel", "channels"}
SOURCE_TRIGGER_WORDS = {"source", "sources", "channel", "channels", "from"}
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
# Source-Wise Funnel Metrics (WITH Events)
# --------------------------------------------
def _compute_funnel_metrics_exact(leads: pd.DataFrame, opps: pd.DataFrame, events: pd.DataFrame) -> Dict[str, Any]:
    total_leads = len(leads)

    if total_leads == 0 and len(opps) == 0:
        return _get_empty_metrics()

    # Customer feedback
    cf_series = leads.get("customer_feedback_c", pd.Series([""] * total_leads))
    cf = cf_series.fillna("").astype(str).str.strip().str.lower()

    junk_leads = (cf == "junk").sum()
    sol_leads = (cf == "interested").sum()
    valid_leads = ((cf != "junk") & (cf != "") & cf.notnull()).sum()  # Your exact logic

    # Sales done
    sales_col = opps.get("sales_order_number_c", pd.Series([""] * len(opps)))
    sales_str = sales_col.fillna("").astype(str).str.strip().str.lower()
    sales_done = (sales_str != "") & (sales_str != "nan")
    sales_done_count = int(sales_done.sum())

    # Events
    subj = events.get("Subject_c", pd.Series([""] * len(events))).fillna("").astype(str).str.strip().str.lower()
    status = events.get("Appointment_Status_c", pd.Series([""] * len(events))).fillna("").astype(str).str.strip().str.lower()

    meeting_booked = (subj == "personal appointment booked").sum()
    meeting_done = ((subj == "personal appointment booked") & (status == "completed")).sum()

    # Ratios
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

# --------------------------------------------
# Source-Wise Funnel (Dynamic with Events)
# --------------------------------------------
# def compute_source_wise_funnel(
#     leads: pd.DataFrame,
#     opps: pd.DataFrame,
#     events: pd.DataFrame,
#     header_col="lead_source_c",
#     question: Optional[str] = None
# ) -> Dict[str, Any]:
#     """
#     Computes source-wise funnel with intelligent project filtering.

#     Supported queries:
#     - "source wise funnel" → all data, grouped by lead_source_c
#     - "source wise funnel for wmcc" → filters to "wmcc sec 32"
#     - "source wise funnel for wave city and wmcc" → combines both projects
#     - "project wise funnel" → groups by project_c
#     """
#     # Work on copies to avoid side effects
#     leads = leads.copy()
#     opps = opps.copy()
#     events = events.copy()

#     group_by_col = "lead_source_c"
#     target_projects: List[str] = []  # Canonical (exact DB) names
#     project_filter_applied = False

#     # ------------------------------------------------------------------
#     # 1. Detect projects from question using alias mapping
#     # ------------------------------------------------------------------
#     if question:
#         q_lower = question.lower().strip()

#         matched_canonical = set()
#         for alias, canonical in PROJECT_ALIASES.items():
#             if alias in q_lower:
#                 matched_canonical.add(canonical)

#         if matched_canonical:
#             target_projects = list(matched_canonical)
#             project_filter_applied = True
#             logger.info(f"Detected projects (via aliases): {target_projects}")

#         # Fallback: generic project keyword → project-wise view
#         if not target_projects and any(t in q_lower for t in GENERIC_PROJECT_TRIGGERS):
#             if any("project_c" in df.columns for df in [leads, opps, events]):
#                 group_by_col = "project_c"
#                 logger.info("Generic 'project' keyword → switching to project-wise funnel")

#     # ------------------------------------------------------------------
#     # 2. Filter data by target projects (if any detected)
#     # ------------------------------------------------------------------
#     if project_filter_applied and target_projects:
#         has_project_col = False
#         for df in (leads, opps, events):
#             if "project_c" in df.columns:
#                 has_project_col = True
#                 df["__proj_norm__"] = df["project_c"].fillna("").astype(str).str.strip().str.lower()
#             else:
#                 df["__proj_norm__"] = ""

#         if not has_project_col:
#             logger.warning("project_c column missing")
#             return {"error": "Project column not available"}

#         target_norm = [p.lower() for p in target_projects]

#         # Apply filter
#         leads = leads[leads["__proj_norm__"].isin(target_norm)].copy()
#         opps = opps[opps["__proj_norm__"].isin(target_norm)].copy()
#         events = events[events["__proj_norm__"].isin(target_norm)].copy()

#         # Cleanup
#         for df in (leads, opps, events):
#             df.drop(columns=["__proj_norm__"], inplace=True, errors="ignore")

#         if leads.empty and opps.empty:
#             return {
#                 "projects": [p.title() for p in target_projects],
#                 "analysis": "Source-wise funnel (combined)",
#                 "message": "No data found for selected projects",
#                 "sources": {"Overall": _get_empty_metrics()}
#             }

#     # ------------------------------------------------------------------
#     # 3. Grouping logic (source or project)
#     # ------------------------------------------------------------------
#     for df in (leads, opps):
#         if group_by_col not in df.columns:
#             df[group_by_col] = ""
#         df["__col_norm__"] = df[group_by_col].fillna("").astype(str).str.strip().str.lower()

#     # Preserve original display names
#     display_map = {}
#     for df in (leads, opps):
#         mask = df["__col_norm__"] != ""
#         if mask.any():
#             pairs = df.loc[mask, [group_by_col, "__col_norm__"]].drop_duplicates("__col_norm__")
#             for norm, orig in zip(pairs["__col_norm__"], pairs[group_by_col]):
#                 if norm not in display_map:
#                     display_map[norm] = str(orig).strip().title()

#     # OwnerId normalization for event matching
#     for df in (leads, events):
#         if "OwnerId" not in df.columns:
#             df["OwnerId"] = ""
#         df["__owner_norm__"] = df["OwnerId"].fillna("").astype(str).str.strip()

#     # Unique groups
#     all_groups = pd.concat([leads["__col_norm__"], opps["__col_norm__"]], ignore_index=True)
#     unique_groups = [g for g in all_groups.unique() if g]

#     sources_output: Dict[str, Any] = {}

#     for group_norm in unique_groups:
#         display_name = display_map.get(group_norm, group_norm.title())

#         leads_g = leads[leads["__col_norm__"] == group_norm].copy()
#         opps_g = opps[opps["__col_norm__"] == group_norm].copy()

#         owner_ids = leads_g["__owner_norm__"].unique()
#         events_g = events[events["__owner_norm__"].isin(owner_ids)].copy()

#         metrics = _compute_funnel_metrics_exact(leads_g, opps_g, events_g)
#         sources_output[display_name] = metrics

#     # Cleanup temporary columns
#     for df in (leads, opps, events):
#         df.drop(columns=["__col_norm__", "__owner_norm__"], inplace=True, errors="ignore")

#     # ------------------------------------------------------------------
#     # 4. Final response
#     # ------------------------------------------------------------------
#     result: Dict[str, Any] = {"sources": sources_output}

#     if project_filter_applied:
#         result.update({
#             "projects": [p.title() for p in target_projects],
#             "analysis": "Source-wise funnel for selected projects (combined)"
#         })
#     elif group_by_col == "project_c":
#         result["analysis"] = "Project-wise funnel"

#     logger.info("Funnel computation completed successfully")
#     return result



# def compute_source_wise_funnel(
#     leads: pd.DataFrame,
#     opps: pd.DataFrame,
#     events: pd.DataFrame,
#     header_col="lead_source_c",
#     question: Optional[str] = None
# ) -> Dict[str, Any]:
#     """
#     Smart funnel supporting:
#     - Source-wise (default)
#     - Project-wise (if "project" mentioned)
#     - Product-wise (if "product"/"digital"/etc. mentioned)
#     - Filtering: "source wise for digital and outdoor" → source-wise only for those products
#     """
#     leads = leads.copy()
#     opps = opps.copy()
#     events = events.copy()

#     group_by_col = "lead_source_c"
#     filter_mode = None  # None | "project" | "product"
#     target_values: List[str] = []  # canonical values to filter on

#     # ------------------------------------------------------------------
#     # 1. Intent Detection
#     # ------------------------------------------------------------------
#     if question:
#         q_lower = question.lower().strip()
        

#         # --- Product detection ---
#         matched_products = set()
#         for alias, canonical in PRODUCT_ALIASES.items():
#             if alias in q_lower:
#                 matched_products.add(canonical)

#         if matched_products:
#             filter_mode = "product"
#             target_values = list(matched_products)
#             logger.info(f"Product filter detected: {target_values}")

#         # --- Project detection ---
#         elif not matched_products:
#             matched_projects = set()
#             for alias, canonical in PROJECT_ALIASES.items():
#                 if alias in q_lower:
#                     matched_projects.add(canonical)

#             if matched_projects:
#                 filter_mode = "project"
#                 target_values = list(matched_projects)
#                 logger.info(f"Project filter detected: {target_values}")

#         # --- Generic triggers: switch grouping ---
#         if not filter_mode:
#             if any(t in q_lower for t in GENERIC_PRODUCT_TRIGGERS):
#                 group_by_col = "product_category_c"  # will use logic below to pick correct col
#                 logger.info("Generic product trigger → grouping by product category")
#             elif any(t in q_lower for t in GENERIC_PROJECT_TRIGGERS):
#                 group_by_col = "project_c"
#                 logger.info("Generic project trigger → grouping by project")

#     # ------------------------------------------------------------------
#     # 2. Apply filtering (project or product)
#     # ------------------------------------------------------------------
#     if filter_mode and target_values:
#         target_norm = [v.lower() for v in target_values]

#         if filter_mode == "project":
#             col_name = "project_c"
#         else:  # product
#             # We'll create a unified product column below
#             col_name = "__unified_product__"

#         # Add unified product column if filtering by product
#         if filter_mode == "product":
#             for df, prod_col in [(leads, "product_category_c"), (opps, "project_category_c"), (events, "product_category_c")]:
#                 if prod_col in df.columns:
#                     df["__unified_product__"] = df[prod_col].fillna("").astype(str).str.strip().str.lower()
#                 else:
#                     df["__unified_product__"] = ""

#             filter_col = "__unified_product__"
#         else:
#             # Project: use project_c if exists
#             filter_col = "project_c"
#             has_col = False
#             for df in (leads, opps, events):
#                 if filter_col in df.columns:
#                     has_col = True
#                     df["__temp_norm__"] = df[filter_col].fillna("").astype(str).str.strip().str.lower()
#                 else:
#                     df["__temp_norm__"] = ""

#             if not has_col:
#                 return {"error": "project_c column missing"}

#             filter_col = "__temp_norm__"

#         # Apply filter
#         leads = leads[leads[filter_col].isin(target_norm)].copy()
#         opps = opps[opps[filter_col].isin(target_norm)].copy()
#         events = events[events[filter_col].isin(target_norm)].copy()

#         # Cleanup temp filter columns
#         for df in (leads, opps, events):
#             df.drop(columns=[c for c in df.columns if c.startswith("__temp") or c == "__unified_product__"], inplace=True, errors="ignore")

#         if leads.empty and opps.empty:
#             return {
#                 "filter_type": filter_mode,
#                 "values": [v.title() for v in target_values],
#                 "analysis": "Source-wise funnel (filtered)",
#                 "message": "No data found",
#                 "sources": {"Overall": _get_empty_metrics()}
#             }

#     # ------------------------------------------------------------------
#     # 3. Determine final grouping column
#     # ------------------------------------------------------------------
#     if group_by_col == "product_category_c":
#         # Use unified product column for grouping
#         for df, prod_col in [(leads, "product_category_c"), (opps, "project_category_c"), (events, "product_category_c")]:
#             if prod_col in df.columns:
#                 df["__group_col__"] = df[prod_col].fillna("").astype(str).str.strip()
#             else:
#                 df["__group_col__"] = ""
#         group_by_col = "__group_col__"
#     else:
#         # Normal source or project_c
#         for df in (leads, opps):
#             if group_by_col not in df.columns:
#                 df[group_by_col] = ""

#     # Normalize grouping column
#     for df in (leads, opps):
#         df["__col_norm__"] = df[group_by_col].astype(str).str.strip().str.lower()

#     # Display name mapping
#     display_map = {}
#     for df in (leads, opps):
#         mask = df["__col_norm__"] != ""
#         if mask.any():
#             pairs = df.loc[mask, [group_by_col, "__col_norm__"]].drop_duplicates("__col_norm__")
#             for norm, orig in zip(pairs["__col_norm__"], pairs[group_by_col]):
#                 if norm not in display_map:
#                     display_map[norm] = str(orig).strip().title()

#     # OwnerId normalization
#     for df in (leads, events):
#         if "OwnerId" not in df.columns:
#             df["OwnerId"] = ""
#         df["__owner_norm__"] = df["OwnerId"].fillna("").astype(str).str.strip()

#     # Unique groups
#     all_groups = pd.concat([leads["__col_norm__"], opps["__col_norm__"]], ignore_index=True)
#     unique_groups = [g for g in all_groups.unique() if g]

#     sources_output: Dict[str, Any] = {}

#     for group_norm in unique_groups:
#         display_name = display_map.get(group_norm, group_norm.title())

#         leads_g = leads[leads["__col_norm__"] == group_norm].copy()
#         opps_g = opps[opps["__col_norm__"] == group_norm].copy()

#         owner_ids = leads_g["__owner_norm__"].unique()
#         events_g = events[events["__owner_norm__"].isin(owner_ids)].copy()

#         metrics = _compute_funnel_metrics_exact(leads_g, opps_g, events_g)
#         sources_output[display_name] = metrics

#     # Cleanup
#     for df in (leads, opps, events):
#         df.drop(columns=["__col_norm__", "__owner_norm__", "__group_col__"], inplace=True, errors="ignore")

#     # ------------------------------------------------------------------
#     # 4. Final Response
#     # ------------------------------------------------------------------
#     result: Dict[str, Any] = {"sources": sources_output}

#     if filter_mode == "product":
#         result.update({
#             "filter_type": "product",
#             "values": [v.title() for v in target_values],
#             "analysis": "Source-wise funnel for selected products"
#         })
#     elif filter_mode == "project":
#         result.update({
#             "filter_type": "project",
#             "values": [v.title() for v in target_values],
#             "analysis": "Source-wise funnel for selected projects"
#         })
#     elif group_by_col in ["product_category_c", "__group_col__"]:
#         result["analysis"] = "Product-wise funnel"
#     elif group_by_col == "project_c":
#         result["analysis"] = "Project-wise funnel"

#     return result


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



def compute_source_wise_funnel(
    leads: pd.DataFrame,
    opps: pd.DataFrame,
    events: pd.DataFrame,
    header_col="lead_source_c",
    question: Optional[str] = None
) -> Dict[str, Any]:
    leads = leads.copy()
    opps = opps.copy()
    events = events.copy()

    group_by_col = "lead_source_c"

    filter_project: List[str] = []
    filter_product: List[str] = []
    filter_source: List[str] = []

    # ------------------------------------------------------------------
    # 1. Intent Detection - Now detects all independently
    # ------------------------------------------------------------------
    if question:
        q_lower = question.lower().strip()

        # Source detection
        if any(word in q_lower for word in SOURCE_TRIGGER_WORDS):
            matched = {canonical for alias, canonical in SOURCE_ALIASES.items() if alias in q_lower}
            if matched:
                filter_source = list(matched)
                logger.info(f"Source filter: {filter_source}")

        # Project detection
        matched_proj = {canonical for alias, canonical in PROJECT_ALIASES.items() if alias in q_lower}
        if matched_proj:
            filter_project = list(matched_proj)
            logger.info(f"Project filter: {filter_project}")

        # Product detection
        matched_prod = {canonical for alias, canonical in PRODUCT_ALIASES.items() if alias in q_lower}
        if matched_prod:
            filter_product = list(matched_prod)
            logger.info(f"Product filter: {filter_product}")

        # Generic grouping fallback
        if not any([filter_source, filter_project, filter_product]):
            if any(t in q_lower for t in GENERIC_PRODUCT_TRIGGERS):
                group_by_col = "product_category_c"
            elif any(t in q_lower for t in GENERIC_PROJECT_TRIGGERS):
                group_by_col = "project_c"

    # ------------------------------------------------------------------
    # 2. Apply Filters - FIXED & WORKING
    # ------------------------------------------------------------------
    # Start with original
    filtered_leads = leads
    filtered_opps = opps
    filtered_events = events

    # Project filter
    if filter_project:
        norm_vals = [p.lower() for p in filter_project]
        for name, df in [("leads", filtered_leads), ("opps", filtered_opps), ("events", filtered_events)]:
            if "project_c" in df.columns:
                df["__temp__"] = df["project_c"].fillna("").astype(str).str.strip().str.lower()
                df = df[df["__temp__"].isin(norm_vals)].copy()
                df.drop(columns=["__temp__"], inplace=True)
            if name == "leads":
                filtered_leads = df
            elif name == "opps":
                filtered_opps = df
            else:
                filtered_events = df

    # Product filter
    if filter_product:
        norm_vals = [p.upper() for p in filter_product]
        for name, df, col in [
            ("leads", filtered_leads, "product_category_c"),
            ("opps", filtered_opps, "project_category_c"),
            ("events", filtered_events, "product_category_c")
        ]:
            if col in df.columns:
                df["__temp__"] = df[col].fillna("").astype(str).str.strip().str.upper()
                df = df[df["__temp__"].isin(norm_vals)].copy()
                df.drop(columns=["__temp__"], inplace=True)
            if name == "leads":
                filtered_leads = df
            elif name == "opps":
                filtered_opps = df
            else:
                filtered_events = df

    # Source filter
    if filter_source:
        norm_vals = [s.lower() for s in filter_source]
        for name, df in [("leads", filtered_leads), ("opps", filtered_opps)]:
            if "lead_source_c" in df.columns:
                df["__temp__"] = df["lead_source_c"].fillna("").astype(str).str.strip().str.lower()
                df = df[df["__temp__"].isin(norm_vals)].copy()
                df.drop(columns=["__temp__"], inplace=True)
            if name == "leads":
                filtered_leads = df
            else:
                filtered_opps = df

    # Assign back
    leads = filtered_leads
    opps = filtered_opps
    events = filtered_events

    # No data check
    if leads.empty and opps.empty:
        return {
            "analysis": "Filtered funnel",
            "filters_applied": {k: v for k, v in {
                "project": filter_project,
                "product": filter_product,
                "source": filter_source
            }.items() if v},
            "message": "No data found",
            "sources": {"Overall": _get_empty_metrics()}
        }

    # ------------------------------------------------------------------
    # 3. Grouping
    # ------------------------------------------------------------------
    if group_by_col == "product_category_c":
        for name, df, col in [
            ("leads", leads, "product_category_c"),
            ("opps", opps, "project_category_c"),
            ("events", events, "product_category_c")
        ]:
            if col in df.columns:
                df["__group__"] = df[col].fillna("").astype(str).str.strip()
            else:
                df["__group__"] = ""
        group_by_col = "__group__"

    # Ensure grouping column exists
    for df in (leads, opps):
        if group_by_col not in df.columns:
            df[group_by_col] = ""

    # Normalize
    for df in (leads, opps):
        df["__norm__"] = df[group_by_col].astype(str).str.strip().str.lower()

    # Display mapping
    display_map = {}
    for df in (leads, opps):
        mask = df["__norm__"] != ""
        if mask.any():
            pairs = df.loc[mask, [group_by_col, "__norm__"]].drop_duplicates("__norm__")
            for norm, orig in zip(pairs["__norm__"], pairs[group_by_col]):
                if norm not in display_map:
                    display_map[norm] = str(orig).strip().title()

    # OwnerId
    for df in (leads, events):
        if "OwnerId" not in df.columns:
            df["OwnerId"] = ""
        df["__owner__"] = df["OwnerId"].fillna("").astype(str).str.strip()

    # Unique groups
    all_groups = pd.concat([leads["__norm__"], opps["__norm__"]], ignore_index=True)
    unique_groups = [g for g in all_groups.unique() if g]

    # Compute metrics
    output = {}
    for g in unique_groups:
        name = display_map.get(g, g.title())
        l_g = leads[leads["__norm__"] == g].copy()
        o_g = opps[opps["__norm__"] == g].copy()
        owners = l_g["__owner__"].unique()
        e_g = events[events["__owner__"].isin(owners)].copy()
        output[name] = _compute_funnel_metrics_exact(l_g, o_g, e_g)

    # Cleanup
    for df in (leads, opps, events):
        df.drop(columns=[c for c in df.columns if c.startswith("__")], inplace=True, errors="ignore")

    # ------------------------------------------------------------------
    # 4. Response
    # ------------------------------------------------------------------
    result = {"sources": output}
    filters = {k: v for k, v in {
        "project": filter_project,
        "product": filter_product,
        "source": filter_source
    }.items() if v}
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


# ================================================================
# HELPER: Empty metrics
# ================================================================
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
        start = today.replace(day=1).strftime("%d-%m-%Y")
        end = today.strftime("%d-%m-%Y")
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
        if current_q == 1:
            start = f"01-04-{current_fy}"
            end = f"30-06-{current_fy}"
        elif current_q == 2:
            start = f"01-07-{current_fy}"
            end = f"30-09-{current_fy}"
        elif current_q == 3:
            start = f"01-10-{current_fy}"
            end = f"31-12-{current_fy}"
        else:  # current_q == 4
            start = f"01-01-{current_fy + 1}"
            end = f"31-03-{current_fy + 1}"

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
        start = f"01-04-{current_fy}"
        end = today.strftime("%d-%m-%Y")
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
        start = monday.strftime("%d-%m-%Y")
        end = today.strftime("%d-%m-%Y")
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
        return f"01-04-{fy_year}", f"31-03-{fy_year + 1}"

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

    # Generate all 4 quarters from Q1 to Q4
    def quarter_dates(q, fy_year):
        if q == 1:
            return f"01-04-{fy_year}", f"30-06-{fy_year}"
        elif q == 2:
            return f"01-07-{fy_year}", f"30-09-{fy_year}"
        elif q == 3:
            return f"01-10-{fy_year}", f"31-12-{fy_year}"
        elif q == 4:
            return f"01-01-{fy_year + 1}", f"31-03-{fy_year + 1}"

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
        "quarters": quarters  # Always Q1 → Q2 → Q3 → Q4
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
        fy = current_fy
        
        if "last year" in q or "previous year" in q:
            fy = current_fy - 1
        # If user gives FY explicitly (e.g. fy24)
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

def month_range_without_year(q:str):
    month_regex = (
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)"
    )

    m = re.search(
        rf"{month_regex}\s*(?:to|or|\-|&|\s)\s*{month_regex}(?:\s+(20\d{{2}}|\d{{2}}))?",
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
            "start_date":start_date, 
            "end_date":end_date
        }
    return None


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

    
    range_data  = month_year_range_parser(q)
    if range_data:
        return range_data


    # -------------------------------------------------------
    # 2️⃣ Detect explicit date range: "april 2024 to june 2024"
    # -------------------------------------------------------
    month_range = month_range_with_year(q)
    if month_range:
        return month_range["start_date"],month_range["end_date"]

        
    # -------------------------------------------------------
    # 6️⃣ Month range WITHOUT year (april to september)
    # -------------------------------------------------------
    month_range_wy = month_range_without_year(q)
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

# ----------------------------start-------------------------
def build_date_filter(date_ranges: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Build SQL date filter and human-readable label from list of (start_date, end_date) tuples.
    """
    if not date_ranges:
        raise ValueError("At least one date range must be provided")

    clauses = []
    labels = []
    for start_date, end_date in date_ranges:
        clause = f"""
            date_parse(replace(trim(created_date_c), '/', '-'), '%d-%m-%Y')
            BETWEEN date_parse('{start_date}', '%d-%m-%Y')
                AND date_parse('{end_date}', '%d-%m-%Y')
        """
        clauses.append(clause.strip())
        labels.append(f"{start_date} to {end_date}")

    if len(clauses) == 1:
        date_filter = f" WHERE {clauses[0]}"
        label = labels[0]
    else:
        date_filter = " WHERE (" + " OR ".join(clauses) + ")"
        label = ", ".join(labels)

    return date_filter.strip(), label


def execute_queries_for_period(date_filter: str) -> Tuple[Any, Any, Any]:
    """
    Execute all required Presto queries for a given date filter.
    Returns (leads_df, opps_df, events_df)
    """
    lead_sql = f"""
        SELECT lead_id_c, status, customer_feedback_c, created_date_c, lead_source_c, OwnerId,project_c,product_category_c
        FROM {CATALOG}.{LEAD_SCHEMA}.{LEAD_TABLE}
        {date_filter}
    """
    opp_sql = f"""
        SELECT opportunity_id_c, lead_id_c, sales_order_number_c, created_date_c, lead_source_c,project_c,project_category_c
        FROM {CATALOG}.{OPP_SCHEMA}.{OPP_TABLE}
        {date_filter}
    """
    event_sql = f"""
        SELECT OwnerId, Subject_c, Appointment_Status_c, created_date_c,project_c,product_category_c
        FROM {CATALOG}.{EVENT_SCHEMA}.{EVENT_TABLE}
        {date_filter}
    """

    try:
        leads = query_presto(CATALOG, LEAD_SCHEMA, lead_sql)
        opps = query_presto(CATALOG, OPP_SCHEMA, opp_sql)
        events = query_presto(CATALOG, EVENT_SCHEMA, event_sql)
        return leads, opps, events
    except Exception as e:
        logger.error(f"Presto query failed: {e}", exc_info=True)
        raise


def process_single_period(date_ranges: List[Tuple[str, str]], label: str, period_key: str = "period",question:Optional[str] = None) -> Dict[str, Any]:
    """
    Process a single logical period (could be one or multiple date ranges).
    """
    date_filter, period_label = build_date_filter(date_ranges)

    logger.info(f"Processing {label}: {period_label}")

    leads, opps, events = execute_queries_for_period(date_filter)

    if leads.empty:
        logger.warning(f"No leads found for {label}: {period_label}")
        # Optionally return partial result or skip — here we return empty funnel
        source_funnel = {}  # or compute empty structure
    else:
        source_funnel = compute_source_wise_funnel(leads, opps, events, header_col="lead_source_c",question=question)

    return {
        period_key: label,
        "period": period_label,
        "funnel": sort_funnel_by_numeric_desc(source_funnel, return_as_list=True)
    }
# -----------------------------end-------------------------

# --------------------------------------------
# Source-Wise Funnel API Endpoint
# --------------------------------------------
@app.post("/funnel/source/question")
async def source_funnel_from_question(payload: dict = Body(...)):
    question = payload.get("question", "").strip().lower()
    logger.info(f"Incoming /funnel/source/question request: {question}")

    try:
        # ------------------------------------
        # 1. QoQ Detection
        # ------------------------------------
        if qoq_result := detect_qoq(question):
            quarters = qoq_result["quarters"]
            analysis_type = qoq_result["type"]
            fy = qoq_result["fy"]

            logger.info(f"Detected {analysis_type.upper()} for FY{fy} → {len(quarters)} quarters")

            results = []
            for qtr in quarters:
                date_ranges = [(qtr["start_date"], qtr["end_date"])]
                result = process_single_period(date_ranges, qtr["quarter"], period_key="quarter",question=question)
                # Add totals calculation - handle nested sources structure
                funnel_data = result["funnel"].get("sources", result["funnel"]) if isinstance(result["funnel"], dict) else result["funnel"]
                result["totals"] = calculate_master_totals(funnel_data)
                results.append(result)

            return {
                "status": "success",
                "analysis_type": analysis_type,
                "fy": f"FY{fy}",
                "data": sort_funnel_by_numeric_desc(results)
            }

        # ------------------------------------
        # 2. YoY Detection
        # ------------------------------------
        elif yoy_result := detect_yoy(question):
            periods = yoy_result["periods"]

            logger.info(f"Running YoY analysis → {len(periods)} years")

            results = []
            for period in periods:
                date_ranges = [(period["start_date"], period["end_date"])]
                result = process_single_period(date_ranges, period["year"], period_key="quarter",question=question)
                # Add totals calculation - handle nested sources structure
                funnel_data = result["funnel"].get("sources", result["funnel"]) if isinstance(result["funnel"], dict) else result["funnel"]
                result["totals"] = calculate_master_totals(funnel_data)
                results.append(result)

            return {
                "status": "success",
                "analysis_type": "year_on_year",
                "fy": "Last 3 completed financial years",
                "data": sort_funnel_by_numeric_desc(results)
            }

        # ------------------------------------
        # 3. MoM Detection
        # ------------------------------------
        elif mom_result := detect_mom(question):
            periods = mom_result["periods"]

            logger.info(f"Running MoM analysis → {len(periods)} months")

            results = []
            for period in periods:
                date_ranges = [(period["start_date"], period["end_date"])]
                result = process_single_period(
                    date_ranges,
                    period["label"],
                    period_key="month",
                    question=question

                )
                result["period"] = period.get("period", result["period"])  # preserve original period label if needed
                # Add totals calculation - handle nested sources structure
                funnel_data = result["funnel"].get("sources", result["funnel"]) if isinstance(result["funnel"], dict) else result["funnel"]
                result["totals"] = calculate_master_totals(funnel_data)
                results.append(result)

            return {
                "status": "success",
                "analysis_type": "month_on_month",
                "comparison": "Last 6 months + Current MTD",
                "data": sort_funnel_by_numeric_desc(results)
            }
            
        elif (mul_year := parse_multi_year_date(question)):
            filters_label = ""
            yearly_results = []
            for start_str, end_str in mul_year:

                start_dt = datetime.strptime(start_str, "%d-%m-%Y")
                end_dt = datetime.strptime(end_str, "%d-%m-%Y")

                start_date = start_dt.strftime("%d-%m-%Y")
                end_date = end_dt.strftime("%d-%m-%Y")
                date_ranges = [(start_str, end_str)]
                lable = "MULTI YEAR"

                result = process_single_period(
                    date_ranges,
                    lable,
                    period_key="year",
                    question=question
                )
                funnel_data = result["funnel"].get("sources", result["funnel"]) if isinstance(result["funnel"], dict) else result["funnel"]
                result["totals"] = calculate_master_totals(funnel_data)
                yearly_results.append(result)

            return {
                "status": "success",
                "analysis_type": "year_on_year",
                "comparison": "Multi Year",
                "data": sort_funnel_by_numeric_desc(yearly_results)
            }
            
        elif (mul_month := parse_multi_month_date(question)):

            filters_label = ""
            combined_response = []
            for start_str, end_str in mul_month:

                date_ranges = [(start_str, end_str)]
                lable = "MULTI MONTH"
                result = process_single_period(
                    date_ranges,
                    lable,
                    period_key="month",
                    question=question

                )
                funnel_data = result["funnel"].get("sources", result["funnel"]) if isinstance(result["funnel"], dict) else result["funnel"]
                result["totals"] = calculate_master_totals(funnel_data)
                # result["period"] = period.get("period", result["period"])  # preserve original period label if needed
                combined_response.append(result)

            return {
                "status": "success",
                "analysis_type": "month_on_month",
                "comparison": "Multi month",
                "data": sort_funnel_by_numeric_desc(combined_response)
            }

        # ------------------------------------
        # 4. Default: Normal Date Parsing
        # ------------------------------------
        else:
            logger.info("Falling back to normal date parsing")

            date_result = parse_date_range(question)

            if isinstance(date_result, dict) and "ranges" in date_result:
                date_ranges = date_result["ranges"]
                filters_label = ", ".join([f"{s} to {e}" for s, e in date_ranges])
            else:
                start_date, end_date = date_result
                date_ranges = [(start_date, end_date)]
                filters_label = f"{start_date} to {end_date}"

            result = process_single_period(date_ranges, filters_label, period_key="period",question=question)
            
            # Handle nested sources structure
            funnel_data = result["funnel"].get("sources", result["funnel"]) if isinstance(result["funnel"], dict) else result["funnel"]

            return {
                "status": "success",
                "filters": filters_label,
                "source_wise_metrics": result["funnel"],
                "totals": calculate_master_totals(funnel_data)
            }

    except Exception as e:
        logger.error(f"Unexpected error in source_funnel_from_question: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# --------------------------------------------
# Health Check Endpoint
# --------------------------------------------
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Source-Wise Funnel Analytics API is running"}