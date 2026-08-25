# Top N / Bottom N Ranking Feature

## Overview
The funnel analytics API now supports filtering funnel results by ranking, allowing you to request only the top or bottom performing sources/projects/products by any numeric metric (default: Total Leads).

## Features

✅ **Top N Ranking**: Return only the N sources with the highest metric values  
✅ **Bottom N Ranking**: Return only the N sources with the lowest metric values  
✅ **Flexible Numbering**: Supports both numeric (1, 5, 10) and word numbers (one, five, ten)  
✅ **Universal Application**: Works with all analysis types (single period, MOM, QOQ, YOY, etc.)  
✅ **Automatic Sorting**: Results are sorted by Total Leads in descending order by default  
✅ **Response Metadata**: Includes filter information in API response  

## Usage Examples

### Basic Top N Queries

```json
{
  "question": "Show me top 5 sources"
}
```
**Returns**: The 5 sources with the highest Total Leads count

```json
{
  "question": "What are the bottom 3 performing sources?"
}
```
**Returns**: The 3 sources with the lowest Total Leads count

### Top N with Date Ranges

```json
{
  "question": "Show me the top 5 sources for April"
}
```
**Returns**: Top 5 sources for the specified date range

```json
{
  "question": "bottom 10 sources this quarter"
}
```
**Returns**: Bottom 10 sources for the current quarter

### Top N with Comparison Analysis

```json
{
  "question": "Top 5 sources month on month"
}
```
**Returns**: Top 5 sources for each month in the period

```json
{
  "question": "bottom 3 leads QoQ this year"
}
```
**Returns**: Bottom 3 sources for each quarter in the current financial year

### Top N with Project/Product Filtering

```json
{
  "question": "Top 10 sources for Wave City project"
}
```
**Returns**: Top 10 sources (filtered by Wave City project)

```json
{
  "question": "bottom 5 sources for AMORE product"
}
```
**Returns**: Bottom 5 sources (filtered by AMORE product)

## Supported Number Formats

### Numeric
- `top 1`, `top 5`, `top 10`, `top 100`, etc.
- `bottom 1`, `bottom 3`, `bottom 50`, etc.

### Word Numbers
- One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve
- Examples: `top five`, `bottom three`, `top ten`

### Case Insensitive
- Works with: `Top`, `TOP`, `top`, `Bottom`, `BOTTOM`, `bottom`

## Response Format

When a ranking filter is applied, the API response includes metadata about the filter:

```json
{
  "status": "success",
  "analysis_type": "single_period",
  "filter": "2024-04-01 to 2024-04-30",
  "source_wise_metrics": [
    {
      "name": "Direct",
      "Total Leads": 450,
      "Valid Leads": 380,
      "Junk Leads": 70,
      "SOL Leads (Interested)": 120,
      ...
    },
    {
      "name": "Digital",
      "Total Leads": 320,
      "Valid Leads": 260,
      "Junk Leads": 60,
      "SOL Leads (Interested)": 80,
      ...
    },
    ...
  ],
  "rank_filter_applied": {
    "type": "top",
    "n": 5,
    "sort_metric": "Total Leads"
  },
  "totals": {
    "Total Leads": 1500,
    ...
  }
}
```

### Response Fields

- **source_wise_metrics**: List containing only the top/bottom N sources
- **rank_filter_applied**: Object describing the applied filter:
  - `type`: Either `"top"` or `"bottom"`
  - `n`: The number of records returned
  - `sort_metric`: The metric used for sorting (always "Total Leads" currently)
- **totals**: Aggregated totals (includes all data for context)

## How It Works

1. **Extraction**: The API extracts the ranking request from your question using regex pattern matching
2. **Sorting**: The funnel data is sorted by Total Leads in descending order
3. **Filtering**: 
   - For **"top N"**: Returns the first N records (highest values)
   - For **"bottom N"**: Returns the last N records (lowest values)
4. **Response**: Returns the filtered list with metadata about the applied filter

## Implementation Details

### Functions Added

#### `extract_rank_filter(question: str) -> Optional[Tuple[str, int]]`
Extracts ranking instruction from question text.
- **Returns**: `("top", N)` or `("bottom", N)`, or `None` if no ranking found
- **Pattern**: `\b(top|bottom)\s+(\d+|word_number)\b`

#### `apply_rank_filter(data, rank_type, n, sort_metric="Total Leads")`
Applies the ranking filter to funnel results.
- **Parameters**:
  - `data`: List of funnel records
  - `rank_type`: "top" or "bottom"
  - `n`: Number of records to return
  - `sort_metric`: Metric to sort by
- **Returns**: Filtered list of N records

### Integration Points

- **`_run_period()`**: Applies filter after computing funnel for each period
- **`_build_single_response()`**: Includes filter metadata in response
- **`_build_multi_response()`**: Applies filter across all periods
- **`_build_comparison_sections()`**: Passes question to enable filtering in comparisons

## Examples with Real API Calls

### Example 1: Get top 5 sources for April 2024

**Request**:
```bash
curl -X POST http://localhost:8000/funnel/source/question \
  -H "Content-Type: application/json" \
  -d '{"question": "top 5 sources for April 2024"}'
```

**Response**:
```json
{
  "status": "success",
  "analysis_type": "single_period",
  "source_wise_metrics": [
    {"name": "Direct", "Total Leads": 450, ...},
    {"name": "Digital", "Total Leads": 320, ...},
    {"name": "Referral", "Total Leads": 280, ...},
    {"name": "Channel Partner", "Total Leads": 200, ...},
    {"name": "Reference Sale", "Total Leads": 150, ...}
  ],
  "rank_filter_applied": {
    "type": "top",
    "n": 5,
    "sort_metric": "Total Leads"
  }
}
```

### Example 2: Get bottom 3 sources month-on-month

**Request**:
```bash
curl -X POST http://localhost:8000/funnel/source/question \
  -H "Content-Type: application/json" \
  -d '{"question": "bottom 3 sources mom this quarter"}'
```

**Response**:
```json
{
  "status": "success",
  "analysis_type": "month_on_month",
  "data": [
    {
      "label": "Apr 2026",
      "funnel": [
        {"name": "Shifting", "Total Leads": 25, ...},
        {"name": "Events / Exhibitions", "Total Leads": 40, ...},
        {"name": "SMS Campaign", "Total Leads": 55, ...}
      ],
      "totals": {...}
    },
    {
      "label": "May 2026",
      "funnel": [
        {"name": "Shifting", "Total Leads": 18, ...},
        {"name": "Transfered Unit", "Total Leads": 32, ...},
        {"name": "SMS Campaign", "Total Leads": 50, ...}
      ],
      "totals": {...}
    },
    ...
  ],
  "rank_filter_applied": {
    "type": "bottom",
    "n": 3,
    "sort_metric": "Total Leads"
  }
}
```

## Limitations & Notes

- **Default Sort Metric**: Currently sorts by "Total Leads". Future versions may support custom sort metrics.
- **Edge Cases**:
  - If N is larger than the number of available sources, all sources are returned
  - If N is 0 or negative, all data is returned (filter is ignored)
  - Empty results return empty list
- **Data Consistency**: Totals in the response include all source data, not just the top/bottom N (for context)
- **Multi-Period Analysis**: Filter is applied independently to each period when used with MOM, QOQ, YOY

## Testing the Feature

### Test Case 1: Simple Top N
```
Question: "top 5 sources"
Expected: 5 records with highest Total Leads
```

### Test Case 2: Word Number
```
Question: "bottom three"
Expected: 3 records with lowest Total Leads
```

### Test Case 3: Complex Query
```
Question: "top 10 sources for April to June mom"
Expected: Top 10 sources for April, May, and June separately
```

### Test Case 4: With Filters
```
Question: "top 5 sources for Wave City"
Expected: 5 sources filtered by Wave City project, sorted by Total Leads
```

## Future Enhancements

Potential improvements for future versions:
- Custom sort metric (sort by "Meeting Done", "Sales Done", etc.)
- Percentile-based ranking (top 10%, bottom 25%, etc.)
- Ranking by multiple metrics
- Reverse ranking (second-highest, third-lowest, etc.)
- Tied records handling (all records with same value as Nth position)

## API Compatibility

The ranking feature is fully backward compatible:
- Queries without ranking filters work exactly as before
- When no ranking filter is detected, all results are returned
- Existing integrations are not affected
