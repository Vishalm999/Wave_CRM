"""
Test script to verify the mom_period fix for "last quarter" queries.
This tests the date calculation logic without requiring database access.
"""

from datetime import date, datetime
from calendar import monthrange


def _today():
    """Return current datetime (for testing with May 17, 2026)"""
    # For testing: May 17, 2026
    return datetime(2026, 5, 17)


def _current_fy():
    """Calculate current financial year (April-March)"""
    t = _today()
    return t.year if t.month >= 4 else t.year - 1


def _fy_quarter(month: int) -> int:
    """
    Calculate which FY quarter a given month falls into.
    Q1: Apr-May-Jun (months 4-6)
    Q2: Jul-Aug-Sep (months 7-9)
    Q3: Oct-Nov-Dec (months 10-12)
    Q4: Jan-Feb-Mar (months 1-3)
    """
    for quarter, rng in enumerate((range(4, 7), range(7, 10), range(10, 13)), start=1):
        if month in rng:
            return quarter
    return 4


def _quarter_dates(q: int, fy: int):
    """Get start and end dates for a given quarter in a given FY"""
    mapping = {
        1: (date(fy, 4, 1),      date(fy, 6, 30)),
        2: (date(fy, 7, 1),      date(fy, 9, 30)),
        3: (date(fy, 10, 1),     date(fy, 12, 31)),
        4: (date(fy + 1, 1, 1),  date(fy + 1, 3, 31)),
    }
    return mapping[q]


def test_last_quarter_calculation():
    """Test that 'last quarter' returns correct date range when in May (FY2026 Q1)"""
    now = _today()
    today = now.date()
    fy = _current_fy()
    
    print("=" * 60)
    print("TEST: MOM Period for 'last quarter' in May 2026")
    print("=" * 60)
    
    print(f"\nCurrent Date: {today}")
    print(f"Current FY: {fy}")
    print(f"Current Month: {today.month}")
    
    # Calculate current quarter
    curr_q = _fy_quarter(today.month)
    print(f"Current Quarter: Q{curr_q}")
    
    # Calculate last quarter
    if curr_q == 1:
        prev_q = 4
        prev_fy = fy - 1
    else:
        prev_q = curr_q - 1
        prev_fy = fy
    
    print(f"\nPrevious Quarter: Q{prev_q} FY{prev_fy}")
    
    # Get the date range for last quarter
    s, e = _quarter_dates(prev_q, prev_fy)
    print(f"Date Range: {s} to {e}")
    
    # Generate monthly periods
    print("\nGenerated Monthly Periods:")
    print("-" * 60)
    
    periods = []
    cur = s.replace(day=1)
    month_count = 0
    
    while cur <= e:
        def month_end(year, month):
            return date(year, month, monthrange(year, month)[1])
        
        me = month_end(cur.year, cur.month)
        end = min(me, e, today)
        lbl = cur.strftime("%b %Y")
        if cur.year == today.year and cur.month == today.month:
            lbl += " (MTD)"
        
        print(f"  {month_count + 1}. {lbl:20} | {cur} to {end}")
        periods.append((lbl, cur, end))
        month_count += 1
        
        # Advance to next month
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    
    # Verify results
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print("=" * 60)
    
    # Expected: Jan 2026, Feb 2026, Mar 2026
    expected_months = ["Jan", "Feb", "Mar"]
    expected_year = 2026
    
    print(f"\nExpected months: {', '.join(expected_months)} {expected_year}")
    print(f"Actual months: {', '.join([p[0].split()[0] for p in periods])} {periods[0][0].split()[1]}")
    
    if len(periods) == 3:
        period_labels = [p[0].split()[0] for p in periods]
        if period_labels == expected_months:
            print("\n✓ SUCCESS: Correctly returning Jan-Mar 2026 for last quarter!")
            return True
        else:
            print(f"\n✗ FAILED: Got {period_labels} instead of {expected_months}")
            return False
    else:
        print(f"\n✗ FAILED: Expected 3 months, got {len(periods)}")
        return False


if __name__ == "__main__":
    success = test_last_quarter_calculation()
    exit(0 if success else 1)
