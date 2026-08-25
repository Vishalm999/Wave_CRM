#!/usr/bin/env python3
"""Validate deterministic "last N quarters" date resolution."""

from datetime import datetime

from sales_user_funnel import _last_n_quarter_periods


def _as_tuples(question: str, now: datetime):
    return [
        (period.label, period.start.isoformat(), period.end.isoformat())
        for period in _last_n_quarter_periods(question, now)
    ]


def test_last_n_quarters():
    cases = [
        (
            "June 9, 2026 (Q1 FY2026), last 2 quarters",
            "last 2 quarters",
            datetime(2026, 6, 9),
            [
                ("Q3 FY2025", "2025-10-01", "2025-12-31"),
                ("Q4 FY2025", "2026-01-01", "2026-03-31"),
            ],
        ),
        (
            "June 9, 2026 (Q1 FY2026), last 4 quarters",
            "last 4 quarters",
            datetime(2026, 6, 9),
            [
                ("Q1 FY2025", "2025-04-01", "2025-06-30"),
                ("Q2 FY2025", "2025-07-01", "2025-09-30"),
                ("Q3 FY2025", "2025-10-01", "2025-12-31"),
                ("Q4 FY2025", "2026-01-01", "2026-03-31"),
            ],
        ),
        (
            "September 15, 2026 (Q2 FY2026), previous 2 quarters",
            "previous 2 quarters",
            datetime(2026, 9, 15),
            [
                ("Q4 FY2025", "2026-01-01", "2026-03-31"),
                ("Q1 FY2026", "2026-04-01", "2026-06-30"),
            ],
        ),
        (
            "December 1, 2025 (Q3 FY2025), past three quarters",
            "past three quarters",
            datetime(2025, 12, 1),
            [
                ("Q4 FY2024", "2025-01-01", "2025-03-31"),
                ("Q1 FY2025", "2025-04-01", "2025-06-30"),
                ("Q2 FY2025", "2025-07-01", "2025-09-30"),
            ],
        ),
        (
            "January 15, 2026 (Q4 FY2025), last 1 quarter",
            "last 1 quarter",
            datetime(2026, 1, 15),
            [("Q3 FY2025", "2025-10-01", "2025-12-31")],
        ),
    ]

    for name, question, now, expected in cases:
        result = _as_tuples(question, now)
        assert result == expected, f"{name}: expected {expected}, got {result}"


if __name__ == "__main__":
    test_last_n_quarters()
    print("All last-N-quarter tests passed.")
