#!/usr/bin/env python3
"""
Test script for Top N / Bottom N ranking feature.
This demonstrates how the ranking filters work in the funnel API.
"""

import sys
from typing import Optional, Tuple, List, Dict, Any

# Mock the WORD_NUM dictionary (from the main file)
WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12
}

def extract_rank_filter(question: str) -> Optional[Tuple[str, int]]:
    """
    Extract Top N or Bottom N ranking filter from question.
    Returns: ("top", N) or ("bottom", N), or None if no ranking filter found.
    """
    import re
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
    """Apply top/bottom N filtering to funnel results list."""
    if not isinstance(data, list) or not data or n <= 0:
        return data
    
    # Sort by the specified metric in descending order
    def get_metric_value(item):
        if isinstance(item, dict):
            value = item.get(sort_metric, 0)
            return value if isinstance(value, (int, float)) else 0
        return 0
    
    sorted_data = sorted(data, key=get_metric_value, reverse=True)
    
    # Return top or bottom N
    if rank_type.lower() == "top":
        return sorted_data[:n]
    elif rank_type.lower() == "bottom":
        return sorted_data[-n:] if n < len(sorted_data) else sorted_data
    
    return data


def test_extraction():
    """Test extract_rank_filter function."""
    print("=" * 70)
    print("Testing extract_rank_filter() function")
    print("=" * 70)
    
    test_cases = [
        ("top 5 sources", ("top", 5)),
        ("Show me top 10", ("top", 10)),
        ("bottom 3 leads", ("bottom", 3)),
        ("top five sources", ("top", 5)),
        ("bottom ten", ("bottom", 10)),
        ("Top 2 projects", ("top", 2)),
        ("BOTTOM 7", ("bottom", 7)),
        ("show me top 1", ("top", 1)),
        ("give me bottom 12", ("bottom", 12)),
        ("no ranking here", None),
        ("just some question", None),
        ("Top N sources", None),  # N is not a valid number
    ]
    
    passed = 0
    failed = 0
    
    for question, expected in test_cases:
        result = extract_rank_filter(question)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status}: '{question}' → {result} (expected: {expected})")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_filtering():
    """Test apply_rank_filter function."""
    print("\n" + "=" * 70)
    print("Testing apply_rank_filter() function")
    print("=" * 70)
    
    # Mock funnel data
    mock_data = [
        {"name": "Direct", "Total Leads": 450},
        {"name": "Digital", "Total Leads": 320},
        {"name": "Referral", "Total Leads": 280},
        {"name": "Channel Partner", "Total Leads": 200},
        {"name": "Reference Sale", "Total Leads": 150},
        {"name": "Events", "Total Leads": 120},
        {"name": "Print Media", "Total Leads": 80},
        {"name": "SMS Campaign", "Total Leads": 60},
        {"name": "Shifting", "Total Leads": 30},
        {"name": "Word of mouth", "Total Leads": 20},
    ]
    
    print(f"\nMock data: {len(mock_data)} sources")
    print("Sources sorted by Total Leads (descending):")
    for i, item in enumerate(mock_data, 1):
        print(f"  {i}. {item['name']}: {item['Total Leads']}")
    
    # Test case 1: Top 5
    print("\n--- Test Case 1: Top 5 sources ---")
    result = apply_rank_filter(mock_data, "top", 5)
    print(f"Result count: {len(result)}")
    for i, item in enumerate(result, 1):
        print(f"  {i}. {item['name']}: {item['Total Leads']}")
    assert len(result) == 5
    assert result[0]["name"] == "Direct"
    print("✓ PASS: Top 5 filtering works correctly")
    
    # Test case 2: Bottom 3
    print("\n--- Test Case 2: Bottom 3 sources ---")
    result = apply_rank_filter(mock_data, "bottom", 3)
    print(f"Result count: {len(result)}")
    for i, item in enumerate(result, 1):
        print(f"  {i}. {item['name']}: {item['Total Leads']}")
    assert len(result) == 3
    # The last 3 items from descending sorted list are: SMS Campaign (60), Shifting (30), Word of mouth (20)
    assert result[-1]["name"] == "Word of mouth"  # lowest value
    print("✓ PASS: Bottom 3 filtering works correctly")
    
    # Test case 3: Top 0 (should return all)
    print("\n--- Test Case 3: Top 0 (invalid) ---")
    result = apply_rank_filter(mock_data, "top", 0)
    print(f"Result count: {len(result)}")
    assert len(result) == len(mock_data)
    print("✓ PASS: Invalid N returns all data")
    
    # Test case 4: Top 100 (more than available)
    print("\n--- Test Case 4: Top 100 (more than available) ---")
    result = apply_rank_filter(mock_data, "top", 100)
    print(f"Result count: {len(result)}")
    assert len(result) == len(mock_data)
    print("✓ PASS: N larger than available returns all data")
    
    print("\n✓ All filtering tests passed!")
    return True


def test_integration():
    """Test integrated extraction and filtering."""
    print("\n" + "=" * 70)
    print("Testing Integrated Extraction + Filtering")
    print("=" * 70)
    
    mock_data = [
        {"name": "Direct", "Total Leads": 450},
        {"name": "Digital", "Total Leads": 320},
        {"name": "Referral", "Total Leads": 280},
        {"name": "Channel Partner", "Total Leads": 200},
        {"name": "Reference Sale", "Total Leads": 150},
    ]
    
    questions = [
        ("top 3 sources", "top", 3),
        ("bottom two", "bottom", 2),
        ("show me top 5", "top", 5),
    ]
    
    for question, expected_type, expected_n in questions:
        print(f"\nQuestion: '{question}'")
        rank_filter = extract_rank_filter(question)
        if rank_filter:
            rank_type, n = rank_filter
            result = apply_rank_filter(mock_data, rank_type, n)
            print(f"  Extracted filter: type={rank_type}, n={n}")
            print(f"  Returned {len(result)} records")
            assert rank_type == expected_type
            assert n == expected_n
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL: Could not extract filter")
            return False
    
    print("\n✓ All integration tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TOP N / BOTTOM N RANKING FEATURE - TEST SUITE")
    print("=" * 70)
    
    all_passed = True
    
    try:
        all_passed &= test_extraction()
        all_passed &= test_filtering()
        all_passed &= test_integration()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
