#!/usr/bin/env python3
"""
Test script for usage tracking functionality in elevio-ctg-chatbot-vanna.
"""

import os
import sys
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.usage_tracker import track_user_action_simple, Actions, track_error, get_usage_stats

def test_basic_tracking():
    """Test basic usage tracking functionality"""
    print("Testing basic usage tracking...")
    
    test_user = "test@example.com"
    
    # Test various actions
    test_actions = [
        Actions.APP_START,
        Actions.GENERATE_QUESTION,
        Actions.SQL_QUERY,
        Actions.EXECUTE_SQL,
        Actions.EDIT_SQL,
        Actions.DOWNLOAD_RESULTS,
        Actions.GENERATE_CHART,
        Actions.GENERATE_SUMMARY,
        Actions.RANDOM_QUESTIONS,
        Actions.SELECT_COLUMNS
    ]
    
    print(f"Tracking {len(test_actions)} actions for user: {test_user}")
    
    for action in test_actions:
        try:
            track_user_action_simple(test_user, action)
            print(f"✅ Successfully tracked: {action}")
        except Exception as e:
            print(f"❌ Failed to track {action}: {e}")
    
    # Test error tracking
    print("\nTesting error tracking...")
    try:
        track_error(test_user, Actions.SQL_ERROR, "Test SQL error message")
        print("✅ Successfully tracked error")
    except Exception as e:
        print(f"❌ Failed to track error: {e}")

def test_usage_stats():
    """Test retrieving usage statistics"""
    print("\nTesting usage statistics retrieval...")
    
    try:
        stats = get_usage_stats(days=1)  # Get stats for last day
        print(f"✅ Retrieved {len(stats)} usage statistics")
        
        if stats:
            print("\nSample statistics:")
            for stat in stats[:5]:  # Show first 5
                print(f"  - {stat['action']}: {stat['count']} times on {stat['date']}")
        else:
            print("No statistics found (this is normal for a new installation)")
            
    except Exception as e:
        print(f"❌ Failed to retrieve usage stats: {e}")

def main():
    """Main test function"""
    print("=" * 60)
    print("USAGE TRACKING TEST SCRIPT - CTG CHATBOT VANNA")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Run tests
    test_basic_tracking()
    test_usage_stats()
    
    print()
    print("=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
