# Usage Tracking Implementation - CTG Chatbot Vanna

This document describes the implementation of user action tracking in the Elevio CTG Chatbot Vanna application.

## Overview

User actions are tracked and saved to a central Databricks table to provide insights into how users interact with the application. The tracking system is designed to be non-intrusive and fail-safe - if tracking fails, it won't break the user experience.

## Database Table Structure

The central tracking table has the following structure:

```sql
CREATE TABLE IF NOT EXISTS elevio_group___rag_pilot.rag_core_blocks.central_app_usage (
  timestamp TIMESTAMP,
  app_name STRING,
  user STRING,
  action STRING
)
USING DELTA;
```

**Fields:**
- `timestamp`: When the action occurred
- `app_name`: Always "elevio-ctg-chatbot-vanna" for this application
- `user`: Email address of the user performing the action (or default identifier)
- `action`: Description of the action performed

## Implementation Files

### 1. `src/utils/usage_tracker.py`
Main tracking implementation with the following key components:

- **`track_user_action_simple(user_email, action)`**: Simple function to track any user action
- **`track_error(user_email, error_type, error_message)`**: Specialized function for tracking errors
- **`Actions` class**: Predefined constants for consistent action naming
- **`get_usage_stats(user_email, days)`**: Function to retrieve usage statistics

### 2. `src/utils/db_connection.py`
Database connection utilities for connecting to Databricks.

### 3. `app.py`
Integration points throughout the main application:

- App startup tracking
- User query generation
- SQL query generation and execution
- SQL editing
- Data downloads
- Summary generation
- Error tracking for various failure modes

## Tracked Actions

### App Lifecycle
- `app_start`: User starts the application
- `app_error`: General application errors

### Query Operations
- `generate_question`: User submits a question/query
- `sql_query`: SQL query is generated from the question
- `execute_sql`: SQL query is executed
- `edit_sql`: User edits the generated SQL

### Data Operations
- `download_results`: User downloads query results as Excel
- `generate_chart`: Chart generation (when enabled)
- `generate_summary`: Summary generation

### User Interactions
- `view_random_questions`: User clicks on suggested questions
- `select_columns`: User selects specific columns to include

### Error Tracking
- `database_error`: Database operation failures
- `llm_error`: LLM API call failures
- `sql_error`: SQL query validation or execution failures
- `chart_error`: Chart generation failures

## Configuration

### Environment Variables
The tracking system requires the following Databricks connection variables:
- `DATABRICKS_SERVER_HOSTNAME`
- `DATABRICKS_HTTP_PATH`
- `DATABRICKS_ACCESS_TOKEN`

### Debug Mode
Set `DEBUG_LOGGING=true` in environment variables or Streamlit secrets to enable debug output.

## User Identification

This application uses a simplified user identification system:
- Attempts to get user email from Streamlit context headers (if available in production)
- Falls back to a default identifier: `ctg_chatbot_user@eleviogroup.com`
- In production, this should be replaced with proper authentication

## Error Handling

The tracking system is designed to be fail-safe:
- If tracking fails, the main application continues normally
- Errors are logged to console but not shown to users (unless in debug mode)
- Database connection issues don't interrupt user workflows

## Testing

Use the provided test script to verify tracking functionality:

```bash
python test_usage_tracking.py
```

This will:
- Test basic action tracking
- Test error tracking
- Test usage statistics retrieval
- Provide feedback on what's working

## Analytics and Insights

The tracked data can be used for:

1. **User Engagement Analysis**
   - Most popular query patterns
   - SQL editing frequency
   - Download patterns

2. **Performance Monitoring**
   - Error rates and types
   - SQL generation success rates
   - User experience issues

3. **Product Development**
   - Feature usage patterns
   - User workflow optimization
   - Priority setting for improvements

## Example Queries

Here are some useful SQL queries for analyzing the tracked data:

```sql
-- Daily active users
SELECT DATE(timestamp) as date, COUNT(DISTINCT user) as active_users
FROM elevio_group___rag_pilot.rag_core_blocks.central_app_usage
WHERE app_name = 'elevio-ctg-chatbot-vanna'
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Most popular actions
SELECT action, COUNT(*) as count
FROM elevio_group___rag_pilot.rag_core_blocks.central_app_usage
WHERE app_name = 'elevio-ctg-chatbot-vanna'
  AND timestamp >= CURRENT_DATE - INTERVAL 7 DAY
GROUP BY action
ORDER BY count DESC;

-- SQL editing patterns
SELECT action, COUNT(*) as count
FROM elevio_group___rag_pilot.rag_core_blocks.central_app_usage
WHERE app_name = 'elevio-ctg-chatbot-vanna'
  AND action IN ('sql_query', 'edit_sql', 'execute_sql')
  AND timestamp >= CURRENT_DATE - INTERVAL 7 DAY
GROUP BY action
ORDER BY count DESC;

-- Error analysis
SELECT action, COUNT(*) as error_count
FROM elevio_group___rag_pilot.rag_core_blocks.central_app_usage
WHERE app_name = 'elevio-ctg-chatbot-vanna'
  AND action LIKE '%_error%'
  AND timestamp >= CURRENT_DATE - INTERVAL 1 DAY
GROUP BY action
ORDER BY error_count DESC;
```

## Maintenance

- The tracking system requires no regular maintenance
- Data retention policies should be set according to organizational requirements
- Monitor the central_app_usage table size and performance
- Review tracked actions periodically to ensure they provide valuable insights
