"""
Usage tracking utilities for logging user actions to the central app usage table.
"""

import os
import streamlit as st
from datetime import datetime
from src.utils.db_connection import with_connection

def get_env_or_secret(key):
    """Get environment variable from .env first, fallback to st.secrets"""
    value = os.getenv(key)
    if value is None:
        try:
            value = st.secrets.get(key)
        except:
            value = None
    return value

# Central usage table configuration
UC_CATALOG = "elevio_group___rag_pilot"
UC_SCHEMA = "rag_core_blocks"
USAGE_TABLE = f"`{UC_CATALOG}`.`{UC_SCHEMA}`.central_app_usage"
APP_NAME = "elevio-ctg-chatbot-vanna"

@with_connection
def log_user_action(user_email, action, conn=None, uc_catalog=None, uc_schema=None):
    """
    Log a user action to the central app usage table.
    
    Args:
        user_email (str): Email of the user performing the action
        action (str): Description of the action performed
        conn: Database connection (provided by decorator)
        uc_catalog (str): Unity Catalog name (provided by decorator)
        uc_schema (str): Schema name (provided by decorator)
    """
    try:
        timestamp = datetime.now()
        
        with conn.cursor() as cursor:
            query = f"""
                INSERT INTO {USAGE_TABLE}
                (timestamp, app_name, user, action)
                VALUES (?, ?, ?, ?)
            """
            cursor.execute(query, (timestamp, APP_NAME, user_email, action))
            
        # Optional: Log to console for debugging
        debug_logging = False
        try:
            debug_logging = st.secrets.get("DEBUG_LOGGING", False) or os.getenv("DEBUG_LOGGING", "false").lower() == "true"
        except:
            pass
            
        if debug_logging:
            print(f"DEBUG: Logged action - User: {user_email}, Action: {action}, Timestamp: {timestamp}")
            
    except Exception as e:
        # Don't break the app if logging fails, but log the error silently
        print(f"Warning: Failed to log user action: {e}")
        # Only show error in debug mode
        try:
            debug_logging = st.secrets.get("DEBUG_LOGGING", False) or os.getenv("DEBUG_LOGGING", "false").lower() == "true"
            if debug_logging:
                st.error(f"Warning: Failed to log user action: {e}")
        except:
            pass


def track_user_action_simple(user_email, action):
    """
    Simple function to track user actions without decorator.
    
    Args:
        user_email (str): Email of the user performing the action
        action (str): Description of the action performed
    """
    try:
        # Convert action to string if it's not already
        action_str = str(action) if action is not None else "unknown_action"
        
        log_user_action(
            user_email=user_email,
            action=action_str,
            db_hostname=get_env_or_secret('DATABRICKS_SERVER_HOSTNAME'),
            db_path=get_env_or_secret('DATABRICKS_HTTP_PATH'),
            db_token=get_env_or_secret('DATABRICKS_ACCESS_TOKEN'),
            uc_catalog=UC_CATALOG,
            uc_schema=UC_SCHEMA
        )
    except Exception as e:
        # Silent fail - don't break the app if tracking fails
        print(f"Warning: Failed to track user action: {e}")
        # Only show error in debug mode
        try:
            debug_logging = st.secrets.get("DEBUG_LOGGING", False) or os.getenv("DEBUG_LOGGING", "false").lower() == "true"
            if debug_logging:
                st.error(f"Warning: Failed to track user action: {e}")
        except:
            pass

# Common action names for consistency
class Actions:
    """Predefined action names for consistency across the app"""
    # App lifecycle
    APP_START = "app_start"
    APP_ERROR = "app_error"
    
    # Query operations
    GENERATE_QUESTION = "generate_question"
    SQL_QUERY = "sql_query"
    EXECUTE_SQL = "execute_sql"
    EDIT_SQL = "edit_sql"
    
    # Data operations
    DOWNLOAD_RESULTS = "download_results"
    GENERATE_CHART = "generate_chart"
    GENERATE_SUMMARY = "generate_summary"
    
    # User interactions
    RANDOM_QUESTIONS = "view_random_questions"
    SELECT_COLUMNS = "select_columns"
    
    # Error tracking
    DATABASE_ERROR = "database_error"
    LLM_ERROR = "llm_error"
    SQL_ERROR = "sql_error"
    CHART_ERROR = "chart_error"


def track_error(user_email, error_type, error_message=None):
    """
    Track an error that occurred in the application.
    
    Args:
        user_email (str): Email of the user who encountered the error
        error_type (str): Type of error (use Actions constants)
        error_message (str, optional): Additional error details
    """
    action = f"{error_type}"
    if error_message:
        action += f": {str(error_message)[:200]}"  # Limit error message length
    
    track_user_action_simple(user_email, action)


def get_usage_stats(user_email=None, days=30):
    """
    Get usage statistics from the database (if needed for analytics).
    
    Args:
        user_email (str, optional): Filter by specific user
        days (int): Number of days to look back
        
    Returns:
        dict: Usage statistics
    """
    try:
        @with_connection
        def _get_stats(conn=None, uc_catalog=None, uc_schema=None):
            with conn.cursor() as cursor:
                # Base query
                query = f"""
                    SELECT action, COUNT(*) as count, DATE(timestamp) as date
                    FROM {USAGE_TABLE}
                    WHERE timestamp >= CURRENT_DATE - INTERVAL {days} DAY
                    AND app_name = '{APP_NAME}'
                """
                
                # Add user filter if specified
                if user_email:
                    query += f" AND user = '{user_email}'"
                
                query += " GROUP BY action, DATE(timestamp) ORDER BY date DESC, count DESC"
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                return [{"action": row[0], "count": row[1], "date": row[2]} for row in results]
        
        return _get_stats(
            db_hostname=get_env_or_secret('DATABRICKS_SERVER_HOSTNAME'),
            db_path=get_env_or_secret('DATABRICKS_HTTP_PATH'),
            db_token=get_env_or_secret('DATABRICKS_ACCESS_TOKEN'),
            uc_catalog=UC_CATALOG,
            uc_schema=UC_SCHEMA
        )
        
    except Exception as e:
        print(f"Warning: Failed to get usage stats: {e}")
        return []
