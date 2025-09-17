import functools
from databricks import sql
import streamlit as st
import os

class DbConnectionManager:
    def __init__(self, hostname, path, token, catalog, schema):
        # Validate required parameters
        if not hostname:
            raise ValueError("DATABRICKS_SERVER_HOSTNAME is required but not set")
        if not path:
            raise ValueError("DATABRICKS_HTTP_PATH is required but not set")
        if not token:
            raise ValueError("DATABRICKS_ACCESS_TOKEN is required but not set")
            
        self.hostname = hostname
        self.path = path
        self.token = token
        self.catalog = catalog
        self.schema = schema
        self._connection = None
        self._is_closed = True
    
    def get_connection(self):
        if self._connection is None or self._is_closed:
            try:
                if self._connection is not None:
                    # Try to close the old connection just in case
                    try:
                        self._connection.close()
                    except:
                        pass
                
                self._connection = sql.connect(
                    server_hostname=self.hostname,
                    http_path=self.path,
                    access_token=self.token
                )
                self._is_closed = False
            except Exception as e:
                st.error(f"Failed to connect to database: {e}")
                raise
        return self._connection
    
    def close(self):
        if self._connection is not None:
            try:
                self._connection.close()
            except:
                pass  # Ignore errors on close
            finally:
                self._is_closed = True
    
    def __del__(self):
        self.close()

# Global connection manager
_connection_manager = None

def get_connection_manager():
    global _connection_manager
    return _connection_manager

def initialize_connection_manager(hostname, path, token, catalog, schema):
    global _connection_manager
    _connection_manager = DbConnectionManager(hostname, path, token, catalog, schema)

def with_connection(func):
    """
    Decorator to provide a database connection to a function.
    
    Usage:
    @with_connection
    def my_function(arg1, arg2, conn=None, uc_catalog=None, uc_schema=None):
        # Use conn, uc_catalog, uc_schema here
        pass
    
    # Call it like:
    my_function(arg1, arg2, 
                db_hostname=hostname, 
                db_path=path, 
                db_token=token, 
                uc_catalog=catalog, 
                uc_schema=schema)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract connection parameters from kwargs
        db_hostname = kwargs.pop('db_hostname', None)
        db_path = kwargs.pop('db_path', None)
        db_token = kwargs.pop('db_token', None)
        uc_catalog = kwargs.pop('uc_catalog', None)
        uc_schema = kwargs.pop('uc_schema', None)
        
        # Use global connection manager if parameters are provided
        if all([db_hostname, db_path, db_token]):
            manager = DbConnectionManager(db_hostname, db_path, db_token, uc_catalog, uc_schema)
            conn = manager.get_connection()
            try:
                return func(*args, **kwargs, conn=conn, uc_catalog=uc_catalog, uc_schema=uc_schema)
            finally:
                manager.close()
        else:
            # Fallback to global connection manager if it exists
            global_manager = get_connection_manager()
            if global_manager:
                conn = global_manager.get_connection()
                return func(*args, **kwargs, conn=conn, 
                          uc_catalog=global_manager.catalog, 
                          uc_schema=global_manager.schema)
            else:
                raise ValueError("No database connection available. Provide connection parameters or initialize global connection manager.")
    
    return wrapper
