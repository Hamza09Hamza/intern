import sqlite3
import logging
import threading
import time
from typing import Dict, Any,  Optional
from datetime import datetime
from contextlib import contextmanager
from config import DB_CONFIG, SQL_CONFIG, ERROR_MESSAGES

logger = logging.getLogger(__name__)

class EnhancedSQLExecutor:
    """Enhanced SQL executor with connection pooling, caching, and better error handling"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self.max_connections = DB_CONFIG.get("max_connections", 5)
        self.query_cache = {}
        self.cache_lock = threading.Lock()
        self.execution_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cache_hits": 0,
            "total_execution_time": 0.0
        }
        
        # Initialize connection pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool"""
        try:
            for _ in range(min(2, self.max_connections)):  # Start with 2 connections
                conn = self._create_connection()
                if conn:
                    self.connection_pool.append(conn)
            logger.info(f"Initialized connection pool with {len(self.connection_pool)} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a new database connection with proper configuration"""
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=DB_CONFIG.get("timeout", 30),
                check_same_thread=False
            )
            
            # Enable foreign keys if configured
            if DB_CONFIG.get("enable_foreign_keys", True):
                conn.execute("PRAGMA foreign_keys = ON")
            
            # Set additional pragmas for performance
            conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
            conn.execute("PRAGMA temp_store = memory")
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap
            
            conn.commit()
            return conn
            
        except sqlite3.Error as e:
            logger.error(f"Failed to create database connection: {e}")
            return None
    
    @contextmanager
    def _get_connection(self):
        """Get a connection from the pool or create a new one"""
        conn = None
        try:
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                else:
                    conn = self._create_connection()
            
            if not conn:
                raise sqlite3.Error("Could not obtain database connection")
            
            yield conn
            
        except sqlite3.Error:
            # Don't return a broken connection to the pool
            if conn:
                try:
                    conn.close()
                except:
                    pass
            raise
            
        finally:
            # Return connection to pool if it's still good
            if conn:
                try:
                    # Test connection
                    conn.execute("SELECT 1")
                    with self.pool_lock:
                        if len(self.connection_pool) < self.max_connections:
                            self.connection_pool.append(conn)
                        else:
                            conn.close()
                except:
                    try:
                        conn.close()
                    except:
                        pass
    
    def execute_sql(self, sql: str, params: Optional[tuple] = None, 
                   use_cache: bool = True) -> Dict[str, Any]:
        """Execute SQL with enhanced error handling and caching"""
        start_time = time.time()
        
        # Update stats
        self.execution_stats["total_queries"] += 1
        
        # Basic validation
        if not sql or not sql.strip():
            self.execution_stats["failed_queries"] += 1
            return self._create_error_result("Empty SQL query", sql)
        
        sql_clean = sql.strip()
        if not sql_clean.upper().startswith("SELECT"):
            self.execution_stats["failed_queries"] += 1
            return self._create_error_result("Only SELECT queries are allowed", sql)
        
        # Check cache first
        if use_cache and SQL_CONFIG.get("cache_queries", True):
            cache_key = self._generate_cache_key(sql, params)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.execution_stats["cache_hits"] += 1
                return cached_result
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Set query timeout if supported
                if hasattr(conn, 'set_progress_handler'):
                    timeout_seconds = SQL_CONFIG.get("validation_timeout", 30)
                    conn.set_progress_handler(self._timeout_handler, 10000)
                
                # Execute query
                if params:
                    cursor.execute(sql_clean, params)
                else:
                    cursor.execute(sql_clean)
                
                # Fetch results
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # Calculate execution time
                execution_time = time.time() - start_time
                self.execution_stats["total_execution_time"] += execution_time
                self.execution_stats["successful_queries"] += 1
                
                # Create result
                result = {
                    "success": True,
                    "sql": sql_clean,
                    "columns": columns,
                    "data": rows,
                    "execution_time": execution_time,
                    "row_count": len(rows),
                    "from_cache": False,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Cache successful results
                if use_cache and SQL_CONFIG.get("cache_queries", True) and len(rows) <= 10000:
                    cache_key = self._generate_cache_key(sql, params)
                    self._cache_result(cache_key, result)
                
                return result
                
        except sqlite3.OperationalError as e:
            self.execution_stats["failed_queries"] += 1
            error_msg = str(e)
            
            # Provide more helpful error messages
            if "no such table" in error_msg.lower():
                error_msg = f"Table not found: {self._extract_table_name_from_error(error_msg)}"
            elif "no such column" in error_msg.lower():
                error_msg = f"Column not found: {self._extract_column_name_from_error(error_msg)}"
            elif "syntax error" in error_msg.lower():
                error_msg = "SQL syntax error - please check your query structure"
            
            return self._create_error_result(error_msg, sql)
            
        except sqlite3.Error as e:
            self.execution_stats["failed_queries"] += 1
            return self._create_error_result(f"Database error: {str(e)}", sql)
            
        except Exception as e:
            self.execution_stats["failed_queries"] += 1
            logger.error(f"Unexpected error executing SQL: {e}")
            return self._create_error_result("An unexpected error occurred", sql)
    
    def _timeout_handler(self):
        """Handle query timeouts"""
        # This is a simple timeout handler - in practice, you might want more sophisticated logic
        return False  # Return False to continue, True to abort
    
    def _generate_cache_key(self, sql: str, params: Optional[tuple]) -> str:
        """Generate a cache key for the query"""
        import hashlib
        key_str = sql
        if params:
            key_str += str(params)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired"""
        with self.cache_lock:
            if cache_key in self.query_cache:
                cached_data, timestamp = self.query_cache[cache_key]
                
                # Check if cache is still valid (1 hour TTL)
                cache_ttl = SQL_CONFIG.get("cache_ttl", 3600)
                if time.time() - timestamp < cache_ttl:
                    result = cached_data.copy()
                    result["from_cache"] = True
                    return result
                else:
                    # Remove expired entry
                    del self.query_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache a query result"""
        with self.cache_lock:
            # Limit cache size
            if len(self.query_cache) >= 1000:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self.query_cache.keys())[:100]
                for key in oldest_keys:
                    del self.query_cache[key]
            
            # Store result with timestamp
            cacheable_result = result.copy()
            del cacheable_result["timestamp"]  # Don't cache timestamp
            self.query_cache[cache_key] = (cacheable_result, time.time())
    
    def _create_error_result(self, error_message: str, sql: str) -> Dict[str, Any]:
        """Create a standardized error result"""
        return {
            "success": False,
            "error": error_message,
            "sql": sql,
            "data": [],
            "columns": [],
            "execution_time": 0.0,
            "row_count": 0,
            "from_cache": False,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _extract_table_name_from_error(self, error_msg: str) -> str:
        """Extract table name from SQLite error message"""
        import re
        match = re.search(r'no such table:\s*(\w+)', error_msg)
        return match.group(1) if match else "unknown"
    
    def _extract_column_name_from_error(self, error_msg: str) -> str:
        """Extract column name from SQLite error message"""
        import re
        match = re.search(r'no such column:\s*(\w+)', error_msg)
        return match.group(1) if match else "unknown"
    
    def format_results(self, result: Dict[str, Any], max_rows: int = 20, 
                      format_style: str = "table") -> str:
        """Format query results for display with multiple formatting options"""
        if not result["success"]:
            error_icon = "❌"
            return f"{error_icon} SQL Error: {result['error']}\n💻 Query: {result['sql']}"
        
        output_lines = []
        
        # Header with query info
        cache_indicator = " (cached)" if result.get("from_cache") else ""
        output_lines.append(f"💻 SQL: {result['sql']}{cache_indicator}")
        
        if result["row_count"] == 0:
            output_lines.append("📊 No results found")
            return "\n".join(output_lines)
        
        # Execution info
        exec_time = result["execution_time"]
        output_lines.append(f"⏱️  Executed in {exec_time:.3f}s")
        output_lines.append(f"📊 Found {result['row_count']:,} record(s)")
        
        if not result["data"]:
            return "\n".join(output_lines)
        
        output_lines.append("")  # Empty line before results
        
        if format_style == "table":
            formatted_data = self._format_as_table(result, max_rows)
        elif format_style == "json":
            formatted_data = self._format_as_json(result, max_rows)
        elif format_style == "csv":
            formatted_data = self._format_as_csv(result, max_rows)
        else:
            formatted_data = self._format_as_list(result, max_rows)
        
        output_lines.append(formatted_data)
        
        # Show truncation notice if needed
        if result["row_count"] > max_rows:
            remaining = result["row_count"] - max_rows
            output_lines.append(f"\n... and {remaining:,} more row(s)")
        
        return "\n".join(output_lines)
    
    def _format_as_table(self, result: Dict[str, Any], max_rows: int) -> str:
        """Format results as a table"""
        columns = result["columns"]
        data = result["data"][:max_rows]
        
        if not columns:
            return "No columns to display"
        
        # Calculate column widths
        col_widths = []
        for i, col in enumerate(columns):
            max_width = len(col)
            for row in data:
                if i < len(row):
                    cell_width = len(str(row[i]))
                    max_width = max(max_width, cell_width)
            col_widths.append(min(max_width, 30))  # Cap at 30 chars
        
        # Build table
        lines = []
        
        # Header
        header_row = " | ".join(col.ljust(col_widths[i])[:col_widths[i]] for i, col in enumerate(columns))
        lines.append(f"📋 {header_row}")
        
        # Separator
        separator = "-+-".join("-" * width for width in col_widths)
        lines.append(f"   {separator}")
        
        # Data rows
        for row_idx, row in enumerate(data, 1):
            formatted_cells = []
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    cell_str = str(cell) if cell is not None else "NULL"
                    truncated = cell_str[:col_widths[i]]
                    formatted_cells.append(truncated.ljust(col_widths[i]))
            
            row_str = " | ".join(formatted_cells)
            lines.append(f"{row_idx:2d}. {row_str}")
        
        return "\n".join(lines)
    
    def _format_as_json(self, result: Dict[str, Any], max_rows: int) -> str:
        """Format results as JSON"""
        import json
        
        columns = result["columns"]
        data = result["data"][:max_rows]
        
        json_data = []
        for row in data:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i] if i < len(row) else None
                row_dict[col] = value
            json_data.append(row_dict)
        
        return json.dumps(json_data, indent=2, default=str)
    
    def _format_as_csv(self, result: Dict[str, Any], max_rows: int) -> str:
        """Format results as CSV"""
        import csv
        import io
        
        columns = result["columns"]
        data = result["data"][:max_rows]
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(columns)
        
        # Write data
        for row in data:
            writer.writerow(row)
        
        return output.getvalue().strip()
    
    def _format_as_list(self, result: Dict[str, Any], max_rows: int) -> str:
        """Format results as a simple list"""
        columns = result["columns"]
        data = result["data"][:max_rows]
        
        lines = []
        for row_idx, row in enumerate(data, 1):
            lines.append(f"{row_idx:2d}. " + " | ".join(str(cell) for cell in row))
        
        return "\n".join(lines)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total_queries = self.execution_stats["total_queries"]
        if total_queries > 0:
            success_rate = self.execution_stats["successful_queries"] / total_queries
            avg_execution_time = self.execution_stats["total_execution_time"] / total_queries
            cache_hit_rate = self.execution_stats["cache_hits"] / total_queries
        else:
            success_rate = 0.0
            avg_execution_time = 0.0
            cache_hit_rate = 0.0
        
        return {
            **self.execution_stats,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "cache_hit_rate": cache_hit_rate,
            "active_connections": len(self.connection_pool),
            "cache_size": len(self.query_cache)
        }
    
    def clear_cache(self):
        """Clear the query cache"""
        with self.cache_lock:
            self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def close(self):
        """Close all connections and cleanup"""
        with self.pool_lock:
            for conn in self.connection_pool:
                try:
                    conn.close()
                except:
                    pass
            self.connection_pool.clear()
        
        self.clear_cache()
        logger.info("SQL executor closed")