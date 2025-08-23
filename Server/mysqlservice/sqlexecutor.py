from typing import Dict, Any
from datetime import datetime
from mysqlservice.config import MYSQL_CONFIG

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    HAS_MYSQL = True
except Exception:
    mysql = None
    MySQLError = Exception
    HAS_MYSQL = False

class MySQLExecutor:
    def __init__(self, mysql_config: dict = None):
        self.mysql_config = MYSQL_CONFIG
        
        if not HAS_MYSQL:
            raise ImportError("mysql-connector-python is required. Install with: pip install mysql-connector-python")

    def get_connection(self):
        """Create MySQL connection"""
        try:
            return mysql.connector.connect(**self.mysql_config)
        except MySQLError as e:
            print(f"❌ MySQL connection failed: {e}")
            raise

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        if not sql.strip().upper().startswith("SELECT"):
            return {'success': False, 'error': 'Only SELECT queries allowed', 'sql': sql, 'data': []}
        
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            start = datetime.utcnow()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            elapsed = (datetime.utcnow() - start).total_seconds()
            conn.close()
            return {
                "success": True,
                "sql": sql,
                "columns": cols,
                "data": rows,
                "execution_time": elapsed,
                "row_count": len(rows)
            }
        except MySQLError as e:
            return {"success": False, "error": str(e), "sql": sql, "data": []}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}", "sql": sql, "data": []}

    def format_results(self, result: Dict[str, Any]) -> str:
        if not result['success']:
            return f"❌ SQL Error: {result['error']}\n💻 Query: {result['sql']}"
        
        out = f"💻 SQL: {result['sql']}\n"
        if result["row_count"] == 0:
            return out + "📊 No results found"
        
        out += f"⏱️ Executed in {result['execution_time']:.3f}s\n"
        out += f"📊 Found {result['row_count']} record(s)\n\n"
        
        if result.get("columns"):
            out += f"📋 Columns: {', '.join(result['columns'])}\n📄 Results:\n"
            for i, r in enumerate(result['data'][:10], 1):
                # Handle None values and convert to string
                row_data = []
                for val in r:
                    if val is None:
                        row_data.append("NULL")
                    else:
                        row_data.append(str(val))
                out += f"  {i:2d}. " + " | ".join(row_data) + "\n"
            
            if result['row_count'] > 10:
                out += f"     ... and {result['row_count'] - 10} more rows\n"
        
        return out

    def test_connection(self) -> bool:
        """Test MySQL connection"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            result = cur.fetchone()
            conn.close()
            return result[0] == 1
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False