import sqlite3
from typing import Dict, Any
from datetime import datetime

class SQLExecutor:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        if not sql.strip().upper().startswith("SELECT"):
            return {'success': False, 'error': 'Only SELECT queries allowed', 'sql': sql, 'data': []}
        try:
            conn = sqlite3.connect(self.db_path, timeout=60)
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
        except sqlite3.Error as e:
            return {"success": False, "error": str(e), "sql": sql, "data": []}

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
                out += f"  {i:2d}. " + " | ".join(str(x) for x in r) + "\n"
            if result['row_count'] > 10:
                out += f"     ... and {result['row_count'] - 10} more rows\n"
        return out
