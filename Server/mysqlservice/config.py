
# Default LLM model for SQL generation
DEFAULT_MODEL = "mistral:latest"  # Change to mistral:24b or another model if available
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# MySQL Database Configuration
MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "2003Hamza2003!",
    "database": "kpis",
    "charset": "utf8mb4",
    "autocommit": True
}

# Ultra-detailed SQL generation prompt (updated for MySQL)
SQL_PROMPT_TEMPLATE = """
You are a world-class SQL expert converting user questions into correct, optimized SQL for a MySQL database.

Rules:
1. Only output the SQL query — no explanations, no markdown.
2. Always use the exact column and table names from the schema.
3. If the question involves "per", "each", or "group", ALWAYS use GROUP BY.
4. If counting, use COUNT(column) or COUNT(DISTINCT column) depending on uniqueness.
5. Avoid SELECT * unless explicitly requested.
6. Use ORDER BY for logical sorting if it improves readability.
7. Only generate SELECT queries. Never write INSERT, UPDATE, or DELETE.
8. If aggregation is used, include all non-aggregated columns in GROUP BY.
9. Keep SQL short, readable, and valid for MySQL.
10. Use MySQL-specific functions when appropriate (e.g., LIMIT instead of SQLite's LIMIT).

Schema:
{schema}

Question:
{question}

Output format:
SQL: <query here>
"""