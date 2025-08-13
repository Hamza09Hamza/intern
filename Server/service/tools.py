import sqlite3
from langchain.agents.tools import tool
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi3", temperature=0.1)

@tool
def run_sql(query: str) -> str:
    """Executes a SQL SELECT query on company.db and returns results."""
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."
    try:
        conn = sqlite3.connect("company.db")
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "No data found for this query."
            
        result = f"Columns: {', '.join(columns)}\nResults:\n"
        for row in rows:
            result += ", ".join(str(val) for val in row) + "\n"
        return result
    except Exception as e:
        return f"SQL Error: {e}"

@tool
def get_database_schema(dummy_input: str = "") -> str:
    """Get the complete database schema with table structures and sample data."""
    try:
        conn = sqlite3.connect("company.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_info = "DATABASE SCHEMA:\n\n"
        
        for table in tables:
            schema_info += f"Table: {table}\n"
            
            cursor.execute(f"PRAGMA table_info({table});")
            cols = cursor.fetchall()
            schema_info += "Columns:\n"
            for col in cols:
                name, typ, notnull, default, pk = col[1], col[2], col[3], col[4], col[5]
                schema_info += f"  - {name}: {typ}" + (" (PK)" if pk else "") + "\n"
            
            cursor.execute(f"SELECT * FROM {table} LIMIT 3;")
            sample = cursor.fetchall()
            if sample:
                schema_info += "Sample data:\n"
                col_names = [d[0] for d in cursor.description]
                for i, row in enumerate(sample, 1):
                    row_str = ", ".join(f"{c}={v}" for c, v in zip(col_names, row))
                    schema_info += f"  {i}. {row_str}\n"
            schema_info += "\n"
        
        conn.close()
        return schema_info
    except Exception as e:
        return f"Error getting schema: {e}"

def generate_sql_with_llm(question: str) -> str:
    """Use LLM to generate SQL query based on database schema."""
    # Fix: Call the schema tool properly with a string argument
    schema = get_database_schema.invoke({"dummy_input": ""})
    
    prompt = f"""
You are a SQL expert. Based on the database schema below, convert the user's question into a proper SQL SELECT query.

{schema}

RULES:
1. Return ONLY the SQL query, nothing else.
2. Use proper table and column names.
3. Always use SELECT statements only.

User Question: {question}

SQL Query:"""
    
    try:
        sql = llm.invoke(prompt).strip()
        # strip code fences if any
        for fence in ("```sql", "```"):
            sql = sql.replace(fence, "")
        sql = sql.splitlines()[0].strip()
        return sql
    except Exception as e:
        return f"SELECT 'Error generating SQL: {e}'"

def simple_query(question: str) -> str:
    """Generate SQL with LLM, run it, and return both."""
    sql = generate_sql_with_llm(question)
    # Fix: Call run_sql tool properly with the query parameter
    result = run_sql.invoke({"query": sql})
    return f"SQL Query: {sql}\n\nResult:\n{result}"