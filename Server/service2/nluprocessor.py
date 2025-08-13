import re
import sqlite3
from typing import Optional, Tuple, List
from langchain_ollama import OllamaLLM
from databasecontext import DatabaseContext
from llmanalyzer import LLMAnalyzer

class ImprovedNLUProcessor:
    def __init__(self, db_context: DatabaseContext, analyzer: LLMAnalyzer, model: str = "mistral:latest"):
        self.context = db_context
        self.analyzer = analyzer
        self.model = model
        self.llm = None
        try:
            self.llm = OllamaLLM(model=self.model, temperature=0.0)
        except Exception as e:
            print(f"⚠️ Could not init OllamaLLM: {e}")

    def natural_language_to_sql(self, question: str, max_retries: int = 3) -> str:
        """Enhanced SQL generation with better reasoning"""
        
        # Get relevant schema snippets
        snippets = self.analyzer.retrieve_relevant_snippets(question, top_k=4)
        
        # Build the enhanced prompt
        prompt = self._build_enhanced_sql_prompt(snippets, question)
        
        last_sql = ""
        for attempt in range(max_retries + 1):
            print(f"🧠 LLM attempt {attempt + 1}/{max_retries + 1}")
            
            try:
                # Get LLM response
                raw_response = self.llm.invoke(prompt)
                sql_query = self._extract_sql_from_response(raw_response)
                
                if not sql_query:
                    print(f"⚠️ No SQL extracted from response: {raw_response[:200]}...")
                    continue
                
                print(f"📝 Generated SQL: {sql_query}")
                last_sql = sql_query
                
                # Validate the SQL
                is_valid, error_msg = self._validate_sql(sql_query)
                
                if is_valid:
                    print("✅ SQL validation passed!")
                    return sql_query
                else:
                    print(f"❌ SQL validation failed: {error_msg}")
                    if attempt < max_retries:
                        # Add feedback to prompt for next attempt
                        prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {error_msg}\nPlease fix this issue in your next attempt.\n"
                        continue
                
            except Exception as e:
                print(f"⚠️ Error during SQL generation: {e}")
                continue
        
        return last_sql or f"ERROR: Could not generate valid SQL after {max_retries + 1} attempts"

    def _build_enhanced_sql_prompt(self, schema_snippets: List[str], question: str) -> str:
        """Build a comprehensive prompt that helps the LLM reason better"""
        
        schema_text = "\n\n".join(schema_snippets)
        
        prompt = f"""You are an expert SQL analyst. Your job is to convert natural language questions into precise SQL queries.

DATABASE SCHEMA:
{schema_text}

CRITICAL RULES:
1. Always start with SELECT and include a proper FROM clause
2. Use explicit JOINs with ON conditions (never implicit joins)
3. When the question asks for "each" or "per" something, use GROUP BY
4. When finding "highest/lowest/best" per group, use window functions or correlated subqueries
5. Use proper aggregation functions (COUNT, SUM, AVG, MAX, MIN)
6. Always use table aliases for readability (e.g., employees AS e, departments AS d)
7. Include ORDER BY for logical result ordering
8. Only generate SELECT queries, never INSERT/UPDATE/DELETE

REASONING PROCESS:
1. First, identify what tables are needed based on the question
2. Determine what columns to select
3. Figure out how tables should be joined
4. Identify if grouping or aggregation is needed
5. Determine appropriate filtering and ordering

EXAMPLES OF GOOD SQL STRUCTURE:

For "highest paid employee in each department":
SELECT d.name AS department, e.first_name, e.last_name, s.salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id  
JOIN salaries s ON e.emp_id = s.emp_id
WHERE s.salary = (
    SELECT MAX(s2.salary) 
    FROM employees e2 
    JOIN salaries s2 ON e2.emp_id = s2.emp_id 
    WHERE e2.dept_id = e.dept_id
)
ORDER BY s.salary DESC;

For "average salary by department":
SELECT d.name AS department, AVG(s.salary) AS avg_salary
FROM departments d
LEFT JOIN employees e ON d.dept_id = e.dept_id
LEFT JOIN salaries s ON e.emp_id = s.emp_id
GROUP BY d.dept_id, d.name
ORDER BY avg_salary DESC;

Now analyze this question and generate the appropriate SQL:

QUESTION: {question}

Think step by step:
1. What information is being requested?
2. Which tables contain this information?
3. How should the tables be joined?
4. What grouping or aggregation is needed?
5. How should results be ordered?

Generate only the SQL query (no explanations):"""

        return prompt

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract clean SQL from LLM response"""
        if not response:
            return ""
        
        # Remove code blocks
        response = re.sub(r'```sql\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        
        # Find SQL query starting with SELECT
        sql_match = re.search(r'(SELECT\b.*?)(?:\n\n|$)', response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql = sql_match.group(1).strip()
            # Remove trailing semicolon if present
            sql = sql.rstrip(';').strip()
            return sql
        
        # Fallback: look for any line starting with SELECT
        lines = response.split('\n')
        for line in lines:
            if line.strip().upper().startswith('SELECT'):
                return line.strip().rstrip(';')
        
        return ""

    def _validate_sql(self, sql: str) -> Tuple[bool, str]:
        """Comprehensive SQL validation"""
        if not sql:
            return False, "Empty SQL query"
        
        sql_upper = sql.upper().strip()
        
        # Must be a SELECT query
        if not sql_upper.startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        # Must have FROM clause
        if ' FROM ' not in sql_upper:
            return False, "Query must include FROM clause"
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False, f"Query contains forbidden keyword: {keyword}"
        
        # Validate against database schema
        try:
            # Test query syntax by using EXPLAIN
            conn = sqlite3.connect(self.analyzer.db_path)
            cur = conn.cursor()
            cur.execute(f"EXPLAIN QUERY PLAN {sql}")
            conn.close()
            return True, "Valid SQL"
            
        except sqlite3.Error as e:
            return False, f"SQL syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
