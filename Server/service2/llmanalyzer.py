import os
import json
import sqlite3
from typing import Dict, List, Optional
from databasecontext import DatabaseContext

# Try imports, handle gracefully if missing
try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMA = True
except Exception:
    chromadb = None
    embedding_functions = None
    HAS_CHROMA = False

try:
    from langchain_ollama import OllamaLLM
    HAS_OLLAMA = True
except Exception:
    OllamaLLM = None
    HAS_OLLAMA = False

DEFAULT_DB_PATH = "company.db"
RAG_TOP_K = 4

def now_iso():
    from datetime import datetime
    return datetime.utcnow().isoformat()

class LLMAnalyzer:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, model: str = "mistral:latest", rag_top_k: int = RAG_TOP_K):
        self.db_path = db_path
        self.model = model
        self.context: Optional[DatabaseContext] = None
        self.rag_top_k = rag_top_k

        self.chroma_client = None
        self.collection = None
        if HAS_CHROMA:
            self.chroma_client = chromadb.Client()
        else:
            print("⚠️ chromadb not installed. RAG disabled. Install chromadb + sentence-transformers for best results.")

        self.llm = None
        if HAS_OLLAMA:
            try:
                self.llm = OllamaLLM(model=self.model, temperature=0.1)
            except Exception as e:
                print(f"⚠️ Could not init OllamaLLM: {e}")
        else:
            print("⚠️ langchain_ollama not installed. LLM calls will not work until installed.")

    def analyze_database(self) -> DatabaseContext:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        schema_lines = ["DATABASE STRUCTURE:\n"]
        tables: List[str] = []
        columns_by_table: Dict[str, List[str]] = {}

        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_rows = [r[0] for r in cur.fetchall()]

        for table in table_rows:
            tables.append(table)
            schema_lines.append(f"TABLE: {table}")
            try:
                cur.execute(f"PRAGMA table_info({table})")
                cols = cur.fetchall()
            except sqlite3.Error:
                cols = []
            col_names = [c[1] for c in cols]
            columns_by_table[table] = col_names
            schema_lines.append("COLUMNS:")
            for c in cols:
                schema_lines.append(f"  {c[1]} ({c[2]})" + (" PRIMARY KEY" if c[5] else ""))

            try:
                cur.execute(f"PRAGMA foreign_key_list({table})")
                fks = cur.fetchall()
            except sqlite3.Error:
                fks = []
            if fks:
                schema_lines.append("FOREIGN KEYS:")
                for fk in fks:
                    schema_lines.append(f"  {fk[3]} -> {fk[2]}.{fk[4]}")

            try:
                cur.execute(f"SELECT * FROM {table} LIMIT 5")
                samples = cur.fetchall()
                if samples:
                    colnames = [d[0] for d in cur.description]
                    schema_lines.append("SAMPLE DATA:")
                    for s in samples:
                        d = dict(zip(colnames, s))
                        schema_lines.append("  " + json.dumps(d, default=str))
            except sqlite3.Error:
                pass
            schema_lines.append("")

        conn.close()
        schema_text = "\n".join(schema_lines)

        self.context = DatabaseContext(schema_text=schema_text, tables=tables, columns_by_table=columns_by_table)

        # build chroma collection per-table if chroma available
        if HAS_CHROMA:
            coll_name = f"db_schema_{os.path.basename(self.db_path)}"
            # delete existing collection if present (safe re-create)
            try:
                existing = self.chroma_client.get_collection(coll_name)
                try:
                    self.chroma_client.delete_collection(coll_name)
                except Exception:
                    pass
            except Exception:
                pass

            # Use SentenceTransformer local embedder (requires sentence-transformers)
            try:
                emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            except Exception as e:
                # fallback to default if that specific class isn't available
                try:
                    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                except Exception:
                    emb_fn = None
                    print("⚠️ Could not create SentenceTransformerEmbeddingFunction. Ensure 'sentence-transformers' is installed.")

            if emb_fn is not None:
                self.collection = self.chroma_client.create_collection(name=coll_name, embedding_function=emb_fn)
                # add per-table snippets
                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()
                ids = []
                docs = []
                metas = []
                for table in tables:
                    snippet_lines = [f"TABLE: {table}"]
                    cur.execute(f"PRAGMA table_info({table})")
                    cols = cur.fetchall()
                    for c in cols:
                        snippet_lines.append(f"{c[1]} ({c[2]})" + (" PK" if c[5] else ""))
                    cur.execute(f"PRAGMA foreign_key_list({table})")
                    fks = cur.fetchall()
                    if fks:
                        snippet_lines.append("FOREIGN KEYS:")
                        for fk in fks:
                            snippet_lines.append(f"  {fk[3]} -> {fk[2]}.{fk[4]}")
                    try:
                        cur.execute(f"SELECT * FROM {table} LIMIT 5")
                        samples = cur.fetchall()
                        if samples:
                            colnames = [d[0] for d in cur.description]
                            snippet_lines.append("SAMPLES:")
                            for s in samples:
                                d = dict(zip(colnames, s))
                                snippet_lines.append("  " + json.dumps(d, default=str))
                    except sqlite3.Error:
                        pass
                    snippet = "\n".join(snippet_lines)
                    ids.append(table)
                    docs.append(snippet)
                    metas.append({"table": table})
                conn.close()
                # add to chroma
                try:
                    self.collection.add(ids=ids, documents=docs, metadatas=metas)
                    self.context.rag_collection_name = coll_name
                except Exception as e:
                    print(f"⚠️ Failed to add docs to chroma collection: {e}")
            else:
                print("⚠️ RAG disabled because embedding function couldn't be created.")
        else:
            print("⚠️ Skipping RAG collection creation (chromadb missing).")

        print("✅ Schema extraction complete")
        return self.context

    def build_sql_generation_context(self, relevant_snippets: List[str]) -> str:
        header = (
          ''' SYSTEM / INSTRUCTIONS FOR SQL GENERATION (PRODUCTION MODE)

            You are an expert, deterministic SQL generator.  Follow these rules EXACTLY.

            INPUT (always provided):
            * {SCHEMA_SNIPPETS}   <-- One or more per-table snippets including: table name, list of columns and types, foreign-key relationships, up to 5 sample rows each.
            * DB_DIALECT: SQLite 3 (use only SQL features supported by SQLite3; avoid vendor-specific extensions).
            * USER_QUESTION: {USER_QUESTION}
            * OPTIONAL CONTEXT: {EXTRA_CONTEXT}  # e.g. timezone, date reference, business rules

            RULES (MUST FOLLOW):
            1. Use ONLY the tables and columns contained in {SCHEMA_SNIPPETS}. DO NOT invent table names, column names, or relationships.
            2. Output EXACTLY ONE SQL statement and NOTHING else except:
            - You MAY include up to one single-line SQL comment at the very top to document an assumption:
                -- ASSUMPTION: <one short sentence>
                This is optional and must be <= 120 characters. No further explanation.
            - Otherwise return only the SQL statement terminated with a semicolon.
            3. SQL must be a SELECT statement (no INSERT / UPDATE / DELETE / DDL / PRAGMA / ATTACH). If the user asks to modify data, return exactly:
            -- FORBIDDEN_OPERATION
            and nothing else.
            4. Prefer explicit column names (no `SELECT *`) unless specifically requested.
            5. Use explicit JOIN ... ON clauses for joins (avoid implicit comma joins).
            6. If the question implies grouping (words: each / per / by / for each / grouped), produce a correct GROUP BY that includes all non-aggregated selected columns.
            7. When aggregating, use appropriate aggregates (COUNT, SUM, AVG, MAX, MIN). For average use `ROUND(AVG(col),2)` unless the user requests raw precision.
            8. If filtering by dates, use DATE('YYYY-MM-DD') or the column directly. Respect the provided timezone only if additional context gives it.
            9. Use aliases for readability but keep them short and unambiguous (e.g., employees AS e, departments AS d).
            10. If a requested column exists in multiple tables and the question is ambiguous, prefer the column from the table most relevant in {SCHEMA_SNIPPETS}; if still ambiguous, qualify with the table that has the foreign-key relationship implied by the snippet.
            11. Limit huge results:
                - If user asks for "top N" use `ORDER BY` + `LIMIT N`.
                - If user asks for full dumps, do not add implicit limits.
            12. Security & safety:
                - Disallow any attempt to access system tables or execute functions not supported in SQLite.
                - Do not produce SQL containing `ATTACH`, `PRAGMA`, `VACUUM`, `BEGIN`/`COMMIT`, or `;` inside strings.
            13. If you cannot produce a valid SELECT adhering to the schema, return exactly:
                -- NO_VALID_SQL
                and nothing else.
            14. Output style:
                - Uppercase SQL keywords (SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY, LIMIT).
                - End statement with a single semicolon.
            15. Execution-friendliness:
                - Avoid ambiguous equality like `dept = 'Sales'` unless `dept` is text in sample rows; prefer `d.name = 'Sales'` when `departments.name` exists.
                - Prefer joining via explicit foreign key columns shown in snippets (e.g. `e.dept_id = d.dept_id`).
            16. When the question includes numeric thresholds (e.g., "more than 500,000") treat values as numeric unless sample rows indicate otherwise.

            EXAMPLES (do NOT output these; they are examples to follow):
            Q: "For each department, count how many employees."
            SQL:
            SELECT d.name AS department_name, COUNT(DISTINCT e.emp_id) AS employee_count
            FROM departments d
            JOIN employees e ON e.dept_id = d.dept_id
            GROUP BY d.name
            ORDER BY employee_count DESC;

            Q: "List top 5 highest-paid employees with job titles."
            SQL:
            SELECT e.emp_id, e.first_name || ' ' || e.last_name AS full_name, j.title, s.salary
            FROM employees e
            JOIN jobs j ON e.job_id = j.job_id
            JOIN salaries s ON s.emp_id = e.emp_id
            ORDER BY s.salary DESC
            LIMIT 5;

            END OF PROMPT.
            '''
        )
        body = "\n\n".join(relevant_snippets)
        return header + body

    def retrieve_relevant_snippets(self, question: str, top_k: Optional[int] = None) -> List[str]:
        if not HAS_CHROMA or not self.collection:
            return [self.context.schema_text]
        top_k = top_k or self.rag_top_k
        try:
            res = self.collection.query(query_texts=[question], n_results=top_k)
            docs = res["documents"][0]
            return docs
        except Exception as e:
            print(f"⚠️ RAG query failed: {e} — falling back to full schema.")
            return [self.context.schema_text]


class ImprovedLLMAnalyzer(LLMAnalyzer):
    def build_sql_generation_context(self, relevant_snippets: List[str]) -> str:
        """Enhanced context building with better examples"""
        
        context = """
DATABASE CONTEXT FOR SQL GENERATION:

You are a highly skilled SQL expert. Your task is to generate precise, executable SQL queries based on natural language questions.

AVAILABLE SCHEMA:
"""
        
        for snippet in relevant_snippets:
            context += f"\n{snippet}\n" + "-" * 50
        
        context += """

KEY PRINCIPLES:
1. ALWAYS include proper FROM clauses
2. Use explicit JOINs with clear ON conditions
3. For "per/each" questions, use GROUP BY or window functions
4. For "highest/lowest in each group", use correlated subqueries or window functions
5. Always use table aliases for clarity
6. Include meaningful column aliases in results
7. Add appropriate ORDER BY for logical result presentation

COMMON PATTERNS:
- "highest X per Y" → Use MAX with GROUP BY or window functions
- "average X by Y" → Use AVG with GROUP BY  
- "count X per Y" → Use COUNT with GROUP BY
- "list all X" → Simple SELECT with JOINs as needed
- "who/what/which" → Focus on specific entities in SELECT

Generate clean, executable SQL only."""

        return context
