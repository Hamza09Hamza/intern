
MODEL_CONFIGS = {
    "primary": {
        "name": "llama3.2",  # Excellent for SQL generation, fits well in 16GB
        "temperature": 0.0,
        "context_window": 8192,
        "description": "Primary model for SQL generation"
    },
    "fallback": {
        "name": "llama3.2:3b",  # Faster fallback
        "temperature": 0.1,
        "context_window": 4096,
        "description": "Lightweight fallback model"
    },
    "reasoning": {
        "name": "qwen2.5:7b",  # Good for complex reasoning
        "temperature": 0.0,
        "context_window": 8192,
        "description": "For complex query analysis"
    }
}

# Embedding configuration
EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "device": "mps",  # Use Metal Performance Shaders on M1
    "normalize_embeddings": True
}

# Database configuration
DB_CONFIG = {
    "default_path": "company.db",
    "timeout": 30,
    "max_connections": 5,
    "enable_foreign_keys": True
}

# RAG configuration
RAG_CONFIG = {
    "top_k": 5,
    "similarity_threshold": 0.3,
    "max_chunk_size": 1000,
    "overlap_size": 100,
    "collection_prefix": "nlsql_"
}

# SQL generation configuration
SQL_CONFIG = {
    "max_retries": 3,
    "validation_timeout": 10,
    "max_result_rows": 1000,
    "enable_explain_plan": True,
    "cache_queries": True
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "nlsql_system.log"
}

# Performance tuning for M1 Pro
PERFORMANCE_CONFIG = {
    "ollama_num_gpu": 1,
    "ollama_num_thread": 8,  # Good for M1 Pro
    "chroma_batch_size": 100,
    "enable_query_cache": True,
    "cache_ttl": 3600  # 1 hour
}

# Advanced SQL prompt template
ADVANCED_SQL_PROMPT = """You are an expert SQL analyst. Convert natural language questions to precise SQLite queries.

CRITICAL RULES:
1. Output ONLY the SQL query - no explanations, markdown, or extra text
2. Use exact table/column names from the schema
3. Always use proper JOINs (never comma-separated tables)
4. For aggregations, include ALL non-aggregated columns in GROUP BY
5. Use meaningful aliases and proper formatting
6. Handle NULL values appropriately
7. Optimize for performance with proper indexing hints

SCHEMA CONTEXT:
{schema_context}

QUERY CONTEXT:
{query_context}

USER QUESTION: {question}

Generate a single, executable SQL query:"""

# Query categorization patterns
QUERY_PATTERNS = {
    "aggregation": [
        r"(count|sum|average|avg|total|maximum|max|minimum|min)",
        r"(how many|number of|total of)",
        r"(per|each|by|grouped?)"
    ],
    "filtering": [
        r"(where|with|having|that have)",
        r"(greater than|less than|equals?|contains?)",
        r"(between|in|like)"
    ],
    "sorting": [
        r"(top|bottom|highest|lowest|best|worst)",
        r"(order|sort|rank)",
        r"(first|last|latest|earliest)"
    ],
    "joining": [
        r"(and|with|including|along with)",
        r"(department|employee|manager|team)",
        r"(related|associated|linked)"
    ]
}

# Error messages
ERROR_MESSAGES = {
    "model_not_available": "Selected model is not available. Please check Ollama installation.",
    "database_connection": "Cannot connect to database. Please check the database path.",
    "invalid_sql": "Generated SQL is invalid or unsafe.",
    "execution_timeout": "Query execution timed out.",
    "no_results": "No results found for your query.",
    "general_error": "An unexpected error occurred. Please try again."
}