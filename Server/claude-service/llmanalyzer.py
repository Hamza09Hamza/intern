import os
import json
import sqlite3
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from databasecontext import DatabaseContext, TableInfo, ColumnInfo
from config import (
    MODEL_CONFIGS, EMBEDDING_CONFIG, RAG_CONFIG, 
    PERFORMANCE_CONFIG, ADVANCED_SQL_PROMPT,SQL_CONFIG
)
import re


# Enhanced imports with fallback
try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMA = True
except ImportError:
    chromadb = None
    embedding_functions = None
    HAS_CHROMA = False

try:
    from langchain_ollama import OllamaLLM
    HAS_OLLAMA = True
except ImportError:
    try:
        from langchain.llms import Ollama as OllamaLLM
        HAS_OLLAMA = True
    except ImportError:
        OllamaLLM = None
        HAS_OLLAMA = False

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedLLMAnalyzer:
    """Enhanced LLM analyzer with better error handling and performance"""
    
    def __init__(self, db_path: str, model_config: Dict[str, Any] = None):
        self.db_path = db_path
        self.model_config = model_config or MODEL_CONFIGS["primary"]
        self.context: Optional[DatabaseContext] = None
        self.query_cache: Dict[str, str] = {}
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Initialize LLM
        self._init_llm()
        
        # Performance tracking
        self.performance_stats = {
            "queries_processed": 0,
            "cache_hits": 0,
            "average_response_time": 0.0
        }
    
    def _init_chromadb(self):
        """Initialize ChromaDB with proper error handling"""
        self.chroma_client = None
        self.collection = None
        
        if not HAS_CHROMA:
            logger.warning("ChromaDB not available. RAG functionality disabled.")
            return
        
        try:
            self.chroma_client = chromadb.Client()
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
    
    def _init_llm(self):
        """Initialize LLM with fallback options"""
        self.llm = None
        self.current_model = None
        
        if not HAS_OLLAMA:
            logger.error("Ollama not available. Please install langchain-ollama")
            return
        
        # Try primary model first, then fallback
        for model_type, config in [("primary", MODEL_CONFIGS["primary"]), 
                                 ("fallback", MODEL_CONFIGS["fallback"])]:
            try:
                self.llm = OllamaLLM(
                    model=config["name"],
                    temperature=config["temperature"],
                    num_ctx=config.get("context_window", 4096),
                    num_gpu=PERFORMANCE_CONFIG.get("ollama_num_gpu", 1),
                    num_thread=PERFORMANCE_CONFIG.get("ollama_num_thread", 4)
                )
                self.current_model = config
                logger.info(f"LLM initialized with {config['name']} ({model_type})")
                
                # Test the model
                test_response = self.llm.invoke("SELECT 1;")
                if test_response:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to initialize {config['name']}: {e}")
                self.llm = None
                continue
        
        if not self.llm:
            logger.error("No LLM models available. Please check Ollama installation.")
    
    def analyze_database(self, force_refresh: bool = False) -> DatabaseContext:
        """Enhanced database analysis with caching"""
        cache_file = f"{self.db_path}.context_cache.json"
        
        # Try to load from cache first
        if not force_refresh and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    self.context = DatabaseContext.from_dict(cached_data)
                    logger.info("Loaded database context from cache")
                    return self.context
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Perform fresh analysis
        logger.info("Performing fresh database analysis...")
        start_time = datetime.now()
        
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA foreign_keys = ON")
            cur = conn.cursor()
            
            # Initialize context
            self.context = DatabaseContext(schema_text="")
            
            # Get all tables
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cur.fetchall()]
            
            schema_lines = ["DATABASE SCHEMA ANALYSIS", "=" * 50, ""]
            
            for table_name in table_names:
                table_info = self._analyze_table(cur, table_name)
                self.context.tables[table_name] = table_info
                
                # Build relationships
                for fk in table_info.foreign_keys:
                    ref_table = fk.get("ref_table")
                    if ref_table:
                        if table_name not in self.context.relationships:
                            self.context.relationships[table_name] = []
                        if ref_table not in self.context.relationships[table_name]:
                            self.context.relationships[table_name].append(ref_table)
                        
                        # Add reverse relationship
                        if ref_table not in self.context.relationships:
                            self.context.relationships[ref_table] = []
                        if table_name not in self.context.relationships[ref_table]:
                            self.context.relationships[ref_table].append(table_name)
                
                # Add to schema text
                schema_lines.extend(self._generate_table_description(table_info))
                schema_lines.append("")
            
            self.context.schema_text = "\n".join(schema_lines)
            self.context.last_analyzed = datetime.now()
            self.context.statistics = self.context.get_table_statistics()
            
            conn.close()
            
            # Initialize RAG collection
            if self.chroma_client:
                self._create_rag_collection()
            
            # Cache the results
            try:
                with open(cache_file, 'w') as f:
                    json.dump(self.context.to_dict(), f, indent=2, default=str)
                logger.info("Database context cached successfully")
            except Exception as e:
                logger.warning(f"Failed to cache context: {e}")
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Database analysis completed in {analysis_time:.2f} seconds")
            logger.info(f"Found {len(self.context.tables)} tables with {sum(len(t.columns) for t in self.context.tables.values())} total columns")
            
            return self.context
            
        except Exception as e:
            logger.error(f"Database analysis failed: {e}")
            raise
    
    def _analyze_table(self, cursor: sqlite3.Cursor, table_name: str) -> TableInfo:
        """Analyze a single table in detail"""
        table_info = TableInfo(name=table_name)
        
        # Get column information
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        column_data = cursor.fetchall()
        
        for col_data in column_data:
            col_name = col_data[1]
            col_info = ColumnInfo(
                name=col_name,
                type=col_data[2],
                nullable=not bool(col_data[3]),  # NOT NULL flag
                primary_key=bool(col_data[5]),   # PK flag
                default_value=col_data[4]
            )
            table_info.columns[col_name] = col_info
        
        # Get foreign key information
        cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
        fk_data = cursor.fetchall()
        
        for fk in fk_data:
            fk_info = {
                "column": fk[3],
                "ref_table": fk[2],
                "ref_column": fk[4]
            }
            table_info.foreign_keys.append(fk_info)
            
            # Mark column as foreign key
            if fk[3] in table_info.columns:
                table_info.columns[fk[3]].foreign_key = f"{fk[2]}.{fk[4]}"
        
        # Get row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM '{table_name}'")
            table_info.row_count = cursor.fetchone()[0]
        except sqlite3.Error:
            table_info.row_count = 0
        
        # Get sample data and analyze values
        try:
            cursor.execute(f"SELECT * FROM '{table_name}' LIMIT 10")
            sample_rows = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            
            for row in sample_rows:
                sample_dict = dict(zip(column_names, row))
                table_info.sample_data.append(sample_dict)
                
                # Collect sample values for each column
                for i, value in enumerate(row):
                    if i < len(column_names):
                        col_name = column_names[i]
                        if col_name in table_info.columns:
                            if value not in table_info.columns[col_name].sample_values:
                                table_info.columns[col_name].sample_values.append(value)
                            if len(table_info.columns[col_name].sample_values) >= 5:
                                break
        except sqlite3.Error as e:
            logger.warning(f"Failed to get sample data for {table_name}: {e}")
        
        return table_info
    
    def _generate_table_description(self, table_info: TableInfo) -> List[str]:
        """Generate human-readable table description"""
        lines = [
            f"TABLE: {table_info.name}",
            f"Rows: {table_info.row_count:,}",
            "Columns:"
        ]
        
        for col_name, col_info in table_info.columns.items():
            col_desc = f"  • {col_name} ({col_info.type})"
            
            if col_info.primary_key:
                col_desc += " [PRIMARY KEY]"
            if col_info.foreign_key:
                col_desc += f" [FK → {col_info.foreign_key}]"
            if not col_info.nullable:
                col_desc += " [NOT NULL]"
            
            if col_info.sample_values:
                sample_str = ", ".join(str(v)[:20] for v in col_info.sample_values[:3])
                col_desc += f" (samples: {sample_str})"
            
            lines.append(col_desc)
        
        if table_info.foreign_keys:
            lines.append("Relationships:")
            for fk in table_info.foreign_keys:
                lines.append(f"  • {fk['column']} → {fk['ref_table']}.{fk['ref_column']}")
        
        return lines
    
    def _create_rag_collection(self):
        """Create enhanced RAG collection"""
        if not self.chroma_client or not self.context:
            return
        
        collection_name = f"{RAG_CONFIG['collection_prefix']}{hashlib.md5(self.db_path.encode()).hexdigest()[:8]}"
        
        # Delete existing collection
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass
        
        try:
            # Create embedding function
            embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_CONFIG["model_name"],
                device=EMBEDDING_CONFIG.get("device", "cpu"),
                normalize_embeddings=EMBEDDING_CONFIG.get("normalize_embeddings", True)
            )
            
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=embed_fn
            )
            
            # Add enhanced documents
            ids, documents, metadatas = [], [], []
            
            for table_name, table_info in self.context.tables.items():
                # Create comprehensive table document
                doc_parts = [
                    f"Table: {table_name}",
                    f"Description: Table with {len(table_info.columns)} columns and {table_info.row_count:,} rows"
                ]
                
                # Add column information
                doc_parts.append("Columns:")
                for col_name, col_info in table_info.columns.items():
                    col_desc = f"- {col_name}: {col_info.type}"
                    if col_info.primary_key:
                        col_desc += " (primary key)"
                    if col_info.foreign_key:
                        col_desc += f" (references {col_info.foreign_key})"
                    if col_info.sample_values:
                        col_desc += f" (examples: {', '.join(str(v) for v in col_info.sample_values[:3])})"
                    doc_parts.append(col_desc)
                
                # Add relationship information
                if table_info.foreign_keys:
                    doc_parts.append("Relationships:")
                    for fk in table_info.foreign_keys:
                        doc_parts.append(f"- Links to {fk['ref_table']} via {fk['column']}")
                
                # Add sample data context
                if table_info.sample_data:
                    doc_parts.append("Sample data patterns:")
                    for sample in table_info.sample_data[:3]:
                        sample_desc = ", ".join(f"{k}={v}" for k, v in list(sample.items())[:3])
                        doc_parts.append(f"- {sample_desc}")
                
                document = "\n".join(doc_parts)
                
                ids.append(f"table_{table_name}")
                documents.append(document)
                metadatas.append({
                    "type": "table",
                    "table_name": table_name,
                    "row_count": table_info.row_count,
                    "column_count": len(table_info.columns)
                })
            
            # Add relationship documents
            for table, related_tables in self.context.relationships.items():
                if related_tables:
                    doc = f"Table {table} is related to: {', '.join(related_tables)}. "
                    doc += f"This allows joining {table} with these tables for comprehensive queries."
                    
                    ids.append(f"relations_{table}")
                    documents.append(doc)
                    metadatas.append({
                        "type": "relationship",
                        "primary_table": table,
                        "related_tables": related_tables
                    })
            
            # Batch add to collection
            if ids:
                batch_size = PERFORMANCE_CONFIG.get("chroma_batch_size", 100)
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    batch_docs = documents[i:i + batch_size]
                    batch_metas = metadatas[i:i + batch_size]
                    
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_metas
                    )
            
            self.context.rag_collection_name = collection_name
            logger.info(f"RAG collection created with {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to create RAG collection: {e}")
            self.collection = None
    
    def retrieve_context(self, question: str, top_k: int = None) -> Tuple[List[str], Dict[str, Any]]:
        """Enhanced context retrieval with query analysis"""
        top_k = top_k or RAG_CONFIG["top_k"]
        
        if not self.collection:
            logger.warning("No RAG collection available, using full schema")
            return [self.context.schema_text], {"method": "full_schema"}
        
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[question],
                n_results=top_k,
                where=None  # Could add filters based on question analysis
            )
            
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            # Filter by similarity threshold
            filtered_docs = []
            retrieval_info = {"method": "rag", "retrieved_count": 0, "filtered_count": 0}
            
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                retrieval_info["retrieved_count"] += 1
                if distance <= RAG_CONFIG["similarity_threshold"]:
                    filtered_docs.append(doc)
                    retrieval_info["filtered_count"] += 1
            
            if not filtered_docs:
                logger.warning("No relevant documents found, falling back to full schema")
                return [self.context.schema_text], {"method": "fallback_full_schema"}
            
            return filtered_docs, retrieval_info
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return [self.context.schema_text], {"method": "error_fallback"}
    
    def generate_sql(self, question: str, max_retries: int = None) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL with enhanced context and retry logic"""
        max_retries = max_retries or SQL_CONFIG["max_retries"]
        
        # Check cache first
        if PERFORMANCE_CONFIG.get("enable_query_cache", True):
            cache_key = hashlib.md5(question.encode()).hexdigest()
            if cache_key in self.query_cache:
                self.performance_stats["cache_hits"] += 1
                return self.query_cache[cache_key], {"method": "cache"}
        
        if not self.llm:
            return "-- ERROR: No LLM available", {"error": "no_llm"}
        
        start_time = datetime.now()
        
        # Retrieve relevant context
        context_docs, retrieval_info = self.retrieve_context(question)
        
        # Analyze question for better prompting
        query_context = self._analyze_question(question)
        
        # Build enhanced prompt
        schema_context = "\n\n".join(context_docs)
        prompt = ADVANCED_SQL_PROMPT.format(
            schema_context=schema_context,
            query_context=query_context,
            question=question
        )
        
        last_sql = ""
        generation_info = {
            "attempts": 0,
            "retrieval_info": retrieval_info,
            "query_analysis": query_context,
            "model": self.current_model["name"] if self.current_model else "unknown"
        }
        
        for attempt in range(max_retries + 1):
            generation_info["attempts"] += 1
            
            try:
                # Generate SQL
                raw_response = self.llm.invoke(prompt)
                sql = self._extract_sql(raw_response)
                last_sql = sql
                
                if not sql:
                    if attempt < max_retries:
                        logger.warning(f"No SQL generated on attempt {attempt + 1}")
                        prompt += "\n\nIMPORTANT: You must generate a valid SELECT statement."
                        continue
                    else:
                        sql = "-- ERROR: Failed to generate valid SQL"
                        generation_info["error"] = "no_sql_generated"
                        break
                
                # Validate SQL
                validation_result = self._validate_sql_comprehensive(sql)
                if validation_result["valid"]:
                    # Cache successful result
                    if PERFORMANCE_CONFIG.get("enable_query_cache", True):
                        cache_key = hashlib.md5(question.encode()).hexdigest()
                        self.query_cache[cache_key] = sql
                        
                        # Limit cache size
                        if len(self.query_cache) > 1000:
                            # Remove oldest entries (simple FIFO)
                            keys_to_remove = list(self.query_cache.keys())[:100]
                            for key in keys_to_remove:
                                del self.query_cache[key]
                    
                    generation_info.update(validation_result)
                    break
                else:
                    if attempt < max_retries:
                        logger.warning(f"SQL validation failed on attempt {attempt + 1}: {validation_result['error']}")
                        prompt += f"\n\nERROR: {validation_result['error']}. Please fix the SQL query."
                        continue
                    else:
                        sql = f"-- ERROR: {validation_result['error']}\n-- Last generated SQL: {sql}"
                        generation_info.update(validation_result)
                        break
                        
            except Exception as e:
                logger.error(f"SQL generation failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    continue
                else:
                    sql = f"-- ERROR: SQL generation failed: {str(e)}"
                    generation_info["error"] = str(e)
                    break
        
        # Update performance stats
        response_time = (datetime.now() - start_time).total_seconds()
        self.performance_stats["queries_processed"] += 1
        self.performance_stats["average_response_time"] = (
            (self.performance_stats["average_response_time"] * (self.performance_stats["queries_processed"] - 1) + response_time) 
            / self.performance_stats["queries_processed"]
        )
        
        generation_info["response_time"] = response_time
        
        return sql, generation_info
    
    def _analyze_question(self, question: str) -> str:
        """Analyze the question to provide better context for SQL generation"""
        q_lower = question.lower()
        analysis_parts = []
        
        # Detect query patterns
        from config import QUERY_PATTERNS
        import re
        
        detected_patterns = []
        for pattern_type, patterns in QUERY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q_lower):
                    detected_patterns.append(pattern_type)
                    break
        
        if detected_patterns:
            analysis_parts.append(f"Query type: {', '.join(detected_patterns)}")
        
        # Detect mentioned tables/entities
        mentioned_tables = []
        if self.context:
            for table_name in self.context.tables.keys():
                # Check for table name or common variations
                table_variations = [
                    table_name.lower(),
                    table_name.lower().rstrip('s'),  # singular form
                    table_name.lower() + 's'         # plural form
                ]
                
                for variation in table_variations:
                    if variation in q_lower:
                        mentioned_tables.append(table_name)
                        break
        
        if mentioned_tables:
            analysis_parts.append(f"Likely tables: {', '.join(mentioned_tables)}")
        
        # Detect aggregation requirements
        if any(word in q_lower for word in ['count', 'sum', 'average', 'total', 'max', 'min', 'each', 'per']):
            analysis_parts.append("Requires aggregation")
        
        # Detect sorting requirements
        if any(word in q_lower for word in ['top', 'bottom', 'highest', 'lowest', 'first', 'last']):
            analysis_parts.append("Requires sorting")
        
        # Detect filtering requirements
        if any(word in q_lower for word in ['where', 'with', 'having', 'greater than', 'less than', 'equals']):
            analysis_parts.append("Requires filtering")
        
        return "; ".join(analysis_parts) if analysis_parts else "General query"
    
    def _extract_sql(self, raw_response: str) -> str:
        """Extract clean SQL from LLM response"""
        if not raw_response:
            return ""
        
        # Clean up the response
        response = raw_response.strip()
        
        # Remove markdown formatting
        response = re.sub(r'```sql\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        
        # Look for SELECT statements
        sql_pattern = r'(SELECT\b[\s\S]*?(?:;|$))'
        matches = re.findall(sql_pattern, response, re.IGNORECASE | re.MULTILINE)
        
        if matches:
            sql = matches[0].strip()
            # Remove trailing semicolon for consistency
            if sql.endswith(';'):
                sql = sql[:-1].strip()
            return sql
        
        # Fallback: try to extract anything that looks like SQL
        if 'select' in response.lower():
            lines = response.split('\n')
            sql_lines = []
            in_sql = False
            
            for line in lines:
                line = line.strip()
                if line.lower().startswith('select'):
                    in_sql = True
                
                if in_sql:
                    if line.startswith('--') and not sql_lines:
                        continue  # Skip comments before SQL
                    sql_lines.append(line)
                    if line.endswith(';'):
                        break
            
            if sql_lines:
                sql = ' '.join(sql_lines).strip()
                if sql.endswith(';'):
                    sql = sql[:-1].strip()
                return sql
        
        return ""
    
    def _validate_sql_comprehensive(self, sql: str) -> Dict[str, Any]:
        """Comprehensive SQL validation"""
        if not sql or sql.startswith('--'):
            return {"valid": False, "error": "No valid SQL provided"}
        
        # Basic structure validation
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith('SELECT'):
            return {"valid": False, "error": "SQL must start with SELECT"}
        
        if 'FROM' not in sql_upper:
            return {"valid": False, "error": "SQL must contain FROM clause"}
        
        # Security validation
        dangerous_keywords = ['DELETE', 'UPDATE', 'INSERT', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'PRAGMA']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return {"valid": False, "error": f"Dangerous keyword '{keyword}' not allowed"}
        
        # Schema validation
        if self.context:
            validation_result = self._validate_against_schema(sql)
            if not validation_result["valid"]:
                return validation_result
        
        # Syntax validation using SQLite
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            cur = conn.cursor()
            
            # Use EXPLAIN QUERY PLAN to validate without executing
            cur.execute(f"EXPLAIN QUERY PLAN {sql}")
            cur.fetchall()
            conn.close()
            
            return {"valid": True, "validation_method": "syntax_check"}
            
        except sqlite3.Error as e:
            return {"valid": False, "error": f"SQL syntax error: {str(e)}"}
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _validate_against_schema(self, sql: str) -> Dict[str, Any]:
        """Validate SQL against database schema"""
        if not self.context:
            return {"valid": True, "validation_method": "no_schema"}
        
        # Extract table references
        table_pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        table_matches = re.findall(table_pattern, sql, re.IGNORECASE)
        
        # Check if all referenced tables exist
        for table_name in table_matches:
            if table_name.lower() not in [t.lower() for t in self.context.tables.keys()]:
                return {"valid": False, "error": f"Table '{table_name}' does not exist"}
        
        # Extract column references (simplified)
        # This is a basic check - could be enhanced for more complex scenarios
        column_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\b'
        column_matches = re.findall(column_pattern, sql, re.IGNORECASE)
        
        for table_column in column_matches:
            table_name, column_name = table_column.split('.')
            table_name = table_name.lower()
            column_name = column_name.lower()
            
            # Find the actual table name (case-insensitive)
            actual_table = None
            for t in self.context.tables.keys():
                if t.lower() == table_name:
                    actual_table = t
                    break
            
            if actual_table:
                table_columns = [c.lower() for c in self.context.tables[actual_table].columns.keys()]
                if column_name not in table_columns:
                    return {"valid": False, "error": f"Column '{column_name}' does not exist in table '{actual_table}'"}
        
        return {"valid": True, "validation_method": "schema_check"}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = 0.0
        if self.performance_stats["queries_processed"] > 0:
            cache_hit_rate = self.performance_stats["cache_hits"] / self.performance_stats["queries_processed"]
        
        return {
            **self.performance_stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.query_cache),
            "current_model": self.current_model["name"] if self.current_model else None,
            "has_rag": self.collection is not None
        }