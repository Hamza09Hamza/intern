import re
import sqlite3
import logging
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
from databasecontext import DatabaseContext
from llmanalyzer import EnhancedLLMAnalyzer
from config import SQL_CONFIG, ERROR_MESSAGES

logger = logging.getLogger(__name__)

class EnhancedNLUProcessor:
    """Enhanced Natural Language Understanding Processor with better SQL generation"""
    
    def __init__(self, db_context: DatabaseContext, analyzer: EnhancedLLMAnalyzer):
        self.context = db_context
        self.analyzer = analyzer
        self.query_templates = self._load_query_templates()
        self.entity_cache = {}
        
    def _load_query_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined query templates for common patterns"""
        return {
            "count_all": {
                "patterns": [r"how many (\w+)", r"count (\w+)", r"total (\w+)"],
                "template": "SELECT COUNT(*) as total_{entity} FROM {table}",
                "description": "Count all records in a table"
            },
            "average_by_group": {
                "patterns": [r"average (\w+) (?:by|per|for each) (\w+)", r"avg (\w+) (?:by|per) (\w+)"],
                "template": "SELECT {group_col}, ROUND(AVG({value_col}), 2) as avg_{value_col} FROM {table} GROUP BY {group_col} ORDER BY avg_{value_col} DESC",
                "description": "Calculate average value grouped by another column"
            },
            "top_n": {
                "patterns": [r"top (\d+) (\w+)", r"(\d+) highest (\w+)", r"(\d+) best (\w+)"],
                "template": "SELECT * FROM {table} ORDER BY {order_col} DESC LIMIT {limit}",
                "description": "Get top N records ordered by a column"
            },
            "filter_and_select": {
                "patterns": [r"(\w+) with (\w+) (?:greater than|>) (\d+)", r"(\w+) where (\w+) (?:greater than|>) (\d+)"],
                "template": "SELECT * FROM {table} WHERE {filter_col} > {value}",
                "description": "Filter records based on numeric condition"
            }
        }
    
    def process_natural_language_query(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """Main entry point for processing natural language queries"""
        logger.info(f"Processing query: {question}")
        
        # Step 1: Try template matching for common patterns
        template_result = self._try_template_matching(question)
        if template_result["matched"]:
            return template_result["sql"], template_result["metadata"]
        
        # Step 2: Enhanced entity recognition and query analysis
        query_analysis = self._analyze_query_intent(question)
        
        # Step 3: Generate SQL using LLM with enhanced context
        sql, generation_info = self.analyzer.generate_sql(question)
        
        # Step 4: Post-process and validate the generated SQL
        processed_sql = self._post_process_sql(sql, query_analysis)
        
        # Combine metadata
        metadata = {
            "generation_method": "llm_enhanced",
            "query_analysis": query_analysis,
            "generation_info": generation_info,
            "post_processed": processed_sql != sql
        }
        
        return processed_sql, metadata
    
    def _try_template_matching(self, question: str) -> Dict[str, Any]:
        """Attempt to match question against predefined templates"""
        q_lower = question.lower().strip()
        
        for template_name, template_config in self.query_templates.items():
            for pattern in template_config["patterns"]:
                match = re.search(pattern, q_lower)
                if match:
                    try:
                        sql = self._build_from_template(template_name, template_config, match, question)
                        if sql:
                            return {
                                "matched": True,
                                "sql": sql,
                                "metadata": {
                                    "generation_method": "template",
                                    "template_name": template_name,
                                    "pattern": pattern,
                                    "description": template_config["description"]
                                }
                            }
                    except Exception as e:
                        logger.warning(f"Template matching failed for {template_name}: {e}")
                        continue
        
        return {"matched": False}
    
    def _build_from_template(self, template_name: str, template_config: Dict[str, Any], 
                           match: re.Match, original_question: str) -> Optional[str]:
        """Build SQL from a matched template"""
        template = template_config["template"]
        groups = match.groups()
        
        if template_name == "count_all":
            entity = groups[0]
            table = self._find_table_for_entity(entity)
            if table:
                return template.format(entity=entity, table=table)
        
        elif template_name == "average_by_group":
            value_col, group_entity = groups[0], groups[1]
            table = self._find_table_for_entity(group_entity)
            if table:
                # Map entities to actual column names
                value_column = self._find_column_for_entity(table, value_col)
                group_column = self._find_column_for_entity(table, group_entity)
                
                if value_column and group_column:
                    return template.format(
                        group_col=group_column,
                        value_col=value_column,
                        table=table
                    )
        
        elif template_name == "top_n":
            # Handle both "top N entity" and "N highest entity" patterns
            if len(groups) == 2:
                if groups[0].isdigit():
                    limit, entity = groups[0], groups[1]
                else:
                    entity, limit = groups[0], groups[1] if groups[1].isdigit() else "10"
            else:
                limit, entity = "10", groups[0]
            
            table = self._find_table_for_entity(entity)
            if table:
                order_col = self._infer_order_column(table, entity, original_question)
                if order_col:
                    return template.format(table=table, order_col=order_col, limit=limit)
        
        elif template_name == "filter_and_select":
            entity, filter_col, value = groups
            table = self._find_table_for_entity(entity)
            if table:
                actual_filter_col = self._find_column_for_entity(table, filter_col)
                if actual_filter_col:
                    return template.format(table=table, filter_col=actual_filter_col, value=value)
        
        return None
    
    def _find_table_for_entity(self, entity: str) -> Optional[str]:
        """Find the most likely table for a given entity"""
        if entity in self.entity_cache:
            return self.entity_cache[entity]
        
        entity_lower = entity.lower()
        best_match = None
        best_score = 0
        
        for table_name in self.context.tables.keys():
            table_lower = table_name.lower()
            
            # Exact match
            if entity_lower == table_lower:
                best_match = table_name
                break
            
            # Singular/plural matching
            if (entity_lower == table_lower.rstrip('s') or 
                entity_lower + 's' == table_lower or
                entity_lower.rstrip('s') == table_lower):
                score = 0.9
                if score > best_score:
                    best_match = table_name
                    best_score = score
            
            # Substring matching
            if entity_lower in table_lower or table_lower in entity_lower:
                score = 0.7
                if score > best_score:
                    best_match = table_name
                    best_score = score
        
        if best_match:
            self.entity_cache[entity] = best_match
        
        return best_match
    
    def _find_column_for_entity(self, table_name: str, entity: str) -> Optional[str]:
        """Find the most likely column in a table for a given entity"""
        if table_name not in self.context.tables:
            return None
        
        entity_lower = entity.lower()
        table_info = self.context.tables[table_name]
        
        # Check for exact matches first
        for col_name in table_info.columns.keys():
            if entity_lower == col_name.lower():
                return col_name
        
        # Check for partial matches
        best_match = None
        best_score = 0
        
        for col_name in table_info.columns.keys():
            col_lower = col_name.lower()
            
            # Substring matching
            if entity_lower in col_lower:
                score = len(entity_lower) / len(col_lower)
                if score > best_score:
                    best_match = col_name
                    best_score = score
            
            elif col_lower in entity_lower:
                score = len(col_lower) / len(entity_lower)
                if score > best_score:
                    best_match = col_name
                    best_score = score
        
        # For common entities, try known patterns
        entity_mappings = {
            "salary": ["salary", "wage", "pay", "income", "compensation"],
            "name": ["name", "title", "label", "description"],
            "date": ["date", "time", "created", "updated", "hired"],
            "id": ["id", "key", "number", "code"]
        }
        
        if not best_match:
            for mapped_entity, synonyms in entity_mappings.items():
                if entity_lower in synonyms:
                    for col_name in table_info.columns.keys():
                        col_lower = col_name.lower()
                        if any(syn in col_lower for syn in synonyms):
                            return col_name
        
        return best_match
    
    def _infer_order_column(self, table_name: str, entity: str, question: str) -> Optional[str]:
        """Infer the column to order by based on context"""
        if table_name not in self.context.tables:
            return None
        
        table_info = self.context.tables[table_name]
        question_lower = question.lower()
        
        # Look for explicit ordering hints in the question
        if any(word in question_lower for word in ["salary", "pay", "wage", "income"]):
            for col_name in table_info.columns.keys():
                if any(term in col_name.lower() for term in ["salary", "pay", "wage", "income"]):
                    return col_name
        
        if any(word in question_lower for word in ["date", "time", "recent", "latest"]):
            for col_name in table_info.columns.keys():
                if any(term in col_name.lower() for term in ["date", "time", "created", "updated"]):
                    return col_name
        
        # Default to numeric columns or primary key
        numeric_columns = []
        for col_name, col_info in table_info.columns.items():
            if any(t in col_info.type.upper() for t in ["INT", "REAL", "NUMERIC", "DECIMAL"]):
                numeric_columns.append(col_name)
        
        if numeric_columns:
            return numeric_columns[0]
        
        # Fall back to primary key
        for col_name, col_info in table_info.columns.items():
            if col_info.primary_key:
                return col_name
        
        return None
    
    def _analyze_query_intent(self, question: str) -> Dict[str, Any]:
        """Analyze the intent and structure of the natural language query"""
        q_lower = question.lower()
        
        analysis = {
            "question_type": [],
            "entities": [],
            "operations": [],
            "filters": [],
            "aggregations": [],
            "sorting": [],
            "complexity_score": 0
        }
        
        # Detect question types
        if any(word in q_lower for word in ["what", "which", "show", "list", "get"]):
            analysis["question_type"].append("selection")
        if any(word in q_lower for word in ["how many", "count", "number"]):
            analysis["question_type"].append("counting")
        if any(word in q_lower for word in ["average", "avg", "mean"]):
            analysis["question_type"].append("averaging")
        if any(word in q_lower for word in ["sum", "total"]):
            analysis["question_type"].append("summing")
        if any(word in q_lower for word in ["max", "maximum", "highest", "most"]):
            analysis["question_type"].append("maximum")
        if any(word in q_lower for word in ["min", "minimum", "lowest", "least"]):
            analysis["question_type"].append("minimum")
        
        # Detect entities (tables/columns)
        for table_name in self.context.tables.keys():
            variations = [table_name, table_name.lower(), table_name.rstrip('s'), table_name.lower().rstrip('s')]
            for var in variations:
                if var in q_lower:
                    analysis["entities"].append({"type": "table", "value": table_name, "confidence": 0.8})
                    break
        
        # Detect operations
        if any(word in q_lower for word in ["join", "with", "and", "along with"]):
            analysis["operations"].append("join")
            analysis["complexity_score"] += 2
        
        if any(word in q_lower for word in ["group by", "per", "each", "by"]):
            analysis["operations"].append("group_by")
            analysis["complexity_score"] += 2
        
        # Detect filters
        filter_patterns = [
            (r"greater than (\d+)", "gt"),
            (r"> (\d+)", "gt"),
            (r"less than (\d+)", "lt"),
            (r"< (\d+)", "lt"),
            (r"equals? (\w+)", "eq"),
            (r"= (\w+)", "eq"),
            (r"between (\d+) and (\d+)", "between"),
            (r"in \((.*?)\)", "in")
        ]
        
        for pattern, filter_type in filter_patterns:
            matches = re.findall(pattern, q_lower)
            if matches:
                analysis["filters"].append({"type": filter_type, "matches": matches})
                analysis["complexity_score"] += 1
        
        # Detect sorting requirements
        if any(word in q_lower for word in ["top", "bottom", "highest", "lowest", "order", "sort"]):
            analysis["sorting"].append("explicit")
            analysis["complexity_score"] += 1
        
        # Calculate final complexity
        base_complexity = len(analysis["question_type"])
        analysis["complexity_score"] += base_complexity
        
        return analysis
    
    def _post_process_sql(self, sql: str, query_analysis: Dict[str, Any]) -> str:
        """Post-process generated SQL based on query analysis"""
        if not sql or sql.startswith("--"):
            return sql
        
        processed_sql = sql
        
        # Add appropriate LIMIT if not present for safety
        if "LIMIT" not in processed_sql.upper() and query_analysis.get("complexity_score", 0) < 2:
            # Only add LIMIT for simple queries to avoid breaking complex ones
            if not any(agg in processed_sql.upper() for agg in ["GROUP BY", "COUNT", "SUM", "AVG"]):
                processed_sql += " LIMIT 100"
        
        # Ensure proper formatting
        processed_sql = self._format_sql(processed_sql)
        
        return processed_sql
    
    def _format_sql(self, sql: str) -> str:
        """Format SQL for better readability"""
        if not sql:
            return sql
        
        # Basic formatting
        sql = re.sub(r'\s+', ' ', sql.strip())  # Normalize whitespace
        sql = re.sub(r'\s*,\s*', ', ', sql)     # Normalize comma spacing
        
        # Capitalize SQL keywords
        keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 
                   'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'AS', 'ON', 'AND', 'OR']
        
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            sql = re.sub(pattern, keyword, sql, flags=re.IGNORECASE)
        
        return sql
    
    def get_query_suggestions(self, partial_query: str) -> List[Dict[str, Any]]:
        """Get query suggestions based on partial input"""
        suggestions = []
        
        if not partial_query:
            return self._get_common_query_suggestions()
        
        partial_lower = partial_query.lower()
        
        # Suggest based on detected entities
        for table_name in self.context.tables.keys():
            if table_name.lower() in partial_lower:
                table_info = self.context.tables[table_name]
                
                # Suggest counting
                suggestions.append({
                    "query": f"How many {table_name.lower()} are there?",
                    "description": f"Count total records in {table_name}",
                    "type": "count"
                })
                
                # Suggest showing all
                suggestions.append({
                    "query": f"Show me all {table_name.lower()}",
                    "description": f"List all records from {table_name}",
                    "type": "select_all"
                })
                
                # Suggest aggregations on numeric columns
                for col_name, col_info in table_info.columns.items():
                    if any(t in col_info.type.upper() for t in ["INT", "REAL", "NUMERIC", "DECIMAL"]):
                        suggestions.append({
                            "query": f"What is the average {col_name} for {table_name.lower()}?",
                            "description": f"Calculate average {col_name} in {table_name}",
                            "type": "aggregation"
                        })
                        
                        if len(suggestions) >= 5:  # Limit suggestions
                            break
                
                break
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def _get_common_query_suggestions(self) -> List[Dict[str, Any]]:
        """Get common query suggestions when no input is provided"""
        suggestions = []
        
        if not self.context or not self.context.tables:
            return suggestions
        
        # Get the first few tables for examples
        table_names = list(self.context.tables.keys())[:3]
        
        for table_name in table_names:
            suggestions.extend([
                {
                    "query": f"How many {table_name.lower()} are there?",
                    "description": f"Count records in {table_name}",
                    "type": "count"
                },
                {
                    "query": f"Show me all {table_name.lower()}",
                    "description": f"List all records from {table_name}",
                    "type": "select"
                }
            ])
        
        # Add some relationship-based suggestions if we have foreign keys
        for table_name, table_info in list(self.context.tables.items())[:2]:
            if table_info.foreign_keys:
                fk = table_info.foreign_keys[0]
                suggestions.append({
                    "query": f"Show {table_name.lower()} with their {fk['ref_table']} information",
                    "description": f"Join {table_name} with {fk['ref_table']}",
                    "type": "join"
                })
        
        return suggestions[:8]
    
    def validate_and_explain_query(self, question: str) -> Dict[str, Any]:
        """Validate a natural language query and explain what SQL it would generate"""
        try:
            # Generate SQL without executing
            sql, metadata = self.process_natural_language_query(question)
            
            if sql.startswith("--"):
                return {
                    "valid": False,
                    "error": sql,
                    "explanation": "Failed to generate valid SQL"
                }
            
            # Analyze the generated SQL
            explanation = self._explain_sql_intent(sql, metadata)
            
            return {
                "valid": True,
                "sql": sql,
                "explanation": explanation,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "explanation": "An error occurred while processing the query"
            }
    
    def _explain_sql_intent(self, sql: str, metadata: Dict[str, Any]) -> str:
        """Generate a human-readable explanation of what the SQL query does"""
        if not sql:
            return "No SQL generated"
        
        sql_upper = sql.upper()
        explanation_parts = []
        
        # Identify the main operation
        if sql_upper.startswith("SELECT"):
            if "COUNT(" in sql_upper:
                explanation_parts.append("This query counts the number of records")
            elif "AVG(" in sql_upper:
                explanation_parts.append("This query calculates the average value")
            elif "SUM(" in sql_upper:
                explanation_parts.append("This query calculates the total sum")
            elif "MAX(" in sql_upper:
                explanation_parts.append("This query finds the maximum value")
            elif "MIN(" in sql_upper:
                explanation_parts.append("This query finds the minimum value")
            else:
                explanation_parts.append("This query selects and displays records")
        
        # Identify tables involved
        from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        if from_match:
            table_name = from_match.group(1)
            explanation_parts.append(f"from the {table_name} table")
        
        # Identify joins
        join_matches = re.findall(r'JOIN\s+(\w+)', sql, re.IGNORECASE)
        if join_matches:
            joined_tables = ", ".join(join_matches)
            explanation_parts.append(f"combined with data from {joined_tables}")
        
        # Identify filters
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            explanation_parts.append("with specific filtering conditions applied")
        
        # Identify grouping
        if "GROUP BY" in sql_upper:
            explanation_parts.append("grouped by specific categories")
        
        # Identify sorting
        if "ORDER BY" in sql_upper:
            if "DESC" in sql_upper:
                explanation_parts.append("sorted in descending order")
            else:
                explanation_parts.append("sorted in ascending order")
        
        # Identify limits
        limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
        if limit_match:
            limit_value = limit_match.group(1)
            explanation_parts.append(f"showing only the top {limit_value} results")
        
        explanation = " ".join(explanation_parts)
        
        # Add generation method info
        if metadata.get("generation_method") == "template":
            explanation += f" (generated using predefined template: {metadata.get('template_name', 'unknown')})"
        elif metadata.get("generation_method") == "llm_enhanced":
            explanation += " (generated using AI language model with database context)"
        
        return explanation.strip() + "."