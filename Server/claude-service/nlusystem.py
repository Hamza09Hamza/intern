import logging
import sys
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from nluprocessor import EnhancedNLUProcessor
from sqlexecutor import EnhancedSQLExecutor
from llmanalyzer import EnhancedLLMAnalyzer
from config import (
    MODEL_CONFIGS, DB_CONFIG, LOGGING_CONFIG, 
    ERROR_MESSAGES, PERFORMANCE_CONFIG
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG.get("level", "INFO")),
    format=LOGGING_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGGING_CONFIG.get("log_file", "nlsql_system.log"))
    ]
)

logger = logging.getLogger(__name__)

class EnhancedNLUSystem:
    """Enhanced Natural Language to SQL System with comprehensive features"""
    
    def __init__(self, db_path: str = None, model_config: Dict[str, Any] = None):
        self.db_path = db_path or DB_CONFIG["default_path"]
        self.model_config = model_config or MODEL_CONFIGS["primary"]
        
        # Core components
        self.analyzer: Optional[EnhancedLLMAnalyzer] = None
        self.processor: Optional[EnhancedNLUProcessor] = None
        self.executor: Optional[EnhancedSQLExecutor] = None
        
        # System state
        self.initialized = False
        self.initialization_error = None
        
        # Session management
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.query_history: List[Dict[str, Any]] = []
        self.session_stats = {
            "start_time": datetime.now(),
            "queries_processed": 0,
            "successful_queries": 0,
            "failed_queries": 0
        }
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        logger.info(f"NLU System initialized with session ID: {self.session_id}")
    
    def initialize(self, force_refresh: bool = False) -> bool:
        """Initialize the system with comprehensive error handling"""
        logger.info("🚀 Starting NLU System initialization...")
        
        try:
            # Check if database exists
            if not Path(self.db_path).exists():
                error_msg = f"Database file not found: {self.db_path}"
                logger.error(error_msg)
                self.initialization_error = error_msg
                return False
            
            # Initialize analyzer
            logger.info("Initializing LLM analyzer...")
            self.analyzer = EnhancedLLMAnalyzer(
                db_path=self.db_path,
                model_config=self.model_config
            )
            
            # Analyze database
            logger.info("Analyzing database structure...")
            context = self.analyzer.analyze_database(force_refresh=force_refresh)
            
            if not context or not context.tables:
                error_msg = "No tables found in database or analysis failed"
                logger.error(error_msg)
                self.initialization_error = error_msg
                return False
            
            # Initialize processor
            logger.info("Initializing NLU processor...")
            self.processor = EnhancedNLUProcessor(context, self.analyzer)
            
            # Initialize executor
            logger.info("Initializing SQL executor...")
            self.executor = EnhancedSQLExecutor(self.db_path)
            
            self.initialized = True
            
            # Log success
            table_count = len(context.tables)
            column_count = sum(len(table.columns) for table in context.tables.values())
            logger.info(f"✅ System ready! Found {table_count} tables with {column_count} columns")
            logger.info(f"📊 Database: {Path(self.db_path).name}")
            logger.info(f"🤖 Model: {self.model_config.get('name', 'unknown')}")
            logger.info(f"🎯 RAG: {'enabled' if context.rag_collection_name else 'disabled'}")
            
            return True
            
        except Exception as e:
            error_msg = f"System initialization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.initialization_error = error_msg
            self.initialized = False
            return False
    
    def query(self, natural_language: str, format_style: str = "table", 
              max_rows: int = 20, explain: bool = False) -> Dict[str, Any]:
        """Process a natural language query with comprehensive options"""
        if not self.initialized:
            return {
                "success": False,
                "error": self.initialization_error or "System not initialized",
                "result": None,
                "metadata": {}
            }
        
        query_start_time = datetime.now()
        
        # Basic input validation
        if not natural_language or not natural_language.strip():
            return {
                "success": False,
                "error": "Empty query provided",
                "result": None,
                "metadata": {}
            }
        
        query_id = f"{self.session_id}_{len(self.query_history) + 1:04d}"
        
        logger.info(f"🔍 Processing query {query_id}: {natural_language}")
        
        try:
            # Step 1: Process natural language to SQL
            with self.performance_monitor.measure("nl_to_sql"):
                sql, nl_metadata = self.processor.process_natural_language_query(natural_language)
            
            if sql.startswith("--"):
                # SQL generation failed
                self.session_stats["failed_queries"] += 1
                error_result = {
                    "success": False,
                    "error": sql,
                    "result": None,
                    "metadata": {
                        "query_id": query_id,
                        "generation_method": nl_metadata.get("generation_method", "unknown"),
                        "processing_time": (datetime.now() - query_start_time).total_seconds()
                    }
                }
                self._log_query(natural_language, error_result)
                return error_result
            
            # Step 2: Execute SQL
            with self.performance_monitor.measure("sql_execution"):
                execution_result = self.executor.execute_sql(sql)
            
            # Step 3: Format results
            with self.performance_monitor.measure("result_formatting"):
                formatted_output = self.executor.format_results(
                    execution_result, 
                    max_rows=max_rows, 
                    format_style=format_style
                )
            
            # Step 4: Generate explanation if requested
            explanation = None
            if explain:
                with self.performance_monitor.measure("explanation_generation"):
                    explanation = self.processor._explain_sql_intent(sql, nl_metadata)
            
            # Update session stats
            if execution_result["success"]:
                self.session_stats["successful_queries"] += 1
            else:
                self.session_stats["failed_queries"] += 1
            
            self.session_stats["queries_processed"] += 1
            
            # Create comprehensive result
            query_result = {
                "success": execution_result["success"],
                "error": execution_result.get("error"),
                "result": formatted_output,
                "metadata": {
                    "query_id": query_id,
                    "original_question": natural_language,
                    "generated_sql": sql,
                    "execution_time": execution_result.get("execution_time", 0),
                    "row_count": execution_result.get("row_count", 0),
                    "from_cache": execution_result.get("from_cache", False),
                    "generation_method": nl_metadata.get("generation_method"),
                    "query_analysis": nl_metadata.get("query_analysis", {}),
                    "explanation": explanation,
                    "processing_time": (datetime.now() - query_start_time).total_seconds(),
                    "performance_metrics": self.performance_monitor.get_last_measurements()
                }
            }
            
            # Log the query
            self._log_query(natural_language, query_result)
            
            return query_result
            
        except Exception as e:
            self.session_stats["failed_queries"] += 1
            error_msg = f"Query processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            error_result = {
                "success": False,
                "error": error_msg,
                "result": None,
                "metadata": {
                    "query_id": query_id,
                    "processing_time": (datetime.now() - query_start_time).total_seconds()
                }
            }
            
            self._log_query(natural_language, error_result)
            return error_result
    
    def _log_query(self, question: str, result: Dict[str, Any]):
        """Log query to history"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "success": result["success"],
            "sql": result["metadata"].get("generated_sql", ""),
            "row_count": result["metadata"].get("row_count", 0),
            "execution_time": result["metadata"].get("execution_time", 0),
            "processing_time": result["metadata"].get("processing_time", 0),
            "error": result.get("error"),
            "from_cache": result["metadata"].get("from_cache", False)
        }
        
        self.query_history.append(log_entry)
        
        # Limit history size
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-500:]  # Keep last 500
    
    def interactive_mode(self):
        """Enhanced interactive mode with better UX"""
        if not self.initialize():
            print(f"❌ Failed to initialize system: {self.initialization_error}")
            return
        
        print("🎉 Welcome to the Enhanced Natural Language SQL Interface!")
        print(f"📊 Database: {Path(self.db_path).name}")
        print(f"🤖 Model: {self.model_config.get('name', 'unknown')}")
        print("\n" + "="*60)
        print("💡 Tips:")
        print("  • Ask questions in natural language")
        print("  • Type 'help' for commands")
        print("  • Type 'stats' for system statistics")
        print("  • Type 'history' to see recent queries")
        print("  • Type 'examples' for sample questions")
        print("  • Type 'exit' or 'quit' to leave")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("🤖 Ask me anything> ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\n👋 Goodbye! Thanks for using the NL-SQL system!")
                    self._show_session_summary()
                    break
                
                elif user_input.lower() == "help":
                    self._show_help()
                    continue
                
                elif user_input.lower() == "stats":
                    self._show_system_stats()
                    continue
                
                elif user_input.lower() == "history":
                    self._show_query_history()
                    continue
                
                elif user_input.lower() == "examples":
                    self._show_examples()
                    continue
                
                elif user_input.lower() == "clear":
                    print("\n" * 50)  # Clear screen
                    continue
                
                elif user_input.lower().startswith("explain "):
                    # Query with explanation
                    question = user_input[8:].strip()
                    if question:
                        result = self.query(question, explain=True)
                        self._display_result(result, show_explanation=True)
                    continue
                
                # Process normal query
                print()  # Add spacing
                result = self.query(user_input)
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self._show_session_summary()
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Interactive mode error: {e}", exc_info=True)
                print(f"❌ An unexpected error occurred: {e}")
    
    def _display_result(self, result: Dict[str, Any], show_explanation: bool = False):
        """Display query result with nice formatting"""
        if result["success"]:
            print(result["result"])
            
            # Show explanation if available and requested
            if show_explanation and result["metadata"].get("explanation"):
                print(f"\n💡 Explanation: {result['metadata']['explanation']}")
            
            # Show performance info for interesting queries
            metadata = result["metadata"]
            if metadata.get("row_count", 0) > 100 or metadata.get("processing_time", 0) > 1:
                print(f"\n📈 Performance: {metadata.get('processing_time', 0):.2f}s processing, "
                     f"{metadata.get('execution_time', 0):.3f}s execution")
        else:
            print(f"❌ Error: {result['error']}")
            
            # Show helpful suggestions for common errors
            error_msg = result['error'].lower()
            if "table" in error_msg and "not" in error_msg:
                print("💡 Tip: Try asking 'what tables do you have?' to see available tables")
            elif "column" in error_msg and "not" in error_msg:
                print("💡 Tip: Try asking 'show me the structure of [table_name]' to see columns")
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 Help - Available Commands:")
        print("="*50)
        print("🔍 Query Commands:")
        print("  • Just type your question in natural language")
        print("  • 'explain <question>' - Get explanation of the generated SQL")
        print("\n📊 Information Commands:")
        print("  • 'stats' - Show system performance statistics")
        print("  • 'history' - Show recent query history")
        print("  • 'examples' - Show example questions you can ask")
        print("\n🛠️  Utility Commands:")
        print("  • 'clear' - Clear the screen")
        print("  • 'help' - Show this help message")
        print("  • 'exit' or 'quit' - Exit the system")
        print("="*50)
    
    def _show_system_stats(self):
        """Show comprehensive system statistics"""
        print("\n📊 System Statistics:")
        print("="*50)
        
        # Session stats
        runtime = datetime.now() - self.session_stats["start_time"]
        print(f"⏱️  Session Runtime: {runtime}")
        print(f"📝 Queries Processed: {self.session_stats['queries_processed']}")
        print(f"✅ Successful: {self.session_stats['successful_queries']}")
        print(f"❌ Failed: {self.session_stats['failed_queries']}")
        
        if self.session_stats['queries_processed'] > 0:
            success_rate = (self.session_stats['successful_queries'] / 
                          self.session_stats['queries_processed']) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Analyzer stats
        if self.analyzer:
            analyzer_stats = self.analyzer.get_performance_stats()
            print(f"\n🤖 LLM Performance:")
            print(f"  • Model: {analyzer_stats.get('current_model', 'unknown')}")
            print(f"  • Cache Hit Rate: {analyzer_stats.get('cache_hit_rate', 0):.1%}")
            print(f"  • Avg Response Time: {analyzer_stats.get('average_response_time', 0):.2f}s")
            print(f"  • RAG Enabled: {'✅' if analyzer_stats.get('has_rag') else '❌'}")
        
        # Executor stats
        if self.executor:
            executor_stats = self.executor.get_execution_stats()
            print(f"\n💾 Database Performance:")
            print(f"  • Cache Hit Rate: {executor_stats.get('cache_hit_rate', 0):.1%}")
            print(f"  • Avg Execution Time: {executor_stats.get('average_execution_time', 0):.3f}s")
            print(f"  • Active Connections: {executor_stats.get('active_connections', 0)}")
        
        # Performance monitor stats
        print(f"\n⚡ Performance Breakdown:")
        perf_stats = self.performance_monitor.get_stats()
        for operation, stats in perf_stats.items():
            if stats['count'] > 0:
                print(f"  • {operation.replace('_', ' ').title()}: "
                     f"{stats['avg']:.3f}s avg ({stats['count']} calls)")
        
        print("="*50)
    
    def _show_query_history(self, limit: int = 10):
        """Show recent query history"""
        print(f"\n📜 Recent Query History (last {limit}):")
        print("="*60)
        
        if not self.query_history:
            print("No queries in history yet.")
            return
        
        recent_queries = self.query_history[-limit:]
        for i, entry in enumerate(recent_queries, 1):
            status = "✅" if entry["success"] else "❌"
            timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M:%S")
            
            print(f"{i:2d}. [{timestamp}] {status} {entry['question']}")
            
            if entry["success"]:
                rows = entry.get("row_count", 0)
                exec_time = entry.get("execution_time", 0)
                cache_hit = " (cached)" if entry.get("from_cache") else ""
                print(f"    ↳ {rows} rows, {exec_time:.3f}s{cache_hit}")
            else:
                error = entry.get("error", "Unknown error")[:50]
                print(f"    ↳ Error: {error}{'...' if len(entry.get('error', '')) > 50 else ''}")
            print()
        
        print("="*60)
    
    def _show_examples(self):
        """Show example questions based on the database schema"""
        print("\n💡 Example Questions You Can Ask:")
        print("="*50)
        
        if self.processor:
            suggestions = self.processor.get_query_suggestions("")
            
            categories = {
                "count": "📊 Counting & Statistics:",
                "select": "📋 Data Retrieval:",
                "aggregation": "🧮 Calculations:",
                "join": "🔗 Relationships:"
            }
            
            for category, title in categories.items():
                category_suggestions = [s for s in suggestions if s.get("type") == category]
                if category_suggestions:
                    print(f"\n{title}")
                    for suggestion in category_suggestions[:3]:
                        print(f"  • {suggestion['query']}")
        
        print(f"\n🎯 General Examples:")
        print(f"  • 'How many records are in each table?'")
        print(f"  • 'Show me the structure of the database'")
        print(f"  • 'What are the most recent entries?'")
        print(f"  • 'Find records where [column] is greater than [value]'")
        print("="*50)
    
    def _show_session_summary(self):
        """Show session summary when exiting"""
        if self.session_stats['queries_processed'] > 0:
            runtime = datetime.now() - self.session_stats['start_time']
            success_rate = (self.session_stats['successful_queries'] / 
                          self.session_stats['queries_processed']) * 100
            
            print(f"\n📊 Session Summary:")
            print(f"  • Runtime: {runtime}")
            print(f"  • Queries: {self.session_stats['queries_processed']}")
            print(f"  • Success Rate: {success_rate:.1f}%")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information"""
        if not self.initialized or not self.analyzer or not self.analyzer.context:
            return {"error": "System not initialized"}
        
        context = self.analyzer.context
        return {
            "database_path": self.db_path,
            "tables": {
                name: {
                    "row_count": table.row_count,
                    "columns": list(table.columns.keys()),
                    "foreign_keys": table.foreign_keys,
                    "sample_data": table.sample_data[:3]  # First 3 samples
                }
                for name, table in context.tables.items()
            },
            "relationships": context.relationships,
            "statistics": context.get_table_statistics(),
            "last_analyzed": context.last_analyzed.isoformat() if context.last_analyzed else None
        }
    
    def export_session_data(self, filepath: str = None) -> str:
        """Export session data for analysis"""
        if not filepath:
            filepath = f"nlsql_session_{self.session_id}.json"
        
        export_data = {
            "session_id": self.session_id,
            "session_stats": self.session_stats,
            "query_history": self.query_history,
            "database_info": self.get_database_info(),
            "performance_stats": self.performance_monitor.get_stats(),
            "exported_at": datetime.now().isoformat()
        }
        
        # Convert datetime objects to strings
        export_data["session_stats"]["start_time"] = export_data["session_stats"]["start_time"].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filepath
    
    def cleanup(self):
        """Cleanup system resources"""
        if self.executor:
            self.executor.close()
        
        logger.info("System cleanup completed")


class PerformanceMonitor:
    """Simple performance monitoring utility"""
    
    def __init__(self):
        self.measurements = {}
        self.current_operations = {}
    
    def measure(self, operation_name: str):
        """Context manager for measuring operation time"""
        return self._OperationTimer(self, operation_name)
    
    def _record_measurement(self, operation_name: str, duration: float):
        """Record a measurement"""
        if operation_name not in self.measurements:
            self.measurements[operation_name] = {
                "total_time": 0.0,
                "count": 0,
                "avg": 0.0,
                "min": float('inf'),
                "max": 0.0
            }
        
        stats = self.measurements[operation_name]
        stats["total_time"] += duration
        stats["count"] += 1
        stats["avg"] = stats["total_time"] / stats["count"]
        stats["min"] = min(stats["min"], duration)
        stats["max"] = max(stats["max"], duration)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.measurements.copy()
    
    def get_last_measurements(self) -> Dict[str, float]:
        """Get the most recent measurement for each operation"""
        return self.current_operations.copy()
    
    class _OperationTimer:
        def __init__(self, monitor, operation_name):
            self.monitor = monitor
            self.operation_name = operation_name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = datetime.now()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration = (datetime.now() - self.start_time).total_seconds()
                self.monitor._record_measurement(self.operation_name, duration)
                self.monitor.current_operations[self.operation_name] = duration