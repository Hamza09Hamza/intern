from mysqlservice.nluprocessor import ImprovedMySQLNLUProcessor
from mysqlservice.sqlexecutor import MySQLExecutor
from mysqlservice.llmanalyzer import ImprovedMySQLLLMAnalyzer
from typing import Dict, Any, Optional, List
from mysqlservice.config import MYSQL_CONFIG
from mysqlservice.formatting import format_results
DEFAULT_MODEL = "mistral:latest"

def now_iso():
    from datetime import datetime
    return datetime.utcnow().isoformat()

class NLUSystem:
    def __init__(self, mysql_config: dict = None, model: str = "mistral:latest"):
        self.mysql_config = mysql_config or MYSQL_CONFIG
        self.model = model
        self.analyzer = ImprovedMySQLLLMAnalyzer(mysql_config=self.mysql_config, model=model)
        self.processor = None
        self.executor = None
        self.initialized = False
        self.query_history = []

    async def initialize(self) -> bool:
        """Initialize the system with enhanced components"""
        print(f"🚀 Initializing Enhanced MySQL NLU System (model={self.model})...")
        print(f"🔗 Connecting to MySQL: {self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}")
        
        try:
            # Test MySQL connection first
            test_executor = MySQLExecutor(mysql_config=self.mysql_config)
            if not test_executor.test_connection():
                print("❌ MySQL connection test failed")
                return False
            print("✅ MySQL connection test passed")
            
            # Analyze database structure
            ctx = self.analyzer.analyze_database()
            print(f"📊 Found {len(ctx.tables)} tables: {', '.join(ctx.tables)}")
            
            # Initialize enhanced processor
            self.processor = ImprovedMySQLNLUProcessor(ctx, self.analyzer, model=self.model, mysql_config=self.mysql_config)
            
            # Initialize SQL executor
            self.executor = MySQLExecutor(mysql_config=self.mysql_config)
            
            self.initialized = True
            print("✅ Enhanced MySQL system ready! The LLM brain is now properly configured.")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def query(self, natural_language: str) -> str:
        """Process natural language query with French responses"""
        if not self.initialized:
            return "❌ Système non initialisé. Veuillez exécuter initialize() d'abord."
        
        print("\n" + "="*70)
        print(f"❓ Question : {natural_language}")
        print("="*70)
        
        # Generate SQL using enhanced reasoning
        sql = self.processor.natural_language_to_sql(natural_language)
        
        if sql.startswith("ERROR:"):
            print(f"❌ Erreur de génération SQL : {sql}")
            return "Désolé, je n'ai pas pu traiter votre question. Pourriez-vous la reformuler ?"
        
        print(f"⚡ SQL généré : {sql}")
        
        # Execute the SQL
        result = self.executor.execute_sql(sql)
        formatted = format_results(result)
        
        # Store in history
        self.query_history.append({
            "question": natural_language,
            "sql": sql,
            "success": result.get("success", False),
            "row_count": result.get("row_count", 0),
            "execution_time": result.get("execution_time", 0),
            "timestamp": self._now_iso()
        })
        
        print(f"\n{formatted}")
        return formatted


    def interactive_mode(self):
        """Enhanced interactive mode with better UX"""
        if not self.initialize():
            print("❌ Failed to initialize system")
            return
        
        print(f"\n🤖 Enhanced MySQL Voice Assistant Brain Ready!")
        print(f"🔗 Connected to: {self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}")
        print("💡 Try questions like:")
        print("   - Who is the highest paid employee in each department?")
        print("   - What's the average salary by department?")
        print("   - How many employees work in each department?")
        print("   - List all employees hired this year")
        print("   - Show me the top 10 employees by salary")
        print("\nType 'exit' to quit, 'history' to see past queries, 'tables' to see available tables\n")
        
        while True:
            try:
                question = input("🧠 Ask me anything> ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ["exit", "quit", "bye"]:
                    print("👋 Goodbye! The MySQL brain is shutting down...")
                    break
                    
                if question.lower() == "history":
                    self._show_history()
                    continue
                    
                if question.lower() in ["tables", "schema"]:
                    self._show_tables()
                    continue
                
                # Process the query
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! MySQL brain shutdown complete.")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                import traceback
                traceback.print_exc()

    def _show_history(self):
        """Show query history with MySQL-specific metrics"""
        if not self.query_history:
            print("📜 No query history yet")
            return
            
        print(f"\n📜 Recent Query History ({len(self.query_history)} queries):")
        print("-" * 80)
        
        total_time = sum(q.get("execution_time", 0) for q in self.query_history)
        successful_queries = sum(1 for q in self.query_history if q.get("success", False))
        
        print(f"📊 Statistics: {successful_queries}/{len(self.query_history)} successful, {total_time:.3f}s total execution time")
        print("-" * 80)
        
        for i, entry in enumerate(self.query_history[-10:], 1):  # Show last 10
            status = "✅" if entry["success"] else "❌"
            exec_time = entry.get("execution_time", 0)
            row_count = entry.get("row_count", 0)
            print(f"{i}. {status} [{entry['timestamp']}] ({exec_time:.3f}s, {row_count} rows)")
            print(f"   Q: {entry['question']}")
            print(f"   SQL: {entry['sql'][:100]}{'...' if len(entry['sql']) > 100 else ''}")
            print()

    def _show_tables(self):
        """Show available tables and their basic structure"""
        if not self.analyzer.context:
            print("❌ No database context available")
            return
            
        print(f"\n📊 Available Tables in '{self.mysql_config['database']}':")
        print("-" * 60)
        
        for table in self.analyzer.context.tables:
            columns = self.analyzer.context.columns_by_table.get(table, [])
            print(f"📋 {table}: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
        
        print(f"\n💡 Total: {len(self.analyzer.context.tables)} tables")
        print("Use any table names in your natural language questions!")

    def get_database_info(self) -> Dict[str, Any]:
        """Get detailed database information"""
        if not self.initialized:
            return {"error": "System not initialized"}
            
        return {
            "database": self.mysql_config['database'],
            "host": self.mysql_config['host'],
            "port": self.mysql_config['port'],
            "tables": self.analyzer.context.tables if self.analyzer.context else [],
            "total_tables": len(self.analyzer.context.tables) if self.analyzer.context else 0,
            "columns_by_table": self.analyzer.context.columns_by_table if self.analyzer.context else {},
            "query_history_count": len(self.query_history),
            "model": self.model
        }

    def _now_iso(self):
        from datetime import datetime
        return datetime.utcnow().isoformat()