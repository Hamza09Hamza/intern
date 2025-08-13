from nluprocessor import ImprovedNLUProcessor
from sqlexecutor import SQLExecutor
from llmanalyzer import ImprovedLLMAnalyzer
from typing import Dict, Any, Optional, List

DEFAULT_DB_PATH = "company.db"
DEFAULT_MODEL = "mistral:latest"

def now_iso():
    from datetime import datetime
    return datetime.utcnow().isoformat()
class ImprovedNLUSystem:
    def __init__(self, db_path: str = "company.db", model: str = "mistral:latest"):
        self.db_path = db_path
        self.model = model
        self.analyzer = ImprovedLLMAnalyzer(db_path=db_path, model=model)
        self.processor = None
        self.executor = None
        self.initialized = False
        self.query_history = []

    def initialize(self) -> bool:
        """Initialize the system with enhanced components"""
        print(f"🚀 Initializing Enhanced NLU System (model={self.model})...")
        
        try:
            # Analyze database structure
            ctx = self.analyzer.analyze_database()
            
            # Initialize enhanced processor
            self.processor = ImprovedNLUProcessor(ctx, self.analyzer, model=self.model)
            
            # Initialize SQL executor (assuming it exists)
            from sqlexecutor import SQLExecutor
            self.executor = SQLExecutor(db_path=self.db_path)
            
            self.initialized = True
            print("✅ Enhanced system ready! The LLM brain is now properly configured.")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    def query(self, natural_language: str) -> str:
        """Process natural language query with enhanced reasoning"""
        if not self.initialized:
            return "❌ System not initialized."
        
        print("\n" + "="*70)
        print(f"❓ Question: {natural_language}")
        print("="*70)
        
        # Generate SQL using enhanced reasoning
        sql = self.processor.natural_language_to_sql(natural_language)
        
        if sql.startswith("ERROR:"):
            print(f"❌ SQL Generation Error: {sql}")
            return sql
        
        print(f"⚡ Generated SQL: {sql}")
        
        # Execute the SQL
        result = self.executor.execute_sql(sql)
        formatted = self.executor.format_results(result)
        
        # Store in history
        self.query_history.append({
            "question": natural_language,
            "sql": sql,
            "success": result.get("success", False),
            "timestamp": self._now_iso()
        })
        
        print(f"\n{formatted}")
        return formatted

    def interactive_mode(self):
        """Enhanced interactive mode with better UX"""
        if not self.initialize():
            print("❌ Failed to initialize system")
            return
        
        print("\n🤖 Enhanced Voice Assistant Brain Ready!")
        print("💡 Try questions like:")
        print("   - Who is the highest paid employee in each department?")
        print("   - What's the average salary by department?")
        print("   - How many employees work in each department?")
        print("   - List all employees with their job titles")
        print("\nType 'exit' to quit, 'history' to see past queries\n")
        
        while True:
            try:
                question = input("🧠 Ask me anything> ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ["exit", "quit", "bye"]:
                    print("👋 Goodbye! The brain is shutting down...")
                    break
                    
                if question.lower() == "history":
                    self._show_history()
                    continue
                
                # Process the query
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Brain shutdown complete.")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")

    def _show_history(self):
        """Show query history"""
        if not self.query_history:
            print("📜 No query history yet")
            return
            
        print(f"\n📜 Recent Query History ({len(self.query_history)} queries):")
        print("-" * 60)
        
        for i, entry in enumerate(self.query_history[-10:], 1):  # Show last 10
            status = "✅" if entry["success"] else "❌"
            print(f"{i}. {status} [{entry['timestamp']}]")
            print(f"   Q: {entry['question']}")
            print(f"   SQL: {entry['sql'][:100]}{'...' if len(entry['sql']) > 100 else ''}")
            print()

    def _now_iso(self):
        from datetime import datetime
        return datetime.utcnow().isoformat()

