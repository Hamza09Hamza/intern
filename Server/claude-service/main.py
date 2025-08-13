#!/usr/bin/env python3
"""
Enhanced Natural Language to SQL System
Main application entry point with comprehensive CLI interface
"""

import argparse
import sys
import json
import logging
from pathlib import Path

from nlusystem import EnhancedNLUSystem
from config import MODEL_CONFIGS, DB_CONFIG, ERROR_MESSAGES

# Setup basic logging for main
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import sqlite3
    except ImportError:
        missing_deps.append("sqlite3 (built-in)")
    
    try:
        import chromadb
    except ImportError:
        missing_deps.append("chromadb")
    
    try:
        from langchain_ollama import OllamaLLM
    except ImportError:
        missing_deps.append("langchain-ollama")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  • {dep}")
        print("\nPlease install missing dependencies:")
        print("pip install chromadb sentence-transformers langchain-ollama")
        return False
    
    return True

def check_ollama_models():
    """Check if Ollama models are available"""
    try:
        import subprocess
        import json
        
        # Check if Ollama is installed
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("❌ Ollama is not installed or not running")
            print("Please install Ollama from: https://ollama.ai")
            return False
        
        # Parse available models
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        available_models = []
        for line in lines:
            if line.strip():
                model_name = line.split()[0]
                available_models.append(model_name)
        
        # Check for recommended models
        recommended_models = [config["name"] for config in MODEL_CONFIGS.values()]
        available_recommended = [m for m in recommended_models if m in available_models]
        
        if not available_recommended:
            print("⚠️  No recommended models found. Available models:")
            for model in available_models:
                print(f"  • {model}")
            print(f"\nRecommended models for M1 Pro (16GB RAM):")
            for name, config in MODEL_CONFIGS.items():
                print(f"  • {config['name']} ({config['description']})")
            print(f"\nTo install a model: ollama pull <model_name>")
            return False
        
        print(f"✅ Found {len(available_recommended)} recommended model(s):")
        for model in available_recommended:
            print(f"  • {model}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Ollama command timed out")
        return False
    except FileNotFoundError:
        print("❌ Ollama command not found")
        print("Please install Ollama from: https://ollama.ai")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def create_sample_database(db_path: str):
    """Create a sample database for testing"""
    import sqlite3
    
    print(f"Creating sample database at {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create tables
    cur.execute("""
        CREATE TABLE departments (
            dept_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT,
            budget REAL
        )
    """)
    
    cur.execute("""
        CREATE TABLE employees (
            emp_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            dept_id INTEGER,
            hire_date TEXT,
            FOREIGN KEY (dept_id) REFERENCES departments (dept_id)
        )
    """)
    
    cur.execute("""
        CREATE TABLE salaries (
            emp_id INTEGER,
            salary REAL NOT NULL,
            effective_date TEXT,
            FOREIGN KEY (emp_id) REFERENCES employees (emp_id)
        )
    """)
    
    # Insert sample data
    departments = [
        (1, 'Engineering', 'San Francisco', 2000000.00),
        (2, 'Sales', 'New York', 1500000.00),
        (3, 'Marketing', 'Los Angeles', 800000.00),
        (4, 'HR', 'Chicago', 600000.00)
    ]
    
    employees = [
        (1, 'John', 'Doe', 1, '2020-01-15'),
        (2, 'Jane', 'Smith', 1, '2019-03-20'),
        (3, 'Mike', 'Johnson', 2, '2021-06-10'),
        (4, 'Sarah', 'Wilson', 2, '2020-11-05'),
        (5, 'David', 'Brown', 3, '2022-01-30'),
        (6, 'Lisa', 'Garcia', 4, '2019-08-12')
    ]
    
    salaries = [
        (1, 120000.00, '2024-01-01'),
        (2, 135000.00, '2024-01-01'),
        (3, 95000.00, '2024-01-01'),
        (4, 88000.00, '2024-01-01'),
        (5, 75000.00, '2024-01-01'),
        (6, 82000.00, '2024-01-01')
    ]
    
    cur.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", departments)
    cur.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", employees)
    cur.executemany("INSERT INTO salaries VALUES (?, ?, ?)", salaries)
    
    conn.commit()
    conn.close()
    
    print("✅ Sample database created successfully!")

def run_interactive_mode(args):
    """Run the interactive mode"""
    # Determine database path
    db_path = args.database or DB_CONFIG["default_path"]
    
    # Create sample database if it doesn't exist and user agrees
    if not Path(db_path).exists():
        response = input(f"Database '{db_path}' not found. Create sample database? (y/N): ")
        if response.lower().startswith('y'):
            create_sample_database(db_path)
        else:
            print("❌ No database provided. Exiting.")
            return 1
    
    # Determine model configuration
    model_config = MODEL_CONFIGS.get(args.model, MODEL_CONFIGS["primary"])
    
    print("🚀 Starting Enhanced NL-SQL System...")
    print(f"📊 Database: {db_path}")
    print(f"🤖 Model: {model_config['name']}")
    
    # Initialize and run system
    system = EnhancedNLUSystem(db_path=db_path, model_config=model_config)
    
    try:
        system.interactive_mode()
        return 0
    except KeyboardInterrupt:
        print("\n👋 System interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        return 1
    finally:
        system.cleanup()

def run_single_query(args):
    """Run a single query and exit"""
    db_path = args.database or DB_CONFIG["default_path"]
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return 1
    
    model_config = MODEL_CONFIGS.get(args.model, MODEL_CONFIGS["primary"])
    
    system = EnhancedNLUSystem(db_path=db_path, model_config=model_config)
    
    try:
        if not system.initialize():
            print(f"❌ Failed to initialize: {system.initialization_error}")
            return 1
        
        result = system.query(
            args.query, 
            format_style=args.format,
            max_rows=args.limit,
            explain=args.explain
        )
        
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            system._display_result(result, show_explanation=args.explain)
        
        return 0 if result["success"] else 1
        
    except Exception as e:
        logger.error(f"Query execution error: {e}", exc_info=True)
        return 1
    finally:
        system.cleanup()

def run_database_info(args):
    """Show database information"""
    db_path = args.database or DB_CONFIG["default_path"]
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return 1
    
    system = EnhancedNLUSystem(db_path=db_path)
    
    try:
        if not system.initialize():
            print(f"❌ Failed to initialize: {system.initialization_error}")
            return 1
        
        db_info = system.get_database_info()
        
        if args.json:
            print(json.dumps(db_info, indent=2, default=str))
        else:
            print("📊 Database Information:")
            print("="*50)
            print(f"Database: {db_info['database_path']}")
            
            if 'statistics' in db_info:
                stats = db_info['statistics']
                print(f"Tables: {stats.get('total_tables', 0)}")
                print(f"Columns: {stats.get('total_columns', 0)}")
                print(f"Total Rows: {stats.get('total_rows', 0):,}")
            
            print("\nTables:")
            for table_name, table_info in db_info['tables'].items():
                print(f"  • {table_name} ({table_info['row_count']:,} rows)")
                print(f"    Columns: {', '.join(table_info['columns'])}")
                if table_info['foreign_keys']:
                    fks = [f"{fk['column']}→{fk['ref_table']}.{fk['ref_column']}" 
                          for fk in table_info['foreign_keys']]
                    print(f"    Foreign Keys: {', '.join(fks)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Database info error: {e}", exc_info=True)
        return 1
    finally:
        system.cleanup()

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Natural Language to SQL System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode with default settings
  %(prog)s -d employees.db                    # Interactive mode with custom database
  %(prog)s -q "How many employees are there?" # Single query
  %(prog)s --info -d mydb.db                  # Show database information
  %(prog)s -q "Show top 5 salaries" --json   # Query with JSON output
        """
    )
    
    parser.add_argument(
        "-d", "--database",
        help="Database file path (default: company.db)"
    )
    
    parser.add_argument(
        "-m", "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="primary",
        help="Model configuration to use"
    )
    
    parser.add_argument(
        "-q", "--query",
        help="Single query to execute (non-interactive mode)"
    )
    
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv", "list"],
        default="table",
        help="Output format for query results"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of rows to display"
    )
    
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show explanation of generated SQL"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show database information and exit"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check system dependencies and exit"
    )
    
    parser.add_argument(
        "--create-sample",
        help="Create sample database at specified path"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle special commands first
    if args.check_deps:
        print("🔍 Checking system dependencies...")
        deps_ok = check_dependencies()
        models_ok = check_ollama_models() if deps_ok else False
        
        if deps_ok and models_ok:
            print("✅ All dependencies are satisfied!")
            return 0
        else:
            print("❌ Some dependencies are missing")
            return 1
    
    if args.create_sample:
        try:
            create_sample_database(args.create_sample)
            return 0
        except Exception as e:
            print(f"❌ Failed to create sample database: {e}")
            return 1
    
    # Check basic dependencies before proceeding
    if not check_dependencies():
        return 1
    
    # Route to appropriate handler
    try:
        if args.info:
            return run_database_info(args)
        elif args.query:
            return run_single_query(args)
        else:
            return run_interactive_mode(args)
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())