"""
Enhanced Mistral NL2SQL - Main Application with Multi-Database Support
=====================================================================

This is the enhanced main entry point for the Mistral NL2SQL fine-tuning project
with support for multiple domain-specific databases.

New features:
- Automatic database generation for multiple domains
- Schema extraction from database files
- Domain-specific training
- Multi-domain model training
- Enhanced inference with domain context

Author: Fine-tuning Guide
"""

import os
import sys
import argparse
import json
from typing import Dict, List

# Import our enhanced modules
from multi_trainer import EnhancedMistralNL2SQLTrainer
from inference import MistralNL2SQLInference
from db_creator import create_all_databases, generate_training_data_files

def create_databases(domains: List[str] = None):
    """Create databases for specified domains or all available domains."""
    print("=== Creating Domain-Specific Databases ===")
    
    trainer = EnhancedMistralNL2SQLTrainer()
    
    if domains:
        print(f"Creating databases for domains: {', '.join(domains)}")
        created_dbs = trainer.generate_databases(domains)
    else:
        print("Creating databases for all available domains...")
        created_dbs = create_all_databases()
        
        # Also generate training data files
        print("Generating training data files...")
        generate_training_data_files()
    
    print(f"\n✅ Created {len(created_dbs)} databases:")
    for domain, path in created_dbs.items():
        print(f"  - {domain.title()}: {path}")
    
    return created_dbs


def train_multi_domain_model(args):
    """Train a model on multiple domains."""
    print("=== Multi-Domain Model Training ===")
    
    # Initialize enhanced trainer
    trainer = EnhancedMistralNL2SQLTrainer()
    
    # Parse domains if specified
    domains = None
    if hasattr(args, 'domains') and args.domains:
        domains = args.domains.split(',')
        domains = [d.strip() for d in domains]
        print(f"Training on specified domains: {', '.join(domains)}")
    else:
        print("Training on all available domains")
    
    # Set output directory
    output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else "./mistral-nl2sql-multi-domain"
    
    try:
        # Run complete multi-domain training pipeline
        model_path = trainer.train_multi_domain(domains=domains, output_dir=output_dir)
        
        print(f"✅ Multi-domain training completed!")
        print(f"Model saved to: {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Multi-domain training failed: {str(e)}")
        return False


def train_domain_specific_model(args):
    """Train a model specific to one domain."""
    print("=== Domain-Specific Model Training ===")
    
    if not hasattr(args, 'domain') or not args.domain:
        print("❌ Domain must be specified for domain-specific training")
        return False
    
    domain = args.domain
    output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else f"./mistral-nl2sql-{domain}"
    
    # Initialize enhanced trainer
    trainer = EnhancedMistralNL2SQLTrainer()
    
    try:
        model_path = trainer.create_domain_specific_model(domain, output_dir)
        
        print(f"✅ {domain.title()} domain-specific training completed!")
        print(f"Model saved to: {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Domain-specific training failed: {str(e)}")
        return False


def test_model(args):
    """Test a trained model."""
    print("=== Model Testing ===")
    
    # Set model path
    model_path = args.model_path if hasattr(args, 'model_path') and args.model_path else "./mistral-nl2sql-multi-domain"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at: {model_path}")
        return False
    
    try:
        # Initialize inference
        inference = MistralNL2SQLInference()
        
        # Load trained model
        print(f"Loading trained model from: {model_path}")
        inference.load_model(model_path)
        
        # Check for test data
        test_data_sources = []
        
        # Look for domain-specific test data
        if os.path.exists("questions"):
            for filename in os.listdir("questions"):
                if filename.endswith("_questions.json"):
                    test_data_sources.append(os.path.join("questions", filename))
        
        # Fallback to general test data
        if os.path.exists("test_data.json"):
            test_data_sources.append("test_data.json")
        
        if not test_data_sources:
            print("❌ No test data found")
            print("Create databases first with: python main.py create-databases")
            return False
        
        # Test on available data
        all_results = []
        
        for test_file in test_data_sources:
            domain = os.path.basename(test_file).replace("_questions.json", "").replace("test_data.json", "general")
            print(f"\nTesting on {domain} data...")
            
            try:
                results = inference.run_batch_test(test_file)
                all_results.append({
                    "domain": domain,
                    "results": results
                })
                
                accuracy = results["test_metadata"]["accuracy"]
                print(f"✅ {domain.title()} accuracy: {accuracy:.2%}")
                
            except Exception as e:
                print(f"❌ Failed to test on {domain}: {e}")
        
        # Save combined results
        output_file = args.results_file if hasattr(args, 'results_file') and args.results_file else "multi_domain_test_results.json"
        
        combined_results = {
            "model_path": model_path,
            "domains_tested": [r["domain"] for r in all_results],
            "domain_results": all_results,
            "overall_summary": {
                "total_domains": len(all_results),
                "average_accuracy": sum(r["results"]["test_metadata"]["accuracy"] for r in all_results) / len(all_results) if all_results else 0
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"\n✅ Testing completed!")
        print(f"Results saved to: {output_file}")
        print(f"Overall average accuracy: {combined_results['overall_summary']['average_accuracy']:.2%}")
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")
        return False


def interactive_mode(args):
    """Run interactive mode with domain-aware inference."""
    print("=== Enhanced Interactive Mode ===")
    
    # Set model path
    model_path = args.model_path if hasattr(args, 'model_path') and args.model_path else "./mistral-nl2sql-multi-domain"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at: {model_path}")
        return False
    
    try:
        # Initialize inference
        inference = MistralNL2SQLInference()
        
        # Load trained model
        print(f"Loading trained model from: {model_path}")
        inference.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Load available schemas from databases
        available_schemas = {}
        if os.path.exists("databases"):
            for filename in os.listdir("databases"):
                if filename.endswith('.db'):
                    domain = filename.replace('.db', '')
                    db_path = os.path.join("databases", filename)
                    
                    try:
                        from trainer import DatabaseSchemaExtractor
                        extractor = DatabaseSchemaExtractor(db_path)
                        schema = extractor.extract_schema()
                        available_schemas[domain] = schema
                    except Exception as e:
                        print(f"Warning: Could not load schema for {domain}: {e}")
        
        if not available_schemas:
            print("⚠️  No database schemas found. Using default schema.")
            # Default schema fallback
            available_schemas['default'] = {
                "tables": [
                    {
                        "name": "customers",
                        "columns": [
                            {"name": "id", "type": "INT", "primary_key": True},
                            {"name": "name", "type": "VARCHAR(100)"},
                            {"name": "email", "type": "VARCHAR(100)"}
                        ]
                    }
                ]
            }
        
        # Interactive loop
        current_domain = list(available_schemas.keys())[0]
        
        while True:
            print("\n" + "="*80)
            print("Enhanced Interactive SQL Generation")
            print("="*80)
            print(f"Current domain: {current_domain.title()}")
            print(f"Available domains: {', '.join(available_schemas.keys())}")
            print("\nCommands:")
            print("  - Type your question to generate SQL")
            print("  - 'switch <domain>' to change domain")
            print("  - 'schema' to view current schema")
            print("  - 'domains' to list all domains")
            print("  - 'quit' or 'exit' to stop")
            
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
                
            elif user_input.lower() == 'domains':
                print(f"\nAvailable domains: {', '.join(available_schemas.keys())}")
                continue
                
            elif user_input.lower().startswith('switch '):
                new_domain = user_input[7:].strip()
                if new_domain in available_schemas:
                    current_domain = new_domain
                    print(f"✅ Switched to {current_domain.title()} domain")
                else:
                    print(f"❌ Domain '{new_domain}' not found")
                continue
                
            elif user_input.lower() == 'schema':
                print(f"\n{current_domain.title()} Schema:")
                print(json.dumps(available_schemas[current_domain], indent=2))
                continue
                
            elif not user_input:
                continue
            
            # Generate SQL
            try:
                schema = available_schemas[current_domain]
                sql = inference.generate_sql(user_input, schema)
                print(f"\n🎯 Generated SQL ({current_domain.title()}):")
                print(f"🔍 {sql}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Interactive mode failed: {str(e)}")
        return False


def analyze_databases():
    """Analyze existing databases and show statistics."""
    print("=== Database Analysis ===")
    
    if not os.path.exists("databases"):
        print("❌ No databases directory found")
        return False
    
    from trainer import DatabaseSchemaExtractor
    
    databases = []
    for filename in os.listdir("databases"):
        if filename.endswith('.db'):
            db_path = os.path.join("databases", filename)
            domain = filename.replace('.db', '')
            
            try:
                extractor = DatabaseSchemaExtractor(db_path)
                schema = extractor.extract_schema()
                
                # Calculate stats
                table_count = len(schema['tables'])
                total_columns = sum(len(table['columns']) for table in schema['tables'])
                total_rows = sum(table['row_count'] for table in schema['tables'])
                
                databases.append({
                    "domain": domain,
                    "path": db_path,
                    "tables": table_count,
                    "columns": total_columns,
                    "rows": total_rows
                })
                
            except Exception as e:
                print(f"❌ Error analyzing {domain}: {e}")
    
    if not databases:
        print("❌ No valid databases found")
        return False
    
    # Display analysis
    print(f"\n📊 Database Analysis Results:")
    print(f"{'Domain':<15} {'Tables':<8} {'Columns':<8} {'Rows':<8} {'Path'}")
    print("-" * 60)
    
    for db in databases:
        print(f"{db['domain']:<15} {db['tables']:<8} {db['columns']:<8} {db['rows']:<8} {db['path']}")
    
    total_tables = sum(db['tables'] for db in databases)
    total_columns = sum(db['columns'] for db in databases)
    total_rows = sum(db['rows'] for db in databases)
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {total_tables:<8} {total_columns:<8} {total_rows:<8}")
    
    print(f"\n✅ Analyzed {len(databases)} databases")
    print(f"Total training potential: ~{total_rows * 2} questions")  # Rough estimate
    
    return True


def main():
    """Enhanced main application entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Mistral NL2SQL Multi-Domain Training and Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Database Management
  python main.py create-databases                                 # Create all domain databases
  python main.py create-databases --domains "company,finance"     # Create specific domains
  python main.py analyze-databases                               # Analyze existing databases
  
  # Training
  python main.py train-multi                                     # Train on all domains
  python main.py train-multi --domains "healthcare,finance"     # Train on specific domains
  python main.py train-domain --domain healthcare               # Train domain-specific model
  
  # Testing & Inference
  python main.py test                                           # Test multi-domain model
  python main.py test --model-path ./my-model                  # Test specific model
  python main.py interactive                                   # Interactive mode
  python main.py interactive --model-path ./healthcare-model   # Domain-specific interactive
  
  # Complete Workflows
  python main.py full-pipeline                                 # Create DBs + Train + Test
  python main.py domain-pipeline --domain finance              # Complete domain workflow
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Database creation commands
    create_db_parser = subparsers.add_parser("create-databases", help="Create domain databases")
    create_db_parser.add_argument("--domains", type=str, help="Comma-separated list of domains")
    
    analyze_parser = subparsers.add_parser("analyze-databases", help="Analyze existing databases")
    
    # Training commands
    train_multi_parser = subparsers.add_parser("train-multi", help="Train multi-domain model")
    train_multi_parser.add_argument("--domains", type=str, help="Comma-separated list of domains")
    train_multi_parser.add_argument("--output-dir", type=str, default="./mistral-nl2sql-multi-domain")
    
    train_domain_parser = subparsers.add_parser("train-domain", help="Train domain-specific model")
    train_domain_parser.add_argument("--domain", type=str, required=True, help="Domain to train on")
    train_domain_parser.add_argument("--output-dir", type=str, help="Output directory")
    
    # Testing commands
    test_parser = subparsers.add_parser("test", help="Test a trained model")
    test_parser.add_argument("--model-path", type=str, help="Path to trained model")
    test_parser.add_argument("--results-file", type=str, help="Results output file")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive mode")
    interactive_parser.add_argument("--model-path", type=str, help="Path to trained model")
    
    # Pipeline commands
    full_pipeline_parser = subparsers.add_parser("full-pipeline", help="Complete pipeline")
    full_pipeline_parser.add_argument("--domains", type=str, help="Domains to include")
    
    domain_pipeline_parser = subparsers.add_parser("domain-pipeline", help="Complete domain pipeline")
    domain_pipeline_parser.add_argument("--domain", type=str, required=True, help="Domain to process")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    success = True
    
    if args.command == "create-databases":
        domains = None
        if hasattr(args, 'domains') and args.domains:
            domains = args.domains.split(',')
            domains = [d.strip() for d in domains]
        create_databases(domains)
        
    elif args.command == "analyze-databases":
        success = analyze_databases()
        
    elif args.command == "train-multi":
        success = train_multi_domain_model(args)
        
    elif args.command == "train-domain":
        success = train_domain_specific_model(args)
        
    elif args.command == "test":
        success = test_model(args)
        
    elif args.command == "interactive":
        success = interactive_mode(args)
        
    elif args.command == "full-pipeline":
        print("=== Running Complete Multi-Domain Pipeline ===")
        
        # Parse domains
        domains = None
        if hasattr(args, 'domains') and args.domains:
            domains = args.domains.split(',')
            domains = [d.strip() for d in domains]
        
        # Step 1: Create databases
        print("\n1. Creating databases...")
        create_databases(domains)
        
        # Step 2: Train model
        print("\n2. Training multi-domain model...")
        success = train_multi_domain_model(args)
        
        if success:
            # Step 3: Test model
            print("\n3. Testing model...")
            success = test_model(args)
        
        if success:
            print("\n🎉 Complete pipeline successful!")
        else:
            print("\n❌ Pipeline failed!")
            
    elif args.command == "domain-pipeline":
        print(f"=== Running Complete {args.domain.title()} Domain Pipeline ===")
        
        # Step 1: Create database for this domain
        print(f"\n1. Creating {args.domain} database...")
        create_databases([args.domain])
        
        # Step 2: Train domain-specific model
        print(f"\n2. Training {args.domain} model...")
        success = train_domain_specific_model(args)
        
        if success:
            # Step 3: Test model
            print(f"\n3. Testing {args.domain} model...")
            args.model_path = getattr(args, 'output_dir', f"./mistral-nl2sql-{args.domain}")
            success = test_model(args)
        
        if success:
            print(f"\n🎉 {args.domain.title()} domain pipeline successful!")
        else:
            print(f"\n❌ {args.domain} domain pipeline failed!")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()