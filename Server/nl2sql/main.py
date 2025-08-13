"""
Mistral NL2SQL - Main Application
=================================

This is the main entry point for the Mistral NL2SQL fine-tuning project.
It provides a unified interface for training, testing, and running inference.

Author: Fine-tuning Guide
"""

import os
import sys
import argparse
import json
from typing import Dict, List

# Import our custom modules
from trainer import MistralNL2SQLTrainer
from Server.nl2sql.data.inference import MistralNL2SQLInference

def create_sample_data():
    """
    Create sample training and test data files for demonstration.
    This helps users understand the expected data format.
    """
    print("Creating sample data files...")
    
    # Sample training data
    training_data = [
        {
            "question": "How many customers do we have?",
            "schema": {
                "tables": [
                    {
                        "name": "customers",
                        "columns": [
                            {"name": "id", "type": "INT", "primary_key": True},
                            {"name": "name", "type": "VARCHAR(100)"},
                            {"name": "email", "type": "VARCHAR(100)"},
                            {"name": "created_at", "type": "TIMESTAMP"}
                        ]
                    }
                ]
            },
            "business_rules": ["Count all customers in the database"],
            "sql": "SELECT COUNT(*) FROM customers;"
        },
        {
            "question": "What are the names of customers created in 2024?",
            "schema": {
                "tables": [
                    {
                        "name": "customers",
                        "columns": [
                            {"name": "id", "type": "INT", "primary_key": True},
                            {"name": "name", "type": "VARCHAR(100)"},
                            {"name": "email", "type": "VARCHAR(100)"},
                            {"name": "created_at", "type": "TIMESTAMP"}
                        ]
                    }
                ]
            },
            "business_rules": ["Filter by creation year"],
            "sql": "SELECT name FROM customers WHERE YEAR(created_at) = 2024;"
        },
        {
            "question": "Find the top 3 customers by email domain",
            "schema": {
                "tables": [
                    {
                        "name": "customers",
                        "columns": [
                            {"name": "id", "type": "INT", "primary_key": True},
                            {"name": "name", "type": "VARCHAR(100)"},
                            {"name": "email", "type": "VARCHAR(100)"},
                            {"name": "created_at", "type": "TIMESTAMP"}
                        ]
                    }
                ]
            },
            "business_rules": ["Group by email domain and count occurrences"],
            "sql": "SELECT SUBSTRING_INDEX(email, '@', -1) as domain, COUNT(*) as count FROM customers GROUP BY domain ORDER BY count DESC LIMIT 3;"
        }
    ]
    
    # Sample test data
    test_data = [
        {
            "question": "How many customers do we have?",
            "schema": {
                "tables": [
                    {
                        "name": "customers",
                        "columns": [
                            {"name": "id", "type": "INT", "primary_key": True},
                            {"name": "name", "type": "VARCHAR(100)"},
                            {"name": "email", "type": "VARCHAR(100)"},
                            {"name": "created_at", "type": "TIMESTAMP"}
                        ]
                    }
                ]
            },
            "business_rules": ["Count all customers"],
            "expected_sql": "SELECT COUNT(*) FROM customers;"
        },
        {
            "question": "Show me all customer names",
            "schema": {
                "tables": [
                    {
                        "name": "customers",
                        "columns": [
                            {"name": "id", "type": "INT", "primary_key": True},
                            {"name": "name", "type": "VARCHAR(100)"},
                            {"name": "email", "type": "VARCHAR(100)"},
                            {"name": "created_at", "type": "TIMESTAMP"}
                        ]
                    }
                ]
            },
            "business_rules": [],
            "expected_sql": "SELECT name FROM customers;"
        }
    ]
    
    # Save training data
    with open("train_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # Save test data
    with open("test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample data files created:")
    print("  - train_data.json (3 training examples)")
    print("  - test_data.json (2 test examples)")
    print("\nYou can now run training with: python main.py train")


def train_model(args):
    """
    Train a new model using the training data.
    
    Args:
        args: Command line arguments
    """
    print("=== Starting Model Training ===")
    
    # Check if training data exists
    if not os.path.exists("train_data.json"):
        print("❌ Training data file 'train_data.json' not found!")
        print("Create sample data with: python main.py create-sample-data")
        return False
    
    try:
        # Initialize trainer
        trainer = MistralNL2SQLTrainer()
        
        # Load base model
        print("Loading base Mistral model...")
        trainer.load_base_model()
        
        # Load and process training data
        print("Loading training data...")
        train_dataset = trainer.load_training_data("train_data.json")
        
        print("Tokenizing dataset...")
        tokenized_dataset = trainer.tokenize_dataset(train_dataset)
        
        # Set output directory
        output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else "./mistral-nl2sql-trained"
        
        # Train the model
        print(f"Training model (output: {output_dir})...")
        trainer.train(tokenized_dataset, output_dir=output_dir)
        
        print("✅ Training completed successfully!")
        print(f"Trained model saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False


def test_model(args):
    """
    Test a trained model using test data.
    
    Args:
        args: Command line arguments
    """
    print("=== Starting Model Testing ===")
    
    # Set model path
    model_path = args.model_path if hasattr(args, 'model_path') and args.model_path else "./mistral-nl2sql-trained"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at: {model_path}")
        print("Train a model first with: python main.py train")
        return False
    
    # Check if test data exists
    if not os.path.exists("test_data.json"):
        print("❌ Test data file 'test_data.json' not found!")
        print("Create sample data with: python main.py create-sample-data")
        return False
    
    try:
        # Initialize inference
        inference = MistralNL2SQLInference()
        
        # Load trained model
        print(f"Loading trained model from: {model_path}")
        inference.load_model(model_path)
        
        # Run batch test
        print("Running test cases...")
        results = inference.run_batch_test("test_data.json")
        
        # Save results
        output_file = args.results_file if hasattr(args, 'results_file') and args.results_file else "test_results.json"
        inference.save_test_results(results, output_file)
        
        print(f"✅ Testing completed!")
        print(f"Results saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")
        return False


def interactive_mode(args):
    """
    Run interactive mode for single query testing.
    
    Args:
        args: Command line arguments
    """
    print("=== Interactive Mode ===")
    
    # Set model path
    model_path = args.model_path if hasattr(args, 'model_path') and args.model_path else "./mistral-nl2sql-trained"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at: {model_path}")
        print("Train a model first with: python main.py train")
        return False
    
    try:
        # Initialize inference
        inference = MistralNL2SQLInference()
        
        # Load trained model
        print(f"Loading trained model from: {model_path}")
        inference.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Interactive loop
        while True:
            print("\n" + "="*60)
            print("Interactive SQL Generation")
            print("="*60)
            print("Commands:")
            print("  - Type your question to generate SQL")
            print("  - 'quit' or 'exit' to stop")
            print("  - 'schema' to modify database schema")
            
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'schema':
                print("Using default schema (customers table)")
                continue
            elif not user_input:
                continue
            
            # Default schema for demo
            schema = {
                "tables": [
                    {
                        "name": "customers",
                        "columns": [
                            {"name": "id", "type": "INT", "primary_key": True},
                            {"name": "name", "type": "VARCHAR(100)"},
                            {"name": "email", "type": "VARCHAR(100)"},
                            {"name": "created_at", "type": "TIMESTAMP"}
                        ]
                    },
                    {
                        "name": "orders",
                        "columns": [
                            {"name": "id", "type": "INT", "primary_key": True},
                            {"name": "customer_id", "type": "INT"},
                            {"name": "total", "type": "DECIMAL(10,2)"},
                            {"name": "order_date", "type": "DATE"}
                        ]
                    }
                ]
            }
            
            try:
                sql = inference.generate_sql(user_input, schema)
                print(f"\n🔍 Generated SQL:")
                print(f"📝 {sql}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Interactive mode failed: {str(e)}")
        return False


def main():
    """
    Main application entry point with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Mistral NL2SQL Fine-tuning and Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py create-sample-data           # Create sample training/test data
  python main.py train                        # Train model with default settings
  python main.py train --output-dir ./my-model  # Train with custom output directory
  python main.py test                         # Test model with default settings
  python main.py test --model-path ./my-model   # Test with custom model path
  python main.py interactive                  # Run interactive mode
  python main.py pipeline                     # Run full pipeline (train + test)
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create sample data command
    create_parser = subparsers.add_parser(
        "create-sample-data", 
        help="Create sample training and test data files"
    )
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./mistral-nl2sql-trained",
        help="Directory to save the trained model (default: ./mistral-nl2sql-trained)"
    )
    
    # Testing command
    test_parser = subparsers.add_parser("test", help="Test a trained model")
    test_parser.add_argument(
        "--model-path", 
        type=str, 
        default="./mistral-nl2sql-trained",
        help="Path to the trained model (default: ./mistral-nl2sql-trained)"
    )
    test_parser.add_argument(
        "--results-file", 
        type=str, 
        default="test_results.json",
        help="File to save test results (default: test_results.json)"
    )
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive mode")
    interactive_parser.add_argument(
        "--model-path", 
        type=str, 
        default="./mistral-nl2sql-trained",
        help="Path to the trained model (default: ./mistral-nl2sql-trained)"
    )
    
    # Pipeline command (train + test)
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete pipeline (train + test)")
    pipeline_parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./mistral-nl2sql-trained",
        help="Directory to save the trained model"
    )
    pipeline_parser.add_argument(
        "--results-file", 
        type=str, 
        default="test_results.json",
        help="File to save test results"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    success = True
    
    if args.command == "create-sample-data":
        create_sample_data()
        
    elif args.command == "train":
        success = train_model(args)
        
    elif args.command == "test":
        success = test_model(args)
        
    elif args.command == "interactive":
        success = interactive_mode(args)
        
    elif args.command == "pipeline":
        print("=== Running Complete Pipeline ===")
        
        # Check for sample data
        if not os.path.exists("train_data.json") or not os.path.exists("test_data.json"):
            print("Creating sample data first...")
            create_sample_data()
        
        # Train model
        print("\n1. Training Phase:")
        success = train_model(args)
        
        if success:
            # Test model
            print("\n2. Testing Phase:")
            # Set model path for testing
            args.model_path = args.output_dir
            success = test_model(args)
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
        else:
            print("\n❌ Pipeline failed!")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()