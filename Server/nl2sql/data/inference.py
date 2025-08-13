"""
Mistral NL2SQL Model Inference and Testing
==========================================

This module handles loading trained models and running inference/testing.
It can be used for both single predictions and batch testing.

Author: Fine-tuning Guide
"""

import torch
import json
import os
from typing import List, Dict, Any
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralNL2SQLInference:
    """
    Inference class for generating SQL queries from natural language using
    a fine-tuned Mistral model.
    
    This class handles:
    1. Loading trained models
    2. Single query generation
    3. Batch testing and evaluation
    4. Performance metrics calculation
    """
    
    def __init__(self):
        """Initialize the inference engine."""
        self.tokenizer = None
        self.model = None
        
        # Configure device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple Silicon MPS (GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")
    
    def load_model(self, model_path: str):
        """
        Load a previously trained model for inference.
        
        This loads both the tokenizer and the fine-tuned model weights.
        The model should be saved using the trainer.py script.
        
        Args:
            model_path (str): Path to the trained model directory
        """
        logger.info(f"Loading trained model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load the trained model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully")
    
    def generate_sql(self, question: str, schema: Dict, business_rules: List[str] = None) -> str:
        """
        Generate SQL query for a given natural language question.
        
        This method formats the input according to the training template,
        runs inference through the model, and extracts the generated SQL.
        
        Args:
            question (str): Natural language question
            schema (dict): Database schema information
            business_rules (list): Optional business rules to consider
            
        Returns:
            str: Generated SQL query
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Format the input prompt (same format as training)
        business_rules = business_rules or []
        prompt = f"""### Task: Convert natural language to SQL

### Database Schema:
{json.dumps(schema, indent=2)}

### Business Rules:
{chr(10).join([f"- {rule}" for rule in business_rules]) if business_rules else "None"}

### Question:
{question}

### SQL Query:
"""
        
        # Tokenize the input prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate response using the model
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,         # Maximum length of generated SQL
                temperature=0.1,            # Low temperature for consistent output
                do_sample=True,             # Enable sampling for slight variation
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,     # Discourage repetitive output
            )
        
        # Decode the generated tokens back to text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the SQL portion (after our prompt)
        sql_start = generated_text.find("### SQL Query:") + len("### SQL Query:")
        generated_sql = generated_text[sql_start:].strip()
        
        # Clean up the output
        if "### End Example" in generated_sql:
            generated_sql = generated_sql.split("### End Example")[0].strip()
        
        # Remove any trailing content that might have been generated
        lines = generated_sql.split('\n')
        sql_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('###'):
                sql_lines.append(line)
            elif line.startswith('###'):
                break
        
        return ' '.join(sql_lines).strip()
    
    def test_single_query(self, question: str, schema: Dict, business_rules: List[str] = None):
        """
        Test a single query and display the results.
        
        Args:
            question (str): Natural language question
            schema (dict): Database schema
            business_rules (list): Optional business rules
        """
        print(f"\n=== Single Query Test ===")
        print(f"Question: {question}")
        print(f"Schema: {json.dumps(schema, indent=2)}")
        if business_rules:
            print(f"Business Rules: {business_rules}")
        
        try:
            sql = self.generate_sql(question, schema, business_rules)
            print(f"Generated SQL: {sql}")
        except Exception as e:
            print(f"Error generating SQL: {str(e)}")
    
    def run_batch_test(self, test_data_path: str) -> Dict[str, Any]:
        """
        Run comprehensive testing on a batch of test cases.
        
        Expected test data format (JSON):
        [
            {
                "question": "How many customers do we have?",
                "schema": {...},
                "business_rules": [...],
                "expected_sql": "SELECT COUNT(*) FROM customers;"
            }
        ]
        
        Args:
            test_data_path (str): Path to test data JSON file
            
        Returns:
            dict: Comprehensive test results and metrics
        """
        logger.info(f"Running batch test with data from: {test_data_path}")
        
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data file not found: {test_data_path}")
        
        # Load test data
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = []
        correct_predictions = 0
        total_tests = len(test_data)
        
        print(f"\n=== Running {total_tests} Test Cases ===")
        
        for i, test_case in enumerate(test_data):
            print(f"\nTest Case {i+1}/{total_tests}")
            print("-" * 50)
            
            try:
                # Generate SQL for this test case
                generated_sql = self.generate_sql(
                    question=test_case['question'],
                    schema=test_case['schema'],
                    business_rules=test_case.get('business_rules', [])
                )
                
                # Compare with expected SQL
                expected_sql = test_case.get('expected_sql', '').strip()
                
                # Simple string comparison (normalize whitespace and case)
                generated_normalized = ' '.join(generated_sql.lower().split())
                expected_normalized = ' '.join(expected_sql.lower().split())
                is_correct = generated_normalized == expected_normalized
                
                if is_correct:
                    correct_predictions += 1
                
                # Store detailed result
                result = {
                    'test_case_id': i + 1,
                    'question': test_case['question'],
                    'generated_sql': generated_sql,
                    'expected_sql': expected_sql,
                    'is_correct': is_correct,
                    'generated_normalized': generated_normalized,
                    'expected_normalized': expected_normalized
                }
                results.append(result)
                
                # Print this test case result
                print(f"Question: {test_case['question']}")
                print(f"Generated: {generated_sql}")
                print(f"Expected:  {expected_sql}")
                print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
                
            except Exception as e:
                print(f"ERROR in test case {i+1}: {str(e)}")
                result = {
                    'test_case_id': i + 1,
                    'question': test_case['question'],
                    'generated_sql': f"ERROR: {str(e)}",
                    'expected_sql': test_case.get('expected_sql', ''),
                    'is_correct': False,
                    'error': str(e)
                }
                results.append(result)
        
        # Calculate metrics
        accuracy = correct_predictions / total_tests if total_tests > 0 else 0
        
        # Create comprehensive summary
        summary = {
            'test_metadata': {
                'total_tests': total_tests,
                'correct_predictions': correct_predictions,
                'incorrect_predictions': total_tests - correct_predictions,
                'accuracy': accuracy,
                'test_file': test_data_path,
                'timestamp': str(torch.datetime.now()) if hasattr(torch, 'datetime') else "N/A"
            },
            'detailed_results': results,
            'error_analysis': self._analyze_errors(results)
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Correct: {correct_predictions}")
        print(f"Incorrect: {total_tests - correct_predictions}")
        print(f"Accuracy: {accuracy:.2%}")
        
        return summary
    
    def _analyze_errors(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze test results to identify common error patterns.
        
        Args:
            results (List[Dict]): List of test results
            
        Returns:
            Dict[str, Any]: Error analysis summary
        """
        errors = [r for r in results if not r.get('is_correct', False)]
        
        error_analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(results) if results else 0,
            'sample_errors': errors[:5] if errors else [],  # First 5 errors for review
        }
        
        return error_analysis
    
    def save_test_results(self, results: Dict[str, Any], output_file: str = "test_results.json"):
        """
        Save test results to a JSON file for further analysis.
        
        Args:
            results (Dict[str, Any]): Test results from run_batch_test()
            output_file (str): Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test results saved to {output_file}")


def main():
    """
    Main inference/testing script.
    Run this to test a trained model.
    """
    print("=== Mistral NL2SQL Inference ===")
    
    # Initialize inference engine
    inference = MistralNL2SQLInference()
    
    # Load the trained model
    model_path = "./mistral-nl2sql-trained"
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        print("Please train a model first using trainer.py")
        return
    
    try:
        print("Loading model...")
        inference.load_model(model_path)
        print("Model loaded successfully!")
        
        # Check for test data
        test_data_path = "test_data.json"
        
        if os.path.exists(test_data_path):
            print(f"\nFound test data at {test_data_path}")
            print("Running batch test...")
            
            # Run batch testing
            results = inference.run_batch_test(test_data_path)
            
            # Save results
            inference.save_test_results(results)
            print(f"\nDetailed results saved to test_results.json")
            
        else:
            print(f"\nNo test data found at {test_data_path}")
            print("Running interactive mode...")
            
            # Interactive testing mode
            while True:
                print("\n" + "="*50)
                print("Interactive SQL Generation")
                print("="*50)
                
                question = input("Enter your question (or 'quit' to exit): ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Simple schema for demonstration
                schema = {
                    "tables": [
                        {
                            "name": "customers",
                            "columns": [
                                {"name": "id", "type": "INT"},
                                {"name": "name", "type": "VARCHAR"},
                                {"name": "email", "type": "VARCHAR"}
                            ]
                        }
                    ]
                }
                
                inference.test_single_query(question, schema)
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()