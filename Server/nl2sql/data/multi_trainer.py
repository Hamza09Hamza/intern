import torch
import json
import os
import sqlite3
import logging
from typing import Dict, Any, List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from db_creator import (
    CompanyDatabase, 
    FinanceDatabase, 
    HealthcareDatabase, 
    EcommerceDatabase,
    SecurityDatabase,
    EducationDatabase,
    RealEstateDatabase,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseSchemaExtractor:
    """Extract schema information from SQLite database files."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema_info = {}
    
    def extract_schema(self) -> Dict[str, Any]:
        """Extract comprehensive schema information from database."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
            table_names = [row[0] for row in cursor.fetchall()]
            
            schema_tables = []
            
            for table_name in table_names:
                # Get column information
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = cursor.fetchall()
                
                table_columns = []
                for col in columns_info:
                    col_info = {
                        "name": col[1],
                        "type": col[2],
                        "not_null": bool(col[3]),
                        "default_value": col[4],
                        "primary_key": bool(col[5])
                    }
                    table_columns.append(col_info)
                
                # Get foreign key information
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                foreign_keys_info = cursor.fetchall()
                
                table_fks = []
                for fk in foreign_keys_info:
                    fk_info = {
                        "column": fk[3],
                        "references_table": fk[2],
                        "references_column": fk[4]
                    }
                    table_fks.append(fk_info)
                
                # Get indexes
                cursor.execute(f"PRAGMA index_list({table_name})")
                indexes_info = cursor.fetchall()
                
                table_indexes = []
                for idx in indexes_info:
                    if not idx[1].startswith('sqlite_'):  # Skip auto-generated indexes
                        cursor.execute(f"PRAGMA index_info({idx[1]})")
                        idx_columns = [col[2] for col in cursor.fetchall()]
                        table_indexes.append({
                            "name": idx[1],
                            "unique": bool(idx[2]),
                            "columns": idx_columns
                        })
                
                # Get sample data for better context
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                table_info = {
                    "name": table_name,
                    "columns": table_columns,
                    "foreign_keys": table_fks,
                    "indexes": table_indexes,
                    "row_count": row_count
                }
                schema_tables.append(table_info)
            
            self.schema_info = {
                "database_file": os.path.basename(self.db_path),
                "tables": schema_tables
            }
            
            return self.schema_info
            
        finally:
            conn.close()
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> List[Dict]:
        """Get sample data from a table for better understanding."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            rows = cursor.fetchall()
            
            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Convert to list of dictionaries
            sample_data = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                sample_data.append(row_dict)
            
            return sample_data
            
        finally:
            conn.close()


class EnhancedMistralNL2SQLTrainer:
    """Enhanced trainer that works with multiple database domains."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.database_generators = {}
        
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
        
        # Initialize database generators
        self._initialize_database_generators()
    
    def _initialize_database_generators(self):
        """Initialize all available database generators."""
        self.database_generators = {
            'company': CompanyDatabase,
            'finance': FinanceDatabase,
            'healthcare': HealthcareDatabase,
            'ecommerce': EcommerceDatabase,
            'security': SecurityDatabase,
            'education': EducationDatabase,
            'real_estate': RealEstateDatabase
        }
        logger.info(f"Initialized {len(self.database_generators)} database generators")
    
    def generate_databases(self, domains: List[str] = None) -> Dict[str, str]:
        """Generate specified databases or all available domains."""
        if domains is None:
            domains = list(self.database_generators.keys())
        
        generated_dbs = {}
        
        logger.info(f"Generating databases for domains: {domains}")
        
        for domain in domains:
            if domain not in self.database_generators:
                logger.warning(f"Unknown domain: {domain}. Skipping.")
                continue
            
            try:
                logger.info(f"Generating {domain} database...")
                generator_class = self.database_generators[domain]
                generator = generator_class()
                generator.create_database()
                
                generated_dbs[domain] = generator.db_path
                logger.info(f"✅ {domain} database created: {generator.db_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate {domain} database: {e}")
        
        return generated_dbs
    
    def load_multi_database_training_data(self, database_paths: Dict[str, str] = None) -> Dataset:
        """Load training data from multiple databases."""
        if database_paths is None:
            # Auto-discover databases
            database_paths = self.discover_databases()
        
        all_training_data = []
        
        for domain, db_path in database_paths.items():
            if not os.path.exists(db_path):
                logger.warning(f"Database not found: {db_path}. Skipping {domain}.")
                continue
            
            try:
                logger.info(f"Loading training data from {domain} database...")
                
                # Extract schema
                schema_extractor = DatabaseSchemaExtractor(db_path)
                schema = schema_extractor.extract_schema()
                
                # Load domain-specific questions
                domain_data = self._load_domain_questions(domain, db_path, schema)
                
                all_training_data.extend(domain_data)
                logger.info(f"✅ Loaded {len(domain_data)} examples from {domain}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load data from {domain}: {e}")
        
        logger.info(f"Total training examples: {len(all_training_data)}")
        
        # Format for training
        formatted_texts = []
        for example in all_training_data:
            formatted_text = self.format_training_example(example)
            formatted_texts.append({"text": formatted_text})
        
        return Dataset.from_list(formatted_texts)
    
    def discover_databases(self) -> Dict[str, str]:
        """Discover available database files."""
        database_paths = {}
        databases_dir = "databases"
        
        if not os.path.exists(databases_dir):
            logger.info("No databases directory found. Will generate databases.")
            return self.generate_databases()
        
        for filename in os.listdir(databases_dir):
            if filename.endswith('.db'):
                domain = filename.replace('.db', '')
                db_path = os.path.join(databases_dir, filename)
                database_paths[domain] = db_path
        
        logger.info(f"Discovered {len(database_paths)} databases: {list(database_paths.keys())}")
        return database_paths
    
    def _load_domain_questions(self, domain: str, db_path: str, schema: Dict) -> List[Dict]:
        """Load domain-specific questions and generate training examples."""
        # First, try to get questions from the generator
        if domain in self.database_generators:
            generator_class = self.database_generators[domain]
            generator = generator_class()
            
            # Set the database path and schema
            generator.db_path = db_path
            generator.schema_info = schema
            
            # Generate questions
            generator.generate_questions()
            
            # Get training data
            training_data = generator.get_training_data()
            
            return training_data
        
        # Fallback: try to load from JSON file
        questions_file = f"questions/{domain}_questions.json"
        if os.path.exists(questions_file):
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            training_data = []
            for qa in questions_data:
                example = {
                    "question": qa["question"],
                    "schema": schema,
                    "business_rules": qa.get("business_rules", []),
                    "sql": qa["sql"],
                    "domain": domain,
                    "database": os.path.basename(db_path)
                }
                training_data.append(example)
            
            return training_data
        
        logger.warning(f"No questions found for domain: {domain}")
        return []
    
    def load_base_model(self):
        """Load the base Mistral model with LoRA configuration."""
        logger.info(f"Loading base model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Enhanced LoRA configuration for multi-domain training
        lora_config = LoraConfig(
            r=32,                       # Higher rank for more complex multi-domain tasks
            lora_alpha=64,              # Stronger adaptation
            target_modules=[            # Target more modules for better adaptation
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj", 
                "down_proj"
            ],
            lora_dropout=0.05,          # Lower dropout for more capacity
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        
        # Apply LoRA adapters
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Base model and LoRA adapters loaded successfully")
    
    def format_training_example(self, example: Dict[str, Any]) -> str:
        """Format a training example with enhanced context."""
        question = example.get('question', '')
        schema = example.get('schema', {})
        business_rules = example.get('business_rules', [])
        sql_output = example.get('sql', '')
        domain = example.get('domain', '')
        database = example.get('database', '')
        
        # Enhanced prompt with domain context
        prompt = f"""### Task: Convert natural language to SQL

### Domain: {domain.title()}
### Database: {database}

### Database Schema:
{json.dumps(schema, indent=2)}

### Business Rules:
{chr(10).join([f"- {rule}" for rule in business_rules]) if business_rules else "None"}

### Question:
{question}

### SQL Query:
{sql_output}

### End Example
"""
        return prompt
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Convert text data to tokens for model processing."""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=2048,
                return_tensors="pt"
            )
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        logger.info("Tokenization complete")
        return tokenized_dataset
    
    def train_multi_domain(self, domains: List[str] = None, output_dir: str = "./mistral-nl2sql-multi-domain"):
        """Complete multi-domain training pipeline."""
        logger.info("=== Starting Multi-Domain Training Pipeline ===")
        
        # Step 1: Generate databases if needed
        logger.info("Step 1: Generating/discovering databases...")
        if domains:
            database_paths = self.generate_databases(domains)
        else:
            database_paths = self.discover_databases()
            if not database_paths:
                database_paths = self.generate_databases()
        
        # Step 2: Load base model
        logger.info("Step 2: Loading base model...")
        self.load_base_model()
        
        # Step 3: Load training data from all databases
        logger.info("Step 3: Loading multi-database training data...")
        train_dataset = self.load_multi_database_training_data(database_paths)
        
        # Step 4: Tokenize dataset
        logger.info("Step 4: Tokenizing dataset...")
        tokenized_dataset = self.tokenize_dataset(train_dataset)
        
        # Step 5: Train the model
        logger.info("Step 5: Training model...")
        self.train(tokenized_dataset, output_dir)
        
        # Step 6: Save training metadata
        self._save_training_metadata(database_paths, output_dir)
        
        logger.info("✅ Multi-domain training pipeline completed successfully!")
        return output_dir
    
    def train(self, train_dataset: Dataset, output_dir: str = "./mistral-nl2sql-trained"):
        """Fine-tune the model on the training data."""
        logger.info("Starting training...")
        
        # Enhanced training configuration for multi-domain learning
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,                    # More epochs for multi-domain
            per_device_train_batch_size=1,         # Smaller batch size for stability
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,        # Higher accumulation
            learning_rate=1e-4,                    # Slightly lower learning rate
            weight_decay=0.01,
            warmup_steps=200,                      # More warmup steps
            logging_steps=25,
            save_steps=500,
            save_total_limit=3,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=False,
            bf16=False,
            dataloader_num_workers=0,
            report_to=None,
            load_best_model_at_end=False,
            metric_for_best_model="loss",
            greater_is_better=False,
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create and run trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training complete! Model saved to {output_dir}")
        return output_dir
    
    def _save_training_metadata(self, database_paths: Dict[str, str], output_dir: str):
        """Save metadata about the training process."""
        metadata = {
            "model_name": self.model_name,
            "domains": list(database_paths.keys()),
            "databases": database_paths,
            "training_timestamp": str(torch.datetime.now()) if hasattr(torch, 'datetime') else "N/A",
            "device": str(self.device),
            "total_databases": len(database_paths)
        }
        
        metadata_path = os.path.join(output_dir, "training_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to {metadata_path}")
    
    def create_domain_specific_model(self, domain: str, output_dir: str = None):
        """Create a model trained specifically on one domain."""
        if output_dir is None:
            output_dir = f"./mistral-nl2sql-{domain}"
        
        logger.info(f"Creating {domain}-specific model...")
        
        # Generate only this domain's database
        database_paths = self.generate_databases([domain])
        
        # Load base model
        self.load_base_model()
        
        # Load training data for this domain only
        train_dataset = self.load_multi_database_training_data(database_paths)
        
        # Tokenize
        tokenized_dataset = self.tokenize_dataset(train_dataset)
        
        # Train with domain-specific configuration
        self.train(tokenized_dataset, output_dir)
        
        # Save metadata
        self._save_training_metadata(database_paths, output_dir)
        
        logger.info(f"✅ {domain}-specific model created: {output_dir}")
        return output_dir


def main():
    """Demo of the enhanced multi-domain training system."""
    trainer = EnhancedMistralNL2SQLTrainer()
    
    print("=== Multi-Domain NL2SQL Training System ===\n")
    
    # Option 1: Train on all domains
    print("1. Training on all available domains...")
    model_path = trainer.train_multi_domain()
    print(f"✅ Multi-domain model saved to: {model_path}\n")
    
    # Option 2: Train on specific domains
    print("2. Training on specific domains (company + finance)...")
    specific_domains = ['company', 'finance']
    specific_model_path = trainer.train_multi_domain(
        domains=specific_domains, 
        output_dir="./mistral-nl2sql-business"
    )
    print(f"✅ Business-focused model saved to: {specific_model_path}\n")
    
    # Option 3: Create domain-specific model
    print("3. Creating healthcare-specific model...")
    healthcare_model = trainer.create_domain_specific_model('healthcare')
    print(f"✅ Healthcare model saved to: {healthcare_model}\n")
    
    print("🎉 All training completed successfully!")


if __name__ == "__main__":
    main()
    