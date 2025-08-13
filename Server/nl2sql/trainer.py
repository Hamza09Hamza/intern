import torch
import json
import os
import logging
from typing import Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MistralNL2SQLTrainer:
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):

        self.model_name = model_name
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
    
    def load_base_model(self):

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
        
        # Configure LoRA for parameter-efficient fine-tuning
        lora_config = LoraConfig(
            r=16,                       # Rank of adaptation
            lora_alpha=32,              # Scaling factor
            target_modules=[            # Attention modules to adapt
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        
        # Apply LoRA adapters
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Base model and LoRA adapters loaded successfully")
    
    def format_training_example(self, example: Dict[str, Any]) -> str:
        """
        Format a training example into the expected prompt template.
        
        Args:
            example (dict): Training example with question, schema, and SQL
            
        Returns:
            str: Formatted training prompt
        """
        question = example.get('question', '')
        schema = example.get('schema', {})
        business_rules = example.get('business_rules', [])
        sql_output = example.get('sql', '')
        
        prompt = f"""### Task: Convert natural language to SQL

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
    
    def load_training_data(self, train_data_path: str) -> Dataset:
        """
        Load and preprocess training data from JSON file.
        
        Args:
            train_data_path (str): Path to training JSON file
            
        Returns:
            Dataset: Processed dataset ready for training
        """
        logger.info(f"Loading training data from: {train_data_path}")
        
        with open(train_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Format each example
        formatted_texts = []
        for example in raw_data:
            formatted_text = self.format_training_example(example)
            formatted_texts.append({"text": formatted_text})
        
        dataset = Dataset.from_list(formatted_texts)
        logger.info(f"Loaded {len(dataset)} training examples")
        return dataset
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """
        Convert text data to tokens for model processing.
        
        Args:
            dataset (Dataset): Dataset with text examples
            
        Returns:
            Dataset: Tokenized dataset
        """
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
    
    def train(self, train_dataset: Dataset, output_dir: str = "./mistral-nl2sql-trained"):
        """
        Fine-tune the model on the training data.
        
        Args:
            train_dataset (Dataset): Tokenized training dataset
            output_dir (str): Directory to save the trained model
        """
        logger.info("Starting training...")
        
        # Training configuration optimized for efficiency
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=False,
            bf16=False,
            dataloader_num_workers=0,
            report_to=None,
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