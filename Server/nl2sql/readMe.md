# Mistral NL2SQL Fine-tuning Project

A comprehensive implementation for fine-tuning Mistral 7B to convert natural language questions into SQL queries using LoRA (Low-Rank Adaptation).

## 🎯 Overview

This project fine-tunes the Mistral 7B language model to understand natural language questions about databases and generate corresponding SQL queries. It uses parameter-efficient fine-tuning through LoRA to make training feasible on consumer hardware while maintaining high performance.

## 🧠 Why Mistral 7B?

### Model Architecture

Mistral 7B is based on the transformer architecture with several key improvements:

1. **Grouped-Query Attention (GQA)**: Reduces memory usage during inference while maintaining performance
2. **Sliding Window Attention**: Allows the model to handle longer sequences efficiently
3. **Optimized Architecture**: Better performance per parameter compared to similar-sized models

### Why Mistral for NL2SQL?

- **Size Efficiency**: 7B parameters provide a good balance between capability and resource requirements
- **Strong Reasoning**: Excellent at understanding complex relationships (crucial for SQL generation)
- **Code Understanding**: Pre-trained on code datasets, making it naturally suited for SQL
- **Fine-tuning Friendly**: Responds well to instruction fine-tuning

## 🔧 Mistral Architecture Deep Dive

### Core Components We're Training

#### 1. Multi-Head Attention Mechanism

The heart of transformer models, where the model learns relationships between different parts of the input:

```python
# These are the specific components we target with LoRA:
target_modules = [
    "q_proj",  # Query projection - "What am I looking for?"
    "k_proj",  # Key projection - "What information is available?"
    "v_proj",  # Value projection - "What is the actual content?"
    "o_proj",  # Output projection - "How do I combine this information?"
]
```

**Why these components?**

- **Query (q_proj)**: Learns what database information the natural language question is asking for
- **Key (k_proj)**: Learns to identify relevant schema elements and data relationships
- **Value (v_proj)**: Learns the actual content and meaning of database elements
- **Output (o_proj)**: Learns how to combine all information into proper SQL syntax

#### 2. The Attention Process in NL2SQL Context

When processing "Show me customers from California", the attention mechanism:

1. **Query**: "I need customer information with location filtering"
2. **Key**: Identifies relevant schema elements (customers table, location column)
3. **Value**: Extracts the semantic meaning ("California" = location filter)
4. **Output**: Combines into SQL: `SELECT * FROM customers WHERE state = 'California'`

### Self-Attention Layers

Mistral 7B has 32 transformer layers, each containing:

- Multi-head attention (32 heads)
- Feed-forward networks
- Layer normalization
- Residual connections

**What Each Layer Learns:**

- **Early layers (1-10)**: Basic syntax, tokenization, simple patterns
- **Middle layers (11-20)**: Complex relationships, schema understanding, join logic
- **Late layers (21-32)**: High-level reasoning, query optimization, output formatting

## 🎛️ LoRA: The Magic Behind Efficient Fine-tuning

### What is LoRA?

LoRA (Low-Rank Adaptation) is a technique that adds small trainable matrices to existing model weights instead of training the entire model.

### Mathematical Foundation

Instead of updating the full weight matrix W, LoRA adds a low-rank decomposition:

```
W_new = W_original + B × A
```

Where:

- `W_original`: Frozen original weights (7B parameters)
- `A`: Small matrix (rank r × input_dim)
- `B`: Small matrix (output_dim × rank r)
- Total new parameters: Only ~16M instead of 7B!

### LoRA Configuration Explained

```python
lora_config = LoraConfig(
    r=16,                    # Rank - Controls adaptation capacity
    lora_alpha=32,           # Scaling factor - Controls adaptation strength
    target_modules=[...],    # Which layers to adapt
    lora_dropout=0.1,        # Regularization
    bias="none",             # Don't adapt bias terms
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False
)
```

**Parameter Deep Dive:**

- **r=16**: Creates 16-dimensional bottleneck. Higher r = more capacity but more parameters
- **lora_alpha=32**: Scales the adaptation. alpha/r ratio controls how much LoRA affects the model
- **dropout=0.1**: Prevents overfitting by randomly zeroing 10% of adaptation weights during training

### Memory and Parameter Efficiency

- **Original Mistral 7B**: ~7 billion parameters
- **LoRA Adapters**: ~16 million parameters (0.2% of original)
- **Memory Usage**: ~13GB instead of ~28GB during training
- **Training Speed**: 3-5x faster than full fine-tuning

## 🔤 Tokenization: Converting Text to Numbers

### What is Tokenization?

Tokenization breaks text into subword units that the model can understand. Neural networks work with numbers, not text.

### Mistral's Tokenization Process

1. **Byte-Pair Encoding (BPE)**: Mistral uses BPE to create a vocabulary of ~32k tokens
2. **Subword Units**: Words are broken into meaningful pieces

```python
# Example tokenization:
"SELECT customers FROM database"
# Becomes something like:
[SELECT] [customers] [FROM] [database]
# Which becomes numbers:
[1234, 5678, 9012, 3456]
```

### Special Tokens in Our Implementation

```python
# Padding token for batch processing
if self.tokenizer.pad_token is None:
    self.tokenizer.pad_token = self.tokenizer.eos_token
```

**Why padding?** When training in batches, all sequences must be the same length. Padding fills shorter sequences.

### Tokenization in NL2SQL Context

Our prompt template gets tokenized as:

```
### Task: Convert natural language to SQL
### Database Schema:
{schema_json}
### Question:
{user_question}
### SQL Query:
{target_sql}
```

Each section gets converted to token IDs that the model learns to associate with specific types of information.

## 🏗️ Project Structure

```
mistral-nl2sql/
├── main.py              # Main application entry point
├── trainer.py           # Training module
├── inference.py         # Testing and inference module
├── train_data.json      # Training examples
├── test_data.json       # Test examples
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Installation and Setup

### System Requirements

- **RAM**: 16GB+ recommended (12GB minimum)
- **Storage**: 15GB+ free space
- **Python**: 3.8 or higher
- **GPU**: Optional but recommended (supports Apple Silicon MPS, CUDA, or CPU)

### Installation Steps

1. **Clone or create the project directory**:

```bash
mkdir mistral-nl2sql
cd mistral-nl2sql
```

2. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install peft accelerate
pip install bitsandbytes  # For optimization
```

4. **Copy the code files** (trainer.py, inference.py, main.py) to your project directory.

## 📊 Data Format Specification

### Training Data Format (`train_data.json`)

```json
[
  {
    "question": "How many customers are there?",
    "schema": {
      "tables": [
        {
          "name": "customers",
          "columns": [
            { "name": "id", "type": "INT", "primary_key": true },
            { "name": "name", "type": "VARCHAR(100)" },
            { "name": "email", "type": "VARCHAR(100)" }
          ]
        }
      ]
    },
    "business_rules": ["Count all customers in the database"],
    "sql": "SELECT COUNT(*) FROM customers;"
  }
]
```

### Test Data Format (`test_data.json`)

```json
[
  {
    "question": "Show all customer names",
    "schema": {
      "tables": [
        {
          "name": "customers",
          "columns": [
            { "name": "id", "type": "INT" },
            { "name": "name", "type": "VARCHAR(100)" }
          ]
        }
      ]
    },
    "business_rules": [],
    "expected_sql": "SELECT name FROM customers;"
  }
]
```

## 🎮 Usage Guide

### Quick Start

```bash
# 1. Create sample data
python main.py create-sample-data

# 2. Train model
python main.py train

# 3. Test model
python main.py test

# 4. Run interactive mode
python main.py interactive

# 5. Run complete pipeline
python main.py pipeline
```

### Advanced Usage

#### Training with Custom Settings

```bash
python main.py train --output-dir ./my-custom-model
```

#### Testing with Custom Model

```bash
python main.py test --model-path ./my-custom-model --results-file my_results.json
```

#### Interactive Mode

```bash
python main.py interactive --model-path ./my-custom-model
```

## 🧪 Training Process Deep Dive

### Phase 1: Model Loading

```python
# Load base Mistral model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,  # Half precision saves memory
    device_map="auto"           # Automatic device placement
)

# Apply LoRA adapters
model = get_peft_model(model, lora_config)
```

### Phase 2: Data Preprocessing

1. **Template Formatting**: Convert examples to consistent prompt format
2. **Tokenization**: Convert text to token IDs
3. **Batching**: Group examples for efficient training

### Phase 3: Training Loop

For each batch:

1. **Forward Pass**: Model predicts next tokens given input
2. **Loss Calculation**: Compare predictions to target SQL
3. **Backward Pass**: Calculate gradients for LoRA parameters only
4. **Parameter Update**: Update only the small LoRA matrices

### Training Configuration Explained

```python
TrainingArguments(
    num_train_epochs=3,              # See data 3 times
    per_device_train_batch_size=2,   # 2 examples per batch (memory conscious)
    gradient_accumulation_steps=8,   # Accumulate gradients over 8 batches (effective batch size = 16)
    learning_rate=2e-4,              # How fast to learn (0.0002)
    weight_decay=0.01,               # L2 regularization
    warmup_steps=100,                # Gradual learning rate increase
)
```

## 🔍 Code Analysis: Key Components

### 1. Prompt Template Design

```python
def format_training_example(self, example: Dict[str, Any]) -> str:
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
```

**Why this format?**

- **Clear Sections**: Model learns to distinguish between schema, rules, question, and output
- **Consistent Structure**: Same format for training and inference
- **End Marker**: Helps model know when to stop generating

### 2. Token Generation During Inference

```python
outputs = self.model.generate(
    **inputs,
    max_new_tokens=200,      # Limit output length
    temperature=0.1,         # Low temperature = more focused output
    do_sample=True,          # Enable sampling for slight variation
    repetition_penalty=1.1,  # Discourage repetitive output
)
```

**Parameter Meanings:**

- **max_new_tokens**: Prevents runaway generation
- **temperature**: Controls randomness (0.1 = very focused, 1.0 = more creative)
- **do_sample**: Adds slight randomness to prevent identical outputs
- **repetition_penalty**: Reduces repetitive phrases

### 3. Device Detection and Optimization

```python
if torch.backends.mps.is_available():
    self.device = torch.device("mps")      # Apple Silicon
elif torch.cuda.is_available():
    self.device = torch.device("cuda")     # NVIDIA GPU
else:
    self.device = torch.device("cpu")      # CPU fallback
```

This ensures the model runs on the best available hardware automatically.

## 📈 Performance and Optimization

### Memory Usage Optimization

1. **Half Precision (fp16)**: Reduces memory by 50%
2. **LoRA**: Reduces trainable parameters by 99.8%
3. **Gradient Accumulation**: Simulates larger batches without memory cost
4. **Device Mapping**: Automatically distributes model across available hardware

### Training Speed Optimization

1. **Batch Processing**: Process multiple examples simultaneously
2. **Efficient Tokenization**: Vectorized operations
3. **Gradient Checkpointing**: Trade computation for memory
4. **Hardware Acceleration**: GPU/MPS support

### Expected Performance

- **Training Time**: 2-4 hours on M1 Pro, 1-2 hours on modern GPU
- **Memory Usage**: 12-16GB RAM during training
- **Inference Speed**: 1-3 seconds per query
- **Model Size**: ~32MB for LoRA adapters (vs 13GB for full model)

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Errors

```bash
# Reduce batch size
per_device_train_batch_size=1

# Increase gradient accumulation
gradient_accumulation_steps=16

# Use CPU if necessary
CUDA_VISIBLE_DEVICES="" python main.py train
```

#### 2. Slow Training

```bash
# Check GPU usage
nvidia-smi  # For NVIDIA
# or
Activity Monitor  # For Mac
```

#### 3. Poor Model Performance

- **Increase training data**: More examples = better performance
- **Improve data quality**: Clean, consistent examples
- **Adjust LoRA rank**: Higher rank = more capacity
- **Tune learning rate**: Try 1e-4 or 5e-4

#### 4. Installation Issues

```bash
# Update pip and try again
pip install --upgrade pip
pip install --upgrade torch transformers

# For Apple Silicon Macs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 🔬 Advanced Configurations

### Custom LoRA Settings

```python
# For more complex tasks
lora_config = LoraConfig(
    r=32,                    # Higher rank for more capacity
    lora_alpha=64,           # Stronger adaptation
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # Include FFN layers
    ]
)
```

### Training on Multiple GPUs

```python
# Use DataParallel or DistributedDataParallel
training_args = TrainingArguments(
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    # ... other args
)
```

## 📝 Evaluation Metrics

### Automatic Evaluation

The testing module provides:

- **Exact Match Accuracy**: Percentage of exactly matching SQL queries
- **Error Analysis**: Common failure patterns
- **Performance Metrics**: Speed and resource usage

### Manual Evaluation Criteria

1. **Syntactic Correctness**: Does the SQL parse without errors?
2. **Semantic Correctness**: Does it answer the intended question?
3. **Efficiency**: Is the query reasonably optimized?
4. **Schema Compliance**: Does it use correct table/column names?

## 🚀 Production Deployment

### Model Serving Options

1. **FastAPI Server**: Create REST API endpoints
2. **Gradio Interface**: Quick web interface
3. **Streamlit App**: Interactive dashboard
4. **Docker Container**: Containerized deployment

### Example FastAPI Server

```python
from fastapi import FastAPI
from inference import MistralNL2SQLInference

app = FastAPI()
inference = MistralNL2SQLInference()
inference.load_model("./mistral-nl2sql-trained")

@app.post("/generate-sql")
async def generate_sql(question: str, schema: dict):
    sql = inference.generate_sql(question, schema)
    return {"sql": sql}
```

## 📚 Further Reading

### Research Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)
- [Text-to-SQL Survey](https://arxiv.org/abs/2208.13629)

### Related Projects

- [Spider Dataset](https://yale-lily.github.io/spider): Large-scale text-to-SQL dataset
- [HuggingFace PEFT](https://github.com/huggingface/peft): Parameter-Efficient Fine-tuning library
- [Mistral AI](https://mistral.ai/): Official Mistral models and documentation

## 🤝 Contributing

### How to Contribute

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Areas for Improvement

- Multi-table join optimization
- Schema-aware validation
- Query explanation generation
- Support for more SQL dialects
- Integration with database engines

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Mistral AI for the excellent base model
- HuggingFace for the transformers library and PEFT
- The research community for LoRA and fine-tuning techniques

---

## Quick Reference

### Essential Commands

```bash
# Setup
python main.py create-sample-data

# Training
python main.py train

# Testing
python main.py test

# Interactive use
python main.py interactive

# Complete pipeline
python main.py pipeline
```

### File Structure

- `main.py` - Command-line interface
- `trainer.py` - Training logic
- `inference.py` - Testing and inference
- `train_data.json` - Training examples
- `test_data.json` - Test cases

### Key Concepts

- **LoRA**: Parameter-efficient fine-tuning
- **Tokenization**: Text to number conversion
- **Attention**: Relationship learning mechanism
- **Temperature**: Controls output randomness
- **Prompt Template**: Structured input format
