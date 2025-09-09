# SmallGPT Training Pipeline: Step-by-Step Guide

## 🎯 Training Overview

This document provides a detailed walkthrough of the SmallGPT training process, from data preparation to model deployment.

---

## 📋 Training Pipeline Steps

### Step 1: Data Preparation and Tokenization

#### 1.1 Raw Text Collection
```python
# Example: Loading text data
texts = [
    "Once upon a time, in a small village...",
    "The human brain is one of the most complex...",
    "Artificial intelligence is transforming..."
]
```

#### 1.2 BPE Tokenizer Training
**Algorithm**: Byte-Pair Encoding (Sennrich et al., 2016)

**Process:**
1. **Character-level initialization**: Start with individual characters
2. **Frequency counting**: Count adjacent character pair frequencies
3. **Iterative merging**: Merge most frequent pairs into new tokens
4. **Vocabulary building**: Build final vocabulary of specified size

**Code Example:**
```python
tokenizer = SimpleTokenizer(vocab_size=10000)
tokenizer.train(texts)  # Learn BPE merges from training data
tokenizer.save("models/tokenizer.pkl")
```

**Time Complexity**: O(n × m × v) where:
- n = number of merge operations
- m = average text length
- v = vocabulary size

#### 1.3 Text Preprocessing
```python
# Convert texts to token sequences
dataset = TextDataset(
    texts=texts,
    tokenizer=tokenizer,
    max_seq_len=1024,  # Context window size
    stride=512         # Overlapping window stride
)
```

**Reference**: Radford et al. (2019) - GPT-2 preprocessing techniques

---

### Step 2: Model Architecture Setup

#### 2.1 Transformer Configuration
```yaml
model:
  vocab_size: 10000      # From tokenizer
  embed_dim: 512         # Hidden dimension
  num_heads: 8           # Multi-head attention
  num_layers: 6          # Transformer blocks
  max_seq_len: 1024      # Context window
  dropout: 0.1           # Regularization
```

#### 2.2 Model Initialization
```python
model = SmallGPT(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=config.model.embed_dim,
    num_heads=config.model.num_heads,
    num_layers=config.model.num_layers,
    max_seq_len=config.model.max_seq_len,
    dropout=config.model.dropout
)
```

**Parameter Count**: ~25M parameters for default configuration

**Reference**: Vaswani et al. (2017) - Transformer architecture

---

### Step 3: Training Configuration

#### 3.1 Optimizer Setup
```python
# AdamW optimizer (Loshchilov & Hutter, 2017)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,           # Learning rate
    betas=(0.9, 0.999), # Adam momentum parameters
    weight_decay=0.1    # L2 regularization
)
```

#### 3.2 Learning Rate Scheduling
```python
# Warmup + Cosine decay
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,     # Linear warmup steps
    num_training_steps=100000   # Total training steps
)
```

**Formula:**
```
lr(t) = lr_max × min(t/warmup_steps, 0.5 × (1 + cos(π × (t-warmup_steps)/(total_steps-warmup_steps))))
```

**Reference**: Vaswani et al. (2017) - Learning rate scheduling

---

### Step 4: Training Loop Implementation

#### 4.1 Forward Pass
```python
def forward_pass(model, batch):
    input_ids = batch['input_ids']  # Shape: [batch_size, seq_len]
    
    # Model forward pass
    logits = model(input_ids)       # Shape: [batch_size, seq_len, vocab_size]
    
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()  # Remove last token
    shift_labels = input_ids[..., 1:].contiguous()   # Remove first token
    
    return shift_logits, shift_labels
```

#### 4.2 Loss Computation
```python
def compute_loss(logits, labels):
    # Cross-entropy loss for next-token prediction
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # Flatten: [batch_size * seq_len, vocab_size]
        labels.view(-1),                   # Flatten: [batch_size * seq_len]
        ignore_index=tokenizer.pad_token_id
    )
    return loss
```

**Objective Function:**
```
L = -∑ᵢ log P(xᵢ₊₁ | x₁, x₂, ..., xᵢ; θ)
```

#### 4.3 Backward Pass and Optimization
```python
def training_step(model, batch, optimizer, scheduler):
    # Forward pass
    logits, labels = forward_pass(model, batch)
    loss = compute_loss(logits, labels)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping (prevents exploding gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    return loss.item()
```

**Reference**: Pascanu et al. (2013) - Gradient clipping techniques

---

### Step 5: Training Monitoring and Validation

#### 5.1 Training Metrics
```python
def log_training_metrics(step, loss, learning_rate, elapsed_time):
    metrics = {
        'train/loss': loss,
        'train/perplexity': math.exp(loss),
        'train/learning_rate': learning_rate,
        'train/step': step,
        'train/elapsed_time': elapsed_time
    }
    
    # Log to Weights & Biases (optional)
    if wandb_available:
        wandb.log(metrics)
```

#### 5.2 Validation Loop
```python
def validate_model(model, val_dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            logits, labels = forward_pass(model, batch)
            loss = compute_loss(logits, labels)
            
            # Accumulate loss and token count
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    model.train()
    return avg_loss, perplexity
```

#### 5.3 Checkpoint Saving
```python
def save_checkpoint(model, optimizer, scheduler, step, loss):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'loss': loss,
        'config': config
    }
    
    torch.save(checkpoint, f'models/checkpoint_step_{step}.pt')
```

---

### Step 6: Text Generation and Evaluation

#### 6.1 Generation Pipeline
```python
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8):
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]  # Last token predictions
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode to text
    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text
```

#### 6.2 Advanced Sampling Strategies

**Top-k Sampling:**
```python
def top_k_sampling(logits, k=50):
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, num_samples=1)
    return top_k_indices[next_token_idx]
```

**Top-p (Nucleus) Sampling:**
```python
def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0
    
    # Set logits to -inf for removed tokens
    logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**References**: 
- Fan et al. (2018) - Top-k sampling
- Holtzman et al. (2019) - Top-p sampling

---

### Step 7: Evaluation Metrics

#### 7.1 Perplexity
```python
def compute_perplexity(model, dataloader):
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            logits, labels = forward_pass(model, batch)
            loss = compute_loss(logits, labels)
            
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity
```

**Formula:** `Perplexity = exp(CrossEntropyLoss)`

#### 7.2 BLEU Score (for specific tasks)
```python
from nltk.translate.bleu_score import sentence_bleu

def compute_bleu_score(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
    return bleu_score
```

---

## 🔧 Training Best Practices

### 1. Hyperparameter Tuning

**Critical Hyperparameters:**
- **Learning Rate**: Start with 5e-4, adjust based on convergence
- **Batch Size**: Larger is generally better (32-128 per GPU)
- **Warmup Steps**: Typically 1-5% of total training steps
- **Dropout**: 0.1-0.2 for regularization

### 2. Training Stability

**Techniques:**
- **Gradient Clipping**: Prevent exploding gradients (max_norm=1.0)
- **Layer Normalization**: Pre-norm for better gradient flow
- **Residual Connections**: Enable deep network training
- **Learning Rate Scheduling**: Warmup + decay for convergence

### 3. Computational Efficiency

**Optimizations:**
- **Mixed Precision Training**: FP16 for memory and speed
- **Gradient Accumulation**: Simulate larger batch sizes
- **DataLoader Optimization**: Multiple workers, pin_memory=True
- **Model Parallelism**: For very large models

**Reference**: Micikevicius et al. (2017) - Mixed precision training

### 4. Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Exploding Gradients | Loss becomes NaN | Gradient clipping, lower learning rate |
| Vanishing Gradients | Loss plateaus early | Pre-norm, residual connections |
| Overfitting | Val loss > Train loss | Dropout, weight decay, more data |
| Slow Convergence | Loss decreases slowly | Higher learning rate, warmup |
| Memory Issues | OOM errors | Smaller batch size, gradient accumulation |

---

## 📊 Training Timeline Example

For a 25M parameter model on modern hardware:

| Phase | Duration | Description |
|-------|----------|-------------|
| **Data Preparation** | 5-30 minutes | Tokenizer training, dataset creation |
| **Model Setup** | 1-2 minutes | Model initialization, optimizer setup |
| **Training (10K steps)** | 1-4 hours | Main training loop |
| **Validation** | 2-5 minutes | Periodic evaluation |
| **Generation Testing** | 1-2 minutes | Sample text generation |

**Hardware Requirements:**
- **GPU**: RTX 3080/4080 or better (12GB+ VRAM)
- **RAM**: 16GB+ system memory
- **Storage**: SSD recommended for data loading

---

## 📚 Additional Resources

### Academic Papers
1. **Attention Is All You Need** - Vaswani et al. (2017)
2. **Language Models are Unsupervised Multitask Learners** - Radford et al. (2019)
3. **The Curious Case of Neural Text Degeneration** - Holtzman et al. (2019)

### Implementation Guides
1. **The Annotated Transformer** - Harvard NLP
2. **Andrej Karpathy's minGPT** - Educational implementation
3. **Hugging Face Transformers** - Production-ready library

### Debugging Tools
1. **Weights & Biases** - Experiment tracking
2. **TensorBoard** - Training visualization
3. **PyTorch Profiler** - Performance analysis

---

*This step-by-step guide provides a comprehensive overview of the SmallGPT training pipeline with academic references and practical implementation details.*
