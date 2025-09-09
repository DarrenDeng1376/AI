# SmallGPT: Technical Overview and Academic References

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Training Pipeline](#training-pipeline)
4. [Tokenization](#tokenization)
5. [Text Generation](#text-generation)
6. [Implementation Details](#implementation-details)
7. [Academic References](#academic-references)
8. [Related Work](#related-work)

## Overview

SmallGPT is a minimal implementation of the GPT (Generative Pre-trained Transformer) architecture, designed for educational purposes and research experimentation. The implementation follows the decoder-only transformer architecture popularized by the GPT series of models.

### Key Features
- **Autoregressive Language Modeling**: Left-to-right text generation
- **Transformer Architecture**: Multi-head self-attention and feed-forward networks
- **Byte-Pair Encoding (BPE)**: Subword tokenization for vocabulary efficiency
- **Causal Masking**: Ensures models can only attend to previous tokens
- **Configurable Scale**: From toy models to medium-scale experiments

---

## Architecture Components

### 1. Transformer Decoder Architecture

The model implements a stack of transformer decoder blocks, each containing:

#### **Multi-Head Self-Attention (MHSA)**

**Formula:**
```
Attention(Q,K,V) = softmax(QKᵀ/√d_k)V
MultiHead(Q,K,V) = Concat(head₁,...,head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Key Properties:**
- **Causal Masking**: Lower triangular mask prevents future token access
- **Scaled Dot-Product**: Divides by √d_k for gradient stability
- **Multi-Head Design**: Allows model to attend to different representation subspaces

**Academic Reference:** Vaswani et al. (2017) - *Attention Is All You Need* [1]

#### **Position-wise Feed-Forward Network (FFN)**

**Architecture:**
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Properties:**
- **ReLU Activation**: Non-linear transformation between layers
- **Expansion Factor**: Hidden dimension typically 4× the model dimension
- **Position-wise**: Applied independently to each position

#### **Layer Normalization**

**Formula:**
```
LayerNorm(x) = γ ⊙ (x - μ)/σ + β
where μ = mean(x), σ = std(x)
```

**Implementation:** Pre-normalization (applied before sub-layers) for training stability

**Academic Reference:** Ba et al. (2016) - *Layer Normalization* [2]

#### **Residual Connections**

**Pattern:**
```
x = x + SubLayer(LayerNorm(x))
```

**Benefits:**
- **Gradient Flow**: Helps with deep network training
- **Identity Mapping**: Allows layers to learn residual functions

**Academic Reference:** He et al. (2016) - *Deep Residual Learning* [3]

### 2. Embeddings

#### **Token Embeddings**
- **Learnable Matrix**: E ∈ ℝ^(V×d) where V is vocab size, d is model dimension
- **Lookup Operation**: Maps token IDs to dense vectors

#### **Positional Embeddings**
- **Learnable Positions**: Absolute position embeddings for each sequence position
- **Alternative**: Could use sinusoidal encodings (Vaswani et al.) or rotary embeddings (Su et al.)

**Academic Reference:** Su et al. (2021) - *RoFormer: Enhanced Transformer with Rotary Position Embedding* [4]

---

## Training Pipeline

### 1. Autoregressive Language Modeling

**Objective Function:**
```
L = -∑ᵢ log P(xᵢ₊₁ | x₁, x₂, ..., xᵢ; θ)
```

Where:
- **x₁, x₂, ..., xᵢ**: Input sequence up to position i
- **xᵢ₊₁**: Next token to predict
- **θ**: Model parameters

### 2. Optimization Strategy

#### **AdamW Optimizer**
**Parameters:**
- **Learning Rate**: Typically 1e-4 to 5e-4
- **Weight Decay**: 0.1 for regularization
- **Beta Values**: β₁=0.9, β₂=0.999

**Academic Reference:** Loshchilov & Hutter (2017) - *Decoupled Weight Decay Regularization* [5]

#### **Learning Rate Scheduling**
**Warmup + Cosine Decay:**
```
lr(t) = lr_max * min(t/warmup_steps, 0.5 * (1 + cos(π * (t - warmup_steps) / decay_steps)))
```

#### **Gradient Clipping**
**Global Norm Clipping:**
```
g_clipped = g * min(1, max_norm / ||g||₂)
```

**Purpose**: Prevents exploding gradients in deep networks

### 3. Training Techniques

#### **Gradient Accumulation**
- **Effective Batch Size**: Accumulate gradients over multiple mini-batches
- **Memory Efficiency**: Enables larger effective batch sizes with limited GPU memory

#### **Mixed Precision Training**
- **FP16 Forward Pass**: Reduces memory usage and increases throughput
- **FP32 Master Weights**: Maintains numerical stability

**Academic Reference:** Micikevicius et al. (2017) - *Mixed Precision Training* [6]

---

## Tokenization

### Byte-Pair Encoding (BPE)

**Algorithm Overview:**
1. **Initialize**: Start with character-level vocabulary
2. **Count Pairs**: Find most frequent adjacent character pairs
3. **Merge**: Replace most frequent pair with new symbol
4. **Iterate**: Repeat until desired vocabulary size

**Advantages:**
- **Subword Units**: Handles out-of-vocabulary words gracefully
- **Compression**: Efficient representation of common patterns
- **Language Agnostic**: Works across different languages and domains

**Academic Reference:** Sennrich et al. (2016) - *Neural Machine Translation of Rare Words with Subword Units* [7]

### Implementation Details

#### **Training Phase:**
```python
def train_bpe(texts, vocab_size):
    word_freq = get_word_frequencies(texts)
    
    for i in range(num_merges):
        pairs = get_pairs(word_freq)
        best_pair = max(pairs, key=pairs.get)
        word_freq = merge_vocab(best_pair, word_freq)
        merges.append(best_pair)
```

#### **Encoding Phase:**
```python
def encode(text):
    words = pre_tokenize(text)  # Split into words
    tokens = []
    for word in words:
        tokens.extend(apply_bpe(word))  # Apply learned merges
    return tokens
```

---

## Text Generation

### 1. Sampling Strategies

#### **Temperature Sampling**
**Formula:**
```
P(x) = softmax(logits / τ)
where τ is temperature
```

**Effects:**
- **τ → 0**: Deterministic (argmax)
- **τ = 1**: Unmodified distribution
- **τ > 1**: More random, flattened distribution

#### **Top-k Sampling**
**Process:**
1. Select top-k highest probability tokens
2. Redistribute probability mass among selected tokens
3. Sample from truncated distribution

**Academic Reference:** Fan et al. (2018) - *Hierarchical Neural Story Generation* [8]

#### **Top-p (Nucleus) Sampling**
**Process:**
1. Sort tokens by probability
2. Select smallest set with cumulative probability ≥ p
3. Sample from selected tokens

**Benefits:**
- **Adaptive Vocabulary**: Adjusts vocabulary size based on distribution shape
- **Quality Control**: Maintains coherence while allowing diversity

**Academic Reference:** Holtzman et al. (2019) - *The Curious Case of Neural Text Degeneration* [9]

### 2. Beam Search

**Algorithm:**
```
Initialize: beams = [initial_sequence]
For each step:
    candidates = []
    For each beam:
        For each possible next token:
            candidates.append(beam + token)
    beams = top_k(candidates, k=beam_width)
```

**Trade-offs:**
- **Deterministic**: More predictable outputs
- **Search Quality**: Finds high-probability sequences
- **Computational Cost**: k times more expensive than greedy

---

## Implementation Details

### 1. Model Architecture Code Structure

```python
class SmallGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
```

### 2. Training Loop Structure

```python
def training_step(model, batch, optimizer):
    # Forward pass
    logits = model(batch['input_ids'])
    
    # Compute loss (shifted for next-token prediction)
    targets = batch['input_ids'][:, 1:]
    logits = logits[:, :-1]
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

### 3. Performance Optimizations

#### **Flash Attention**
- **Memory Efficiency**: O(√N) memory complexity instead of O(N²)
- **Speed**: 2-4x faster attention computation

**Academic Reference:** Dao et al. (2022) - *FlashAttention: Fast and Memory-Efficient Exact Attention* [10]

#### **Gradient Checkpointing**
- **Memory-Speed Trade-off**: Recompute activations during backward pass
- **Deep Networks**: Enables training deeper models with limited memory

---

## Academic References

### Core Papers

**[1] Attention Is All You Need**
- *Authors*: Vaswani, A., Shazeer, N., Parmar, N., et al.
- *Venue*: NIPS 2017
- *Link*: https://arxiv.org/abs/1706.03762
- *Contribution*: Introduced the transformer architecture

**[2] Layer Normalization**
- *Authors*: Ba, J. L., Kiros, J. R., & Hinton, G. E.
- *Venue*: arXiv preprint 2016
- *Link*: https://arxiv.org/abs/1607.06450
- *Contribution*: Layer normalization technique

**[3] Deep Residual Learning for Image Recognition**
- *Authors*: He, K., Zhang, X., Ren, S., & Sun, J.
- *Venue*: CVPR 2016
- *Link*: https://arxiv.org/abs/1512.03385
- *Contribution*: Residual connections

**[4] RoFormer: Enhanced Transformer with Rotary Position Embedding**
- *Authors*: Su, J., Lu, Y., Pan, S., et al.
- *Venue*: arXiv preprint 2021
- *Link*: https://arxiv.org/abs/2104.09864
- *Contribution*: Rotary position embeddings

**[5] Decoupled Weight Decay Regularization**
- *Authors*: Loshchilov, I., & Hutter, F.
- *Venue*: ICLR 2019
- *Link*: https://arxiv.org/abs/1711.05101
- *Contribution*: AdamW optimizer

### GPT Series Papers

**[6] Improving Language Understanding by Generative Pre-Training**
- *Authors*: Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I.
- *Venue*: OpenAI Technical Report 2018
- *Link*: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- *Contribution*: Original GPT model

**[7] Language Models are Unsupervised Multitask Learners**
- *Authors*: Radford, A., Wu, J., Child, R., et al.
- *Venue*: OpenAI Technical Report 2019
- *Link*: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- *Contribution*: GPT-2 and scaling insights

**[8] Language Models are Few-Shot Learners**
- *Authors*: Brown, T., Mann, B., Ryder, N., et al.
- *Venue*: NeurIPS 2020
- *Link*: https://arxiv.org/abs/2005.14165
- *Contribution*: GPT-3 and in-context learning

### Tokenization

**[9] Neural Machine Translation of Rare Words with Subword Units**
- *Authors*: Sennrich, R., Haddow, B., & Birch, A.
- *Venue*: ACL 2016
- *Link*: https://arxiv.org/abs/1508.07909
- *Contribution*: Byte-pair encoding for NLP

### Text Generation

**[10] Hierarchical Neural Story Generation**
- *Authors*: Fan, A., Lewis, M., & Dauphin, Y.
- *Venue*: ACL 2018
- *Link*: https://arxiv.org/abs/1805.04833
- *Contribution*: Top-k sampling

**[11] The Curious Case of Neural Text Degeneration**
- *Authors*: Holtzman, A., Buys, J., Du, L., et al.
- *Venue*: ICLR 2020
- *Link*: https://arxiv.org/abs/1904.09751
- *Contribution*: Top-p (nucleus) sampling

### Optimization

**[12] Mixed Precision Training**
- *Authors*: Micikevicius, P., Narang, S., Alben, J., et al.
- *Venue*: ICLR 2018
- *Link*: https://arxiv.org/abs/1710.03740
- *Contribution*: FP16 training techniques

**[13] FlashAttention: Fast and Memory-Efficient Exact Attention**
- *Authors*: Dao, T., Fu, D. Y., Ermon, S., et al.
- *Venue*: NeurIPS 2022
- *Link*: https://arxiv.org/abs/2205.14135
- *Contribution*: Efficient attention computation

---

## Related Work

### Alternative Architectures

**BERT (Bidirectional Encoder Representations from Transformers)**
- *Paper*: Devlin et al. (2018) - https://arxiv.org/abs/1810.04805
- *Difference*: Bidirectional encoder vs. autoregressive decoder

**T5 (Text-to-Text Transfer Transformer)**
- *Paper*: Raffel et al. (2019) - https://arxiv.org/abs/1910.10683
- *Difference*: Encoder-decoder architecture with unified text-to-text framework

### Efficiency Improvements

**Linformer**
- *Paper*: Wang et al. (2020) - https://arxiv.org/abs/2006.04768
- *Contribution*: Linear complexity attention mechanism

**Performer**
- *Paper*: Choromanski et al. (2020) - https://arxiv.org/abs/2009.14794
- *Contribution*: Fast attention via positive orthogonal random features

### Scaling Studies

**Scaling Laws for Neural Language Models**
- *Authors*: Kaplan et al. (2020)
- *Link*: https://arxiv.org/abs/2001.08361
- *Contribution*: Empirical scaling relationships for transformer models

**Training Compute-Optimal Large Language Models**
- *Authors*: Hoffmann et al. (2022)
- *Link*: https://arxiv.org/abs/2203.15556
- *Contribution*: Optimal compute allocation (Chinchilla scaling laws)

---

## Implementation Resources

### Code References

1. **Andrej Karpathy's minGPT**: https://github.com/karpathy/minGPT
2. **Hugging Face Transformers**: https://github.com/huggingface/transformers
3. **OpenAI GPT-2**: https://github.com/openai/gpt-2
4. **Karpathy's nanoGPT**: https://github.com/karpathy/nanoGPT

### Educational Resources

1. **The Illustrated GPT-2**: https://jalammar.github.io/illustrated-gpt2/
2. **The Annotated Transformer**: https://nlp.seas.harvard.edu/2018/04/03/attention.html
3. **Transformer from Scratch**: https://peterbloem.nl/blog/transformers

### Datasets for Training

1. **OpenWebText**: https://github.com/jcpeterson/openwebtext
2. **The Pile**: https://pile.eleuther.ai/
3. **WikiText**: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
4. **BookCorpus**: https://yknzhu.wixsite.com/mbweb

---

*This documentation provides a comprehensive technical overview of the SmallGPT implementation with academic references for further study. For implementation details, refer to the source code and configuration files in the repository.*
