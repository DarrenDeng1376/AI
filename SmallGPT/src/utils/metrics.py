"""
Evaluation metrics for language models
"""
import torch
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple
import numpy as np


def calculate_perplexity(
    model, 
    dataset, 
    device: torch.device,
    batch_size: int = 32
) -> float:
    """
    Calculate perplexity on a dataset.
    
    Args:
        model: Language model
        dataset: Dataset to evaluate on
        device: Device to run on
        batch_size: Batch size for evaluation
        
    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=model.pad_token_id,
                reduction='sum'
            )
            
            # Count valid tokens
            if attention_mask is not None:
                valid_tokens = attention_mask[..., 1:].sum()
            else:
                valid_tokens = (shift_labels != model.pad_token_id).sum()
            
            total_loss += loss.item()
            total_tokens += valid_tokens.item()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def calculate_bleu_score(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Calculate BLEU score for generated text.
    
    Args:
        references: List of reference texts
        hypotheses: List of generated texts
        
    Returns:
        Dictionary with BLEU scores
    """
    from collections import Counter
    
    def get_ngrams(text: str, n: int) -> List[str]:
        """Get n-grams from text."""
        words = text.lower().split()
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def calculate_precision(ref_ngrams: List[str], hyp_ngrams: List[str]) -> float:
        """Calculate precision for n-grams."""
        if not hyp_ngrams:
            return 0.0
        
        ref_counts = Counter(ref_ngrams)
        hyp_counts = Counter(hyp_ngrams)
        
        overlap = 0
        for ngram, count in hyp_counts.items():
            overlap += min(count, ref_counts.get(ngram, 0))
        
        return overlap / len(hyp_ngrams)
    
    # Calculate BLEU for different n-gram sizes
    bleu_scores = {}
    
    for n in range(1, 5):  # 1-gram to 4-gram
        precisions = []
        
        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = get_ngrams(ref, n)
            hyp_ngrams = get_ngrams(hyp, n)
            
            precision = calculate_precision(ref_ngrams, hyp_ngrams)
            precisions.append(precision)
        
        bleu_scores[f'bleu_{n}'] = np.mean(precisions)
    
    # Calculate overall BLEU (geometric mean)
    bleu_scores['bleu'] = np.power(
        np.prod([bleu_scores[f'bleu_{n}'] for n in range(1, 5)]), 1/4
    )
    
    return bleu_scores


def calculate_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate diversity metrics for generated texts.
    
    Args:
        texts: List of generated texts
        
    Returns:
        Dictionary with diversity metrics
    """
    from collections import Counter
    
    # Combine all texts
    all_words = []
    all_bigrams = []
    all_trigrams = []
    
    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
        
        # Bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        all_bigrams.extend(bigrams)
        
        # Trigrams
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
        all_trigrams.extend(trigrams)
    
    # Calculate unique ratios
    metrics = {}
    
    if all_words:
        metrics['unique_words'] = len(set(all_words)) / len(all_words)
    else:
        metrics['unique_words'] = 0.0
    
    if all_bigrams:
        metrics['unique_bigrams'] = len(set(all_bigrams)) / len(all_bigrams)
    else:
        metrics['unique_bigrams'] = 0.0
    
    if all_trigrams:
        metrics['unique_trigrams'] = len(set(all_trigrams)) / len(all_trigrams)
    else:
        metrics['unique_trigrams'] = 0.0
    
    # Calculate entropy
    word_counts = Counter(all_words)
    total_words = len(all_words)
    
    if total_words > 0:
        word_probs = [count / total_words for count in word_counts.values()]
        entropy = -sum(p * math.log2(p) for p in word_probs if p > 0)
        metrics['entropy'] = entropy
    else:
        metrics['entropy'] = 0.0
    
    return metrics


def calculate_repetition_metrics(text: str) -> Dict[str, float]:
    """
    Calculate repetition metrics for a single text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with repetition metrics
    """
    words = text.lower().split()
    
    if not words:
        return {
            'repetition_2': 0.0,
            'repetition_3': 0.0,
            'repetition_4': 0.0
        }
    
    metrics = {}
    
    # Calculate n-gram repetition
    for n in [2, 3, 4]:
        if len(words) < n:
            metrics[f'repetition_{n}'] = 0.0
            continue
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        unique_ngrams = set(ngrams)
        
        if ngrams:
            repetition_rate = 1.0 - (len(unique_ngrams) / len(ngrams))
            metrics[f'repetition_{n}'] = repetition_rate
        else:
            metrics[f'repetition_{n}'] = 0.0
    
    return metrics


def evaluate_generation_quality(
    model,
    tokenizer,
    prompts: List[str],
    reference_texts: Optional[List[str]] = None,
    num_samples: int = 5,
    max_length: int = 100,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of text generation quality.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts to generate from
        reference_texts: Optional reference texts for BLEU calculation
        num_samples: Number of samples to generate per prompt
        max_length: Maximum generation length
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    all_generated_texts = []
    
    # Generate texts
    with torch.no_grad():
        for prompt in prompts:
            for _ in range(num_samples):
                # Encode prompt
                input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
                
                # Generate
                generated = model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode
                generated_text = tokenizer.decode(
                    generated[0][input_ids.shape[1]:].tolist(), 
                    skip_special_tokens=True
                )
                all_generated_texts.append(generated_text)
    
    # Calculate metrics
    metrics = {}
    
    # Diversity metrics
    diversity_metrics = calculate_diversity_metrics(all_generated_texts)
    metrics.update(diversity_metrics)
    
    # Average repetition metrics
    repetition_scores = [calculate_repetition_metrics(text) for text in all_generated_texts]
    for key in repetition_scores[0].keys():
        metrics[f'avg_{key}'] = np.mean([scores[key] for scores in repetition_scores])
    
    # BLEU scores if references provided
    if reference_texts and len(reference_texts) == len(all_generated_texts):
        bleu_scores = calculate_bleu_score(reference_texts, all_generated_texts)
        metrics.update(bleu_scores)
    
    # Average text length
    avg_length = np.mean([len(text.split()) for text in all_generated_texts])
    metrics['avg_length'] = avg_length
    
    return metrics


def benchmark_model(
    model,
    tokenizer, 
    test_dataset,
    device: torch.device,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Comprehensive model benchmarking.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        test_dataset: Test dataset
        device: Device to run on
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    # Perplexity
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(model, test_dataset, device, batch_size)
    results['perplexity'] = perplexity
    
    # Generation quality (sample prompts)
    sample_prompts = [
        "The weather today is",
        "In the future, technology will",
        "Once upon a time there was",
        "The most important thing in life is"
    ]
    
    print("Evaluating generation quality...")
    generation_metrics = evaluate_generation_quality(
        model, tokenizer, sample_prompts, device=device
    )
    results.update(generation_metrics)
    
    # Model size metrics
    results['num_parameters'] = model.get_num_params()
    results['vocab_size'] = tokenizer.get_vocab_size()
    
    return results
