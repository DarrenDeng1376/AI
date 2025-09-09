"""
Text generation and sampling utilities
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Callable
import math
import numpy as np


def generate_text(
    model,
    tokenizer,
    prompt: str = "",
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    do_sample: bool = True,
    num_return_sequences: int = 1,
    stop_tokens: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> Union[str, List[str]]:
    """
    Generate text using the model.
    
    Args:
        model: The language model
        tokenizer: Tokenizer instance
        prompt: Input prompt
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Repetition penalty factor
        length_penalty: Length penalty factor
        do_sample: Whether to use sampling or greedy decoding
        num_return_sequences: Number of sequences to return
        stop_tokens: List of stop tokens
        device: Device to run on
        
    Returns:
        Generated text(s)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Encode prompt
    if prompt:
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    else:
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
    
    # Generate multiple sequences if requested
    if num_return_sequences > 1:
        input_ids = input_ids.repeat(num_return_sequences, 1)
    
    generated_sequences = []
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated sequences
    for i in range(num_return_sequences):
        sequence = generated[i]
        
        # Remove prompt from generated sequence
        if prompt:
            sequence = sequence[input_ids.shape[1]:]
        
        # Decode to text
        text = tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        
        # Apply stop tokens
        if stop_tokens:
            for stop_token in stop_tokens:
                if stop_token in text:
                    text = text[:text.index(stop_token)]
                    break
        
        generated_sequences.append(text)
    
    if num_return_sequences == 1:
        return generated_sequences[0]
    return generated_sequences


def top_k_sampling(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply top-k sampling to logits.
    
    Args:
        logits: Logits tensor of shape (..., vocab_size)
        k: Number of top tokens to keep
        
    Returns:
        Filtered logits
    """
    if k <= 0:
        return logits
    
    # Get top k values and indices
    top_k_values, top_k_indices = torch.topk(logits, min(k, logits.size(-1)))
    
    # Create mask for top-k values
    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(-1, top_k_indices, top_k_values)
    
    return mask


def top_p_sampling(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to logits.
    
    Args:
        logits: Logits tensor of shape (..., vocab_size)
        p: Cumulative probability threshold
        
    Returns:
        Filtered logits
    """
    if p >= 1.0:
        return logits
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for tokens to remove
    sorted_indices_to_remove = cumulative_probs > p
    
    # Ensure we keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    
    # Set logits to -inf for removed tokens
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float('-inf')
    
    return filtered_logits


def apply_repetition_penalty(
    logits: torch.Tensor, 
    input_ids: torch.Tensor, 
    penalty: float = 1.0
) -> torch.Tensor:
    """
    Apply repetition penalty to discourage repetitive text.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        input_ids: Previous token IDs of shape (batch_size, seq_len)
        penalty: Repetition penalty factor (> 1.0 discourages repetition)
        
    Returns:
        Modified logits
    """
    if penalty == 1.0:
        return logits
    
    # Create penalty mask for tokens that already appear in the sequence
    batch_size, vocab_size = logits.shape
    penalty_mask = torch.ones_like(logits)
    
    for i in range(batch_size):
        for token_id in input_ids[i]:
            if 0 <= token_id < vocab_size:
                if logits[i, token_id] > 0:
                    penalty_mask[i, token_id] = 1 / penalty
                else:
                    penalty_mask[i, token_id] = penalty
    
    return logits * penalty_mask


def apply_length_penalty(scores: torch.Tensor, length: int, penalty: float = 1.0) -> torch.Tensor:
    """
    Apply length penalty to scores.
    
    Args:
        scores: Sequence scores
        length: Sequence length
        penalty: Length penalty factor
        
    Returns:
        Penalized scores
    """
    if penalty == 1.0:
        return scores
    
    length_penalty = ((5 + length) / 6) ** penalty
    return scores / length_penalty


def beam_search(
    model,
    input_ids: torch.Tensor,
    beam_size: int = 5,
    max_length: int = 100,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    eos_token_id: Optional[int] = None,
    pad_token_id: int = 0
) -> List[torch.Tensor]:
    """
    Beam search decoding.
    
    Args:
        model: Language model
        input_ids: Input token IDs
        beam_size: Number of beams
        max_length: Maximum sequence length
        length_penalty: Length penalty factor
        early_stopping: Whether to stop early when all beams have EOS
        eos_token_id: End-of-sequence token ID
        pad_token_id: Padding token ID
        
    Returns:
        List of generated sequences
    """
    device = input_ids.device
    batch_size, input_length = input_ids.shape
    
    # Initialize beams
    beam_scores = torch.zeros(batch_size, beam_size, device=device)
    beam_tokens = input_ids.unsqueeze(1).repeat(1, beam_size, 1)
    beam_indices = torch.arange(beam_size, device=device).unsqueeze(0).repeat(batch_size, 1)
    
    # Track finished sequences
    done = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)
    
    for step in range(max_length - input_length):
        # Get model outputs for all beams
        current_length = beam_tokens.shape[-1]
        
        # Reshape for model forward pass
        model_input = beam_tokens.view(-1, current_length)
        
        with torch.no_grad():
            outputs = model(model_input)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            next_token_logits = logits[:, -1, :]  # (batch_size * beam_size, vocab_size)
        
        # Reshape back
        next_token_logits = next_token_logits.view(batch_size, beam_size, -1)
        
        # Convert to log probabilities
        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        # Add beam scores
        scores = beam_scores.unsqueeze(-1) + next_token_log_probs
        
        # Reshape for top-k selection
        scores = scores.view(batch_size, -1)
        
        # Get top 2*beam_size candidates
        top_scores, top_indices = torch.topk(scores, 2 * beam_size, dim=-1)
        
        # Convert flat indices back to beam and token indices
        beam_idx = top_indices // next_token_logits.shape[-1]
        token_idx = top_indices % next_token_logits.shape[-1]
        
        # Select best beams
        new_beam_scores = []
        new_beam_tokens = []
        new_done = []
        
        for batch_idx in range(batch_size):
            batch_beam_scores = []
            batch_beam_tokens = []
            batch_done = []
            
            for i in range(beam_size):
                score = top_scores[batch_idx, i]
                beam_id = beam_idx[batch_idx, i]
                token_id = token_idx[batch_idx, i]
                
                # Get previous tokens for this beam
                prev_tokens = beam_tokens[batch_idx, beam_id]
                
                # Add new token
                new_tokens = torch.cat([prev_tokens, token_id.unsqueeze(0)])
                
                # Check if sequence is finished
                is_finished = (eos_token_id is not None and token_id == eos_token_id) or done[batch_idx, beam_id]
                
                batch_beam_scores.append(score)
                batch_beam_tokens.append(new_tokens)
                batch_done.append(is_finished)
            
            new_beam_scores.append(torch.stack(batch_beam_scores))
            new_beam_tokens.append(torch.stack(batch_beam_tokens))
            new_done.append(torch.stack(batch_done))
        
        beam_scores = torch.stack(new_beam_scores)
        beam_tokens = torch.stack(new_beam_tokens)
        done = torch.stack(new_done)
        
        # Early stopping if all beams are finished
        if early_stopping and done.all():
            break
    
    # Apply length penalty and return best sequences
    final_scores = apply_length_penalty(beam_scores, beam_tokens.shape[-1], length_penalty)
    
    best_sequences = []
    for batch_idx in range(batch_size):
        best_beam_idx = torch.argmax(final_scores[batch_idx])
        best_sequences.append(beam_tokens[batch_idx, best_beam_idx])
    
    return best_sequences


def contrastive_search(
    model,
    input_ids: torch.Tensor,
    max_length: int = 100,
    top_k: int = 5,
    alpha: float = 0.6,
    eos_token_id: Optional[int] = None
) -> torch.Tensor:
    """
    Contrastive search decoding for more coherent and diverse text generation.
    
    Args:
        model: Language model
        input_ids: Input token IDs
        max_length: Maximum sequence length
        top_k: Number of top candidates to consider
        alpha: Balance between model confidence and contrastive objective
        eos_token_id: End-of-sequence token ID
        
    Returns:
        Generated sequence
    """
    device = input_ids.device
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model outputs
            outputs = model(generated)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            next_token_logits = logits[0, -1, :]  # Assume batch size 1
            
            # Get top-k candidates
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            
            # Compute contrastive scores
            if generated.shape[1] > 1:
                # Get hidden states for context
                context_hidden = model(generated[:, :-1])
                if isinstance(context_hidden, dict):
                    context_hidden = context_hidden['hidden_states']
                else:
                    # If model doesn't return hidden states, use logits as proxy
                    context_hidden = logits[:, :-1, :]
                
                # Compute similarity scores for each candidate
                contrastive_scores = torch.zeros(top_k, device=device)
                
                for i, candidate_id in enumerate(top_k_indices):
                    # Create candidate sequence
                    candidate_seq = torch.cat([generated, candidate_id.unsqueeze(0).unsqueeze(0)], dim=1)
                    
                    # Get hidden state for candidate
                    candidate_outputs = model(candidate_seq)
                    if isinstance(candidate_outputs, dict):
                        candidate_hidden = candidate_outputs['hidden_states']
                    else:
                        candidate_hidden = candidate_outputs
                    
                    candidate_hidden = candidate_hidden[0, -1, :]  # Last hidden state
                    
                    # Compute similarity with context
                    similarities = F.cosine_similarity(
                        candidate_hidden.unsqueeze(0), 
                        context_hidden[0], 
                        dim=-1
                    )
                    contrastive_scores[i] = -similarities.max()  # Negative for diversity
            else:
                contrastive_scores = torch.zeros(top_k, device=device)
            
            # Combine model confidence and contrastive objective
            final_scores = alpha * torch.log(top_k_probs) + (1 - alpha) * contrastive_scores
            
            # Select best candidate
            best_idx = torch.argmax(final_scores)
            next_token = top_k_indices[best_idx]
            
            # Add to sequence
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and next_token == eos_token_id:
                break
    
    return generated[0]  # Remove batch dimension
