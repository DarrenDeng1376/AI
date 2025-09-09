"""
Dataset classes for text data
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Union
import os
import random
from pathlib import Path


class TextDataset(Dataset):
    """
    Dataset for text data with tokenization and sequence preparation.
    
    Args:
        texts: List of text strings or path to text file
        tokenizer: Tokenizer instance
        max_seq_len: Maximum sequence length
        stride: Stride for overlapping sequences (for long texts)
        return_attention_mask: Whether to return attention masks
    """
    
    def __init__(
        self,
        texts: Union[List[str], str, Path],
        tokenizer,
        max_seq_len: int = 1024,
        stride: Optional[int] = None,
        return_attention_mask: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len // 2
        self.return_attention_mask = return_attention_mask
        
        # Load texts
        if isinstance(texts, (str, Path)):
            self.texts = self._load_from_file(texts)
        else:
            self.texts = texts
        
        # Prepare sequences
        self.sequences = self._prepare_sequences()
    
    def _load_from_file(self, filepath: Union[str, Path]) -> List[str]:
        """Load texts from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines (paragraphs) or single newlines
        if '\n\n' in content:
            texts = content.split('\n\n')
        else:
            texts = content.split('\n')
        
        # Filter empty texts
        texts = [text.strip() for text in texts if text.strip()]
        
        return texts
    
    def _prepare_sequences(self) -> List[Dict[str, torch.Tensor]]:
        """Prepare tokenized sequences from texts."""
        sequences = []
        
        for text in self.texts:
            # Tokenize text
            tokens = self.tokenizer.encode(text)
            
            # Create overlapping sequences if text is longer than max_seq_len
            if len(tokens) <= self.max_seq_len:
                sequences.append(self._create_sequence(tokens))
            else:
                # Create overlapping chunks
                for i in range(0, len(tokens) - self.max_seq_len + 1, self.stride):
                    chunk = tokens[i:i + self.max_seq_len]
                    sequences.append(self._create_sequence(chunk))
        
        return sequences
    
    def _create_sequence(self, tokens: List[int]) -> Dict[str, torch.Tensor]:
        """Create a sequence dictionary from tokens."""
        # Pad to max_seq_len if necessary
        if len(tokens) < self.max_seq_len:
            padding_length = self.max_seq_len - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
        
        sequence = {
            'input_ids': torch.tensor(tokens, dtype=torch.long)
        }
        
        if self.return_attention_mask:
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in tokens]
            sequence['attention_mask'] = torch.tensor(attention_mask, dtype=torch.bool)
        
        return sequence
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.sequences[idx]


class LanguageModelingDataset(Dataset):
    """
    Dataset specifically designed for language modeling with causal masking.
    
    Args:
        texts: List of text strings or path to text file
        tokenizer: Tokenizer instance
        block_size: Size of each training block
        stride: Stride for creating blocks
    """
    
    def __init__(
        self,
        texts: Union[List[str], str, Path],
        tokenizer,
        block_size: int = 1024,
        stride: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or block_size
        
        # Load and tokenize all texts
        if isinstance(texts, (str, Path)):
            texts = self._load_from_file(texts)
        
        # Concatenate all texts with separator tokens
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                all_tokens.append(self.tokenizer.eos_token_id)
        
        # Create blocks
        self.blocks = []
        for i in range(0, len(all_tokens) - self.block_size + 1, self.stride):
            block = all_tokens[i:i + self.block_size]
            self.blocks.append(torch.tensor(block, dtype=torch.long))
    
    def _load_from_file(self, filepath: Union[str, Path]) -> List[str]:
        """Load texts from file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by paragraphs
        texts = [text.strip() for text in content.split('\n\n') if text.strip()]
        return texts
    
    def __len__(self) -> int:
        return len(self.blocks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {'input_ids': self.blocks[idx]}


class ConversationDataset(Dataset):
    """
    Dataset for conversational data (e.g., chatbot training).
    
    Args:
        conversations: List of conversation dictionaries
        tokenizer: Tokenizer instance
        max_seq_len: Maximum sequence length
        system_token: Token to mark system messages
        user_token: Token to mark user messages
        assistant_token: Token to mark assistant messages
    """
    
    def __init__(
        self,
        conversations: List[Dict],
        tokenizer,
        max_seq_len: int = 1024,
        system_token: str = "<|system|>",
        user_token: str = "<|user|>",
        assistant_token: str = "<|assistant|>"
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_token = system_token
        self.user_token = user_token
        self.assistant_token = assistant_token
        
        # Prepare conversation sequences
        self.sequences = self._prepare_conversations()
    
    def _prepare_conversations(self) -> List[Dict[str, torch.Tensor]]:
        """Prepare tokenized conversation sequences."""
        sequences = []
        
        for conversation in self.conversations:
            # Format conversation
            formatted_text = self._format_conversation(conversation)
            
            # Tokenize
            tokens = self.tokenizer.encode(formatted_text)
            
            # Truncate if too long
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            
            # Pad if too short
            if len(tokens) < self.max_seq_len:
                padding_length = self.max_seq_len - len(tokens)
                tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
            
            sequences.append({
                'input_ids': torch.tensor(tokens, dtype=torch.long)
            })
        
        return sequences
    
    def _format_conversation(self, conversation: Dict) -> str:
        """Format a conversation dictionary into a string."""
        formatted_parts = []
        
        # Add system message if present
        if 'system' in conversation:
            formatted_parts.append(f"{self.system_token} {conversation['system']}")
        
        # Add conversation turns
        for turn in conversation.get('messages', []):
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            
            if role == 'user':
                formatted_parts.append(f"{self.user_token} {content}")
            elif role == 'assistant':
                formatted_parts.append(f"{self.assistant_token} {content}")
        
        return '\n'.join(formatted_parts)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.sequences[idx]


class DocumentDataset(Dataset):
    """
    Dataset for processing multiple document files.
    
    Args:
        data_dir: Directory containing text files
        tokenizer: Tokenizer instance
        max_seq_len: Maximum sequence length
        file_extensions: List of file extensions to include
        shuffle_chunks: Whether to shuffle document chunks
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        tokenizer,
        max_seq_len: int = 1024,
        file_extensions: List[str] = ['.txt', '.md'],
        shuffle_chunks: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.file_extensions = file_extensions
        self.shuffle_chunks = shuffle_chunks
        
        # Find all text files
        self.file_paths = self._find_text_files()
        
        # Prepare chunks
        self.chunks = self._prepare_chunks()
        
        if self.shuffle_chunks:
            random.shuffle(self.chunks)
    
    def _find_text_files(self) -> List[Path]:
        """Find all text files in the data directory."""
        file_paths = []
        
        for ext in self.file_extensions:
            pattern = f"**/*{ext}"
            file_paths.extend(self.data_dir.glob(pattern))
        
        return file_paths
    
    def _prepare_chunks(self) -> List[Dict[str, torch.Tensor]]:
        """Prepare chunks from all documents."""
        chunks = []
        
        for file_path in self.file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Tokenize content
                tokens = self.tokenizer.encode(content)
                
                # Create chunks
                for i in range(0, len(tokens), self.max_seq_len):
                    chunk_tokens = tokens[i:i + self.max_seq_len]
                    
                    # Pad if necessary
                    if len(chunk_tokens) < self.max_seq_len:
                        padding_length = self.max_seq_len - len(chunk_tokens)
                        chunk_tokens = chunk_tokens + [self.tokenizer.pad_token_id] * padding_length
                    
                    chunks.append({
                        'input_ids': torch.tensor(chunk_tokens, dtype=torch.long),
                        'source_file': str(file_path)
                    })
            
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        return chunks
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.chunks[idx]
