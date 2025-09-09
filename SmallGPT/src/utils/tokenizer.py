"""
Simple tokenizer implementation with BPE (Byte-Pair Encoding)
"""
import re
import json
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict, Counter
from pathlib import Path
import pickle


class SimpleTokenizer:
    """
    A simple BPE tokenizer implementation.
    
    Args:
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for tokens to be included
        special_tokens: Dictionary of special tokens
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, str]] = None
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        if special_tokens is None:
            special_tokens = {
                'pad_token': '<pad>',
                'unk_token': '<unk>',
                'bos_token': '<bos>',
                'eos_token': '<eos>',
            }
        
        self.special_tokens = special_tokens
        self.pad_token = special_tokens['pad_token']
        self.unk_token = special_tokens['unk_token']
        self.bos_token = special_tokens['bos_token']
        self.eos_token = special_tokens['eos_token']
        
        # Token mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # BPE merges
        self.merges: List[Tuple[str, str]] = []
        
        # Token IDs for special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Initialize with special tokens
        self._add_special_tokens()
        
        # Regex pattern for tokenization (using standard character classes)
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_token_list = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        for i, token in enumerate(special_token_list):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    def _get_word_frequencies(self, texts: List[str]) -> Dict[str, int]:
        """Get word frequencies from texts."""
        word_freq = Counter()
        
        for text in texts:
            words = self._pre_tokenize(text)
            word_freq.update(words)
        
        return dict(word_freq)
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text into words."""
        # Simple whitespace + punctuation tokenization
        words = re.findall(self.pattern, text)
        return [word for word in words if word.strip()]
    
    def _get_char_vocab(self, word_freq: Dict[str, int]) -> Dict[str, int]:
        """Get character vocabulary from word frequencies."""
        char_freq = Counter()
        
        for word, freq in word_freq.items():
            for char in word:
                char_freq[char] += freq
        
        # Filter by minimum frequency
        char_vocab = {char: freq for char, freq in char_freq.items() 
                     if freq >= self.min_frequency}
        
        return char_vocab
    
    def _get_pairs(self, word_freq: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Get all adjacent character pairs and their frequencies."""
        pairs = defaultdict(int)
        
        for word, freq in word_freq.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        
        return dict(pairs)
    
    def _merge_vocab(self, pair: Tuple[str, str], word_freq: Dict[str, int]) -> Dict[str, int]:
        """Merge the most frequent pair in the vocabulary."""
        new_word_freq = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\\S)' + bigram + r'(?!\\S)')
        
        for word, freq in word_freq.items():
            new_word = p.sub(''.join(pair), ' '.join(word))
            new_word_freq[tuple(new_word.split())] = freq
        
        return new_word_freq
    
    def train(self, texts: List[str]) -> 'SimpleTokenizer':
        """
        Train the BPE tokenizer on the given texts.
        
        Args:
            texts: List of training texts
            
        Returns:
            Self for method chaining
        """
        print("Training tokenizer...")
        
        # Get word frequencies
        word_freq = self._get_word_frequencies(texts)
        
        # Convert words to character sequences
        word_freq = {tuple(word): freq for word, freq in word_freq.items()}
        
        # Get initial character vocabulary
        char_vocab = self._get_char_vocab(word_freq)
        
        # Initialize vocabulary with characters
        vocab = set(char_vocab.keys())
        
        # Add characters to token mappings
        current_id = len(self.token_to_id)
        for char in sorted(vocab):
            if char not in self.token_to_id:
                self.token_to_id[char] = current_id
                self.id_to_token[current_id] = char
                current_id += 1
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(self.token_to_id)
        
        for i in range(num_merges):
            pairs = self._get_pairs(word_freq)
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            word_freq = self._merge_vocab(best_pair, word_freq)
            
            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = current_id
                self.id_to_token[current_id] = merged_token
                current_id += 1
            
            # Store the merge
            self.merges.append(best_pair)
            
            if i % 1000 == 0:
                print(f"Merge {i}/{num_merges}: {best_pair}")
        
        print(f"Training complete. Vocabulary size: {len(self.token_to_id)}")
        return self
    
    def _apply_bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a word."""
        if len(word) <= 1:
            return [word]
        
        pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
        
        if not pairs:
            return [word]
        
        while True:
            # Find the earliest merge that applies
            bigram = None
            min_merge_idx = float('inf')
            
            for pair in pairs:
                if pair in self.merges:
                    merge_idx = self.merges.index(pair)
                    if merge_idx < min_merge_idx:
                        min_merge_idx = merge_idx
                        bigram = pair
            
            if bigram is None:
                break
            
            # Apply the merge
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = ''.join(new_word)
            
            if len(word) == 1:
                break
            
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
        
        # Split into subwords
        subwords = []
        current_word = ""
        
        for char in word:
            test_word = current_word + char
            if test_word in self.token_to_id:
                current_word = test_word
            else:
                if current_word:
                    subwords.append(current_word)
                current_word = char
        
        if current_word:
            subwords.append(current_word)
        
        return subwords
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        # Pre-tokenize
        words = self._pre_tokenize(text)
        
        # Apply BPE to each word
        tokens = []
        for word in words:
            subwords = self._apply_bpe(word)
            for subword in subwords:
                if subword in self.token_to_id:
                    tokens.append(self.token_to_id[subword])
                else:
                    tokens.append(self.unk_token_id)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens.values():
                    continue
                
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = re.sub(r'\s+', ' ', text)  # Clean up whitespace
        
        return text.strip()
    
    def save(self, path: Union[str, Path]):
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer_data = {
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'special_tokens': self.special_tokens,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'merges': self.merges,
            'pad_token_id': self.pad_token_id,
            'unk_token_id': self.unk_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id
        }
        
        with open(path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SimpleTokenizer':
        """Load tokenizer from file."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        tokenizer = cls(
            vocab_size=tokenizer_data['vocab_size'],
            min_frequency=tokenizer_data['min_frequency'],
            special_tokens=tokenizer_data['special_tokens']
        )
        
        tokenizer.token_to_id = tokenizer_data['token_to_id']
        tokenizer.id_to_token = tokenizer_data['id_to_token']
        tokenizer.merges = tokenizer_data['merges']
        tokenizer.pad_token_id = tokenizer_data['pad_token_id']
        tokenizer.unk_token_id = tokenizer_data['unk_token_id']
        tokenizer.bos_token_id = tokenizer_data['bos_token_id']
        tokenizer.eos_token_id = tokenizer_data['eos_token_id']
        
        print(f"Tokenizer loaded from {path}")
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_id)
    
    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_id)
