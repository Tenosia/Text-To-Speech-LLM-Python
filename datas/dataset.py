import os
import random
from typing import List, Tuple, Any

import json
import torch
from torch.utils.data import Dataset

from text import cleaned_text_to_sequence

def intersperse(lst: list, item: int) -> list:
    """
    Put a blank token between any two input tokens to improve pronunciation.
    
    See https://github.com/jaywalnut310/glow-tts/issues/43 for more details.
    
    Args:
        lst: List of tokens
        item: Blank token value to insert
        
    Returns:
        List with blank tokens interspersed
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result
    
class StableDataset(Dataset):
    """Dataset for StableTTS training."""
    
    def __init__(self, filelist_path: str, hop_length: int):
        """
        Initialize StableDataset.
        
        Args:
            filelist_path: Path to JSON filelist file
            hop_length: Hop length for audio processing (used for length calculation)
        """
        self.filelist_path = filelist_path     
        self.hop_length = hop_length  
        
        self._load_filelist(filelist_path)

    def _load_filelist(self, filelist_path: str) -> None:
        """Load filelist from JSON file."""
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"Filelist file not found: {filelist_path}")
        
        filelist, lengths = [], []
        try:
            with open(filelist_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if 'mel_path' not in data or 'phone' not in data or 'mel_length' not in data:
                            print(f'Warning: Skipping line {line_idx+1} (missing required fields)')
                            continue
                        filelist.append((data['mel_path'], data['phone']))
                        lengths.append(data['mel_length'])
                    except json.JSONDecodeError as e:
                        print(f'Warning: Skipping line {line_idx+1} (invalid JSON): {e}')
                        continue
        except Exception as e:
            raise RuntimeError(f"Error loading filelist from {filelist_path}: {e}")
        
        if len(filelist) == 0:
            raise ValueError(f"No valid entries found in filelist: {filelist_path}")
            
        self.filelist = filelist
        self.lengths = lengths  # length is used for DistributedBucketSampler
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.filelist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (mel_spectrogram, phone_sequence)
        """
        mel_path, phone = self.filelist[idx]
        
        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"Mel file not found: {mel_path}")
        
        try:
            mel = torch.load(mel_path, map_location='cpu', weights_only=True)
            phone = torch.tensor(intersperse(cleaned_text_to_sequence(phone), 0), dtype=torch.long)
        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx} from {mel_path}: {e}")
        
        return mel, phone
    
def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (mel, phone) tuples
        
    Returns:
        Tuple of padded tensors: (texts, text_lengths, mels, mel_lengths, mels_sliced, mels_sliced_lengths)
    """
    texts = [item[1] for item in batch]
    mels = [item[0] for item in batch]
    mels_sliced = [random_slice_tensor(mel) for mel in mels]
    
    text_lengths = torch.tensor([text.size(-1) for text in texts], dtype=torch.long)
    mel_lengths = torch.tensor([mel.size(-1) for mel in mels], dtype=torch.long)
    mels_sliced_lengths = torch.tensor([mel_sliced.size(-1) for mel_sliced in mels_sliced], dtype=torch.long)
    
    # pad to the same length
    texts_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(texts), padding=0)
    mels_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(mels), padding=0)
    mels_sliced_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(mels_sliced), padding=0)

    return texts_padded, text_lengths, mels_padded, mel_lengths, mels_sliced_padded, mels_sliced_lengths

def random_slice_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Randomly slice mel-spectrogram for reference encoder to prevent overfitting.
    
    Args:
        x: Mel-spectrogram tensor of shape (n_mels, time)
        
    Returns:
        Sliced mel-spectrogram tensor
    """
    length = x.size(-1)
    if length < 12:
        return x 
    segment_size = random.randint(length // 12, length // 3)
    start = random.randint(0, length - segment_size)
    return x[..., start : start + segment_size]