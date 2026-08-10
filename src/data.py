import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class Vocabulary:
    """
    Builds and manages a word-level vocabulary for the AI Model.
    
    Attributes:
        word2idx (dict): Maps word strings to integer IDs.
        idx2word (dict): Reverse maps from IDs to strings.
    """
    
    def __init__(self) -> None:
        """
            Initializes the vocabulary with special tokens.
            
            Args: 
                None
            
            Returns: 
                None
        """
        self.word2idx = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        self.idx2word = {value: key for key,value in self.word2idx.items()}
        self.idx = 3
    
    def build(self, texts: list[str]) -> None:
        """
            Build the vocabulary from a list of sentences.
            
            Args:
                texts (list[str]): A list of string senetences to tokenize and index.
                
            Returns:
                None
        """
        for text in texts:
            for word in text.split():
                if word not in self.word2idx.keys():
                    self.idx += 1
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                
    def encode(self, text: str) -> list[int]:
        """
            Encodes a text string to a list of token IDs.
            
            Args:
                text (str): A string object.
            
            Returns:
                list[int]: A list of integers.
        """
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in text.split()]
            
    def decode(self, nums: list[int]) -> str:
        """
            Decodes a list of integers to a text string.
            
            Args:
                nums (list[int]): A list of integers.
                
            Returns:
                str: A text string.
        """
        words = []
        for num in nums:
            word = self.idx2word.get(num, "<unk>")
            if word in ('<pad>', '<sos>', '<eos>'):
                continue
            words.append(word)
        return " ".join(words)
        
     
class Tokenizer:
    def __init__(self, model_path: str) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_id = self.sp.pad_id()    # 0
        self.sos_id = self.sp.bos_id()    # 2
        self.eos_id = self.sp.eos_id()    # 3
        self.unk_id = self.sp.unk_id()    # 1
    
    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)
    
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()   
    
    
class TranslationDataset(Dataset):
    """
    Pytorch dataset for the AI Translator Model.
    """
    
    def __init__(self, pairs: list[tuple[str,str]], tokenizer: Tokenizer, max_len: int=50) -> None:
        """
        Initializes the custom TranslationDataset class using torch Dataset class.
        
        Args:
            pairs (list[tuple[str,str]]): List of (english_sentences, german_sentences) tuples.
            tokenizer (Tokenizer): A instance of Tokenizer class used to tokenize text.
            max_len (int): Maximum length of sentence, and defaults to 50 unless specified.
            
        Returns:
            None
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        """
        Returns the total number of sentence pairs in the dataset.
        
        Returns:
            int: Number of translation pairs.
        """
        return len(self.pairs)
    
    def __getitem__(self, index: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Retrieve and process a single translation example.
        
        Args:
            index (int): Starting index of the translation pairs.
            
        Returns:
            tuple[torch.LongTensor, torch.LongTensor]
        """
        src_sentence, tgt_sentence = self.pairs[index]
        src_ids = self.tokenizer.encode(src_sentence)
        tgt_ids = [self.tokenizer.sos_id] + self.tokenizer.encode(tgt_sentence) + [self.tokenizer.eos_id]
        if len(src_ids) < self.max_len:
            src_ids.extend([self.tokenizer.pad_id] * (self.max_len - len(src_ids)))
        else:
            src_ids = src_ids[:self.max_len]
        if len(tgt_ids) < self.max_len:
            tgt_ids.extend([self.tokenizer.pad_id] * (self.max_len - len(tgt_ids)))
        else:
            tgt_ids = tgt_ids[:self.max_len]
            
        return (torch.LongTensor(src_ids), torch.LongTensor(tgt_ids))
        