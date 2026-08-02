import torch
from torch.utils.data import Dataset

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
        
        
class TranslationDataset(Dataset):
    """
    Pytorch dataset for the AI Translator Model.
    """
    
    def __init__(self, pairs: list[tuple[str,str]], src_vocab: Vocabulary, tgt_vocab: Vocabulary, max_len: int=50) -> None:
        """
        Initializes the custom TranslationDataset class using torch Dataset class.
        
        Args:
            pairs (list[tuple[str,str]]): List of (english_sentences, german_sentences) tuples.
            src_vocab (Vocabulary): Vocabulary object of source language.
            tgt_vocab (Vocabulary): Vocabulary object of target language.
            max_len (int): Maximum length of sentence, and defaults to 50 unless specified.
            
        Returns:
            None
        """
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
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
        src_ids = self.src_vocab.encode(src_sentence)
        tgt_ids = [self.tgt_vocab.word2idx['<sos>']] + self.tgt_vocab.encode(tgt_sentence) + [self.tgt_vocab.word2idx['<eos>']]
        if len(src_ids) < self.max_len:
            src_ids.extend([self.src_vocab.word2idx.get("<pad>", 0)] * (self.max_len - len(src_ids)))
        else:
            src_ids = src_ids[:self.max_len]
        if len(tgt_ids) < self.max_len:
            tgt_ids.extend([self.tgt_vocab.word2idx.get("<pad>", 0)] * (self.max_len - len(tgt_ids)))
        else:
            tgt_ids = tgt_ids[:self.max_len]
            
        return (torch.LongTensor(src_ids), torch.LongTensor(tgt_ids))
        