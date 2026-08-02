import torch


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
        