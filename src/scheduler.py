import torch
import torch.optim as optim
import math

class WarmupCosineScheduler():
    """
    Implementing a cosine learning rate scheduler.
    """
    
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, total_steps: int, min_lr: float= 0.0) -> None:
        """
        Initializing the learning rate scheduler class.
        """
        self.current_step = 0
        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]['lr']
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
    def step(self) -> None:
        """
        Update the learning rate.
        """
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            cosine_value = math.pi * ((self.current_step - self.warmup_steps)/(self.total_steps - self.warmup_steps))
            lr = self.min_lr + (((self.base_lr - self.min_lr) / 2) * (1 + math.cos(cosine_value)))
        for group in self.optimizer.param_groups:
            group['lr'] = lr
            
    def get_lr(self) -> float:
        """
        Returns the current learning rate.
        
        Returns:
            lr (float): the current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']