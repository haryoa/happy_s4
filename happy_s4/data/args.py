"""
All Args for data goes here
"""
from dataclasses import dataclass
from happy_s4.mixin import MappingMixin


@dataclass
class BatchArgs(MappingMixin):
    """
    Batch Size Args. The formula is:
    ```
    batch_size = total_batch_size / grad_accumulation / num_gpus (args in Trainer)
    ```
    """
    total_batch_size: int = 64
    grad_accumulation: int = 1
