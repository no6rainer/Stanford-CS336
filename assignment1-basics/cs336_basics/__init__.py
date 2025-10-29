import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .train_bpe import train_bpe
from . import pretokenization_example