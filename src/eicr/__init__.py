"""Evidence-Induced Cue Representation. Scores are associations, not probabilities."""

from .config import IndexConfig
from .index import Index, build_index, load_index, save_index

__version__ = "0.1.0"
__all__ = ["IndexConfig", "Index", "build_index", "load_index", "save_index"]
