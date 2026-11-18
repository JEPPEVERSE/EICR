"""Executable parameters; the historical design JSON is documentation, not this schema."""

from dataclasses import asdict, dataclass

from .contracts import finite_unit, require


@dataclass(frozen=True)
class IndexConfig:
    structure: bool = False
    beta: float = 0.25
    lambda_: float = 0.0
    topic_min_df: int = 2
    topic_max_df: int = 64
    topic_max_fraction: float = 0.05
    max_topics: int = 32
    topic_strength: float = 0.25
    min_strength: float = 0.05
    neighbors: int = 8
    explicit_link_types: tuple = ()
    max_explicit_candidates: int = 64
    weight_mode: str = "source_family"

    def __post_init__(self):
        for name in ("beta", "lambda_"):
            finite_unit(getattr(self, name), name, closed=False)
        for name in ("topic_max_fraction", "topic_strength", "min_strength"):
            finite_unit(getattr(self, name), name)
        for name in ("topic_min_df", "topic_max_df", "max_topics", "neighbors", "max_explicit_candidates"):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0, name + " must be a positive integer")
        require(self.weight_mode in ("source_family", "uniform"), "Unknown weight mode")
        require(self.structure or self.lambda_ == 0, "A positive lambda requires structure=true")
        require(self.topic_min_df <= self.topic_max_df, "Invalid topic bounds")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if "explicit_link_types" in value:
            value["explicit_link_types"] = tuple(value["explicit_link_types"])
        return cls(**value)
