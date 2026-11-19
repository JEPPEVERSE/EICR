"""Executable, strict configuration. Research design JSON is not executable."""
from dataclasses import asdict, dataclass, fields
import math
from eicr.contracts import require


@dataclass(frozen=True)
class Config:
    top_k: int = 16
    max_hops: int = 2
    path_decay: float = 0.8
    max_query_cues: int = 8
    max_keys_per_cue: int = 3
    max_tags_per_key: int = 4
    max_seed_tags: int = 32
    max_reached_tags: int = 64
    max_state_pops: int = 384
    max_edge_evaluations: int = 1024
    max_core_atoms_per_tag: int = 16
    max_core_atoms: int = 256
    max_candidate_atoms: int = 512
    direct_core_reservation: int = 64
    atom_tokens: int = 128
    context_each_side: int = 1
    reference_tokens: int = 128
    eta: float = 0.5
    support_floor: float = 0.1
    evidence_budget: int = 6000
    direct_fraction: float = 1 / 3
    seed_fraction: float = 1 / 6
    max_actions: int = 128
    max_score_evaluations: int = 100000
    explorer_rounds: int = 8
    explorer_actions_per_round: int = 4

    def __post_init__(self):
        fractional = {"path_decay", "eta", "support_floor", "direct_fraction", "seed_fraction"}
        nonnegative = {"max_hops", "context_each_side"}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in fractional:
                require(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1,
                        "Invalid fraction: " + f.name)
            else:
                require(type(v) is int and v >= (0 if f.name in nonnegative else 1),
                        "Invalid integer: " + f.name)
        require(self.support_floor > 0, "Support floor must protect isolated entries")
        require(self.direct_fraction + self.seed_fraction <= 1, "Reserve fractions exceed budget")
        require(self.max_seed_tags <= self.max_reached_tags, "Seed cap exceeds graph cap")
        require(self.direct_core_reservation <= self.max_core_atoms <= self.max_candidate_atoms,
                "Inconsistent candidate caps")
        require(self.max_hops <= 2, "v2 is bounded to two hops")
        require(self.context_each_side <= 1, "v2 context is at most one atom per side")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        require(isinstance(data, dict), "Config must be an object")
        require(set(data) <= {f.name for f in fields(cls)}, "Unknown executable config fields")
        return cls(**data)
