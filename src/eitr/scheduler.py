"""Exact coverage marginals and full-render cost; no sufficiency claims."""
from collections import defaultdict
from time import perf_counter
from eicr.contracts import digest, require

ARMS = ("P0", "P1", "P2", "P3", "P4", "P4_no_support", "P4_no_coverage", "P4_active", "strict_seed")


def coverage_gain(current, candidate):
    require(len(current) == len(candidate), "Coverage dimension mismatch")
    return sum(max(0., a-c) for c, a in zip(current, candidate)) / len(current) if current else 0.


def build_interaction_matrix(index, query, top_edges=3):
    """Build a bounded cue-group interaction matrix from existing tag edges.

    The matrix is a retrieval preference signal. It does not assert that two
    cue groups describe the same event or that an edge is evidence of truth.
    """
    require(type(top_edges) is int and top_edges >= 1, "Invalid interaction edge cap")
    tag_sets = [set(group["seed_tags"]) for group in query["groups"]]
    seeds = set().union(*tag_sets)
    # Read only admitted seed rows. Two reciprocal edges describe one tag pair.
    edges = {(tag, edge["target"]): edge["score"] for tag in seeds
             for edge in index.neighbors[tag] if edge["target"] in seeds}
    matrix = [[0. for _ in tag_sets] for _ in tag_sets]
    for i, left in enumerate(tag_sets):
        for j in range(i):
            right = tag_sets[j]
            # Shared seeds cannot independently corroborate either group.
            values = sorted((max(edges.get((a, b), 0.), edges.get((b, a), 0.))
                             for a in left - right for b in right - left), reverse=True)
            top = [value for value in values[:top_edges] if value > 0]
            if top:
                matrix[i][j] = matrix[j][i] = sum(top) / len(top)
    for i, row in enumerate(matrix):
        scale = max(1., sum(row))  # Preserve weak isolated associations.
        matrix[i] = [value / scale for value in row]
    return matrix


class Scheduler:
    def __init__(self, corpus, pool, q_scores, cue_scores, config, interaction_matrix=None):
        require(pool["corpus_hash"] == corpus.identity, "Pool/corpus mismatch")
        require(pool["config_hash"] == digest(config.to_dict()), "Pool/config mismatch")
        require(pool["pool_hash"] == digest({k: v for k, v in pool.items() if k != "pool_hash"}), "Pool hash mismatch")
        self.corpus, self.pool, self.config = corpus, pool, config
        self.actions = pool["actions"]
        self.q, self.cues = q_scores, cue_scores
        self.interaction = interaction_matrix or []
        require(not self.interaction or len(self.interaction) == len(self.cues),
                "Interaction dimension mismatch")
        for j, row in enumerate(self.interaction):
            require(len(row) == len(self.cues) and row[j] == 0 and sum(row) <= 1 + 1e-12 and all(type(v) in (float, int) and 0 <= v <= 1 for v in row),
                    "Invalid cue interaction matrix")
        for values in [q_scores] + cue_scores:
            require(set(pool["atom_ids"]) <= set(values) and all(
                type(values[a]) in (float, int) and 0 <= values[a] <= 1 for a in pool["atom_ids"]), "Invalid relevance")
        for b in self.actions:
            require(b["core"] in corpus.atoms and b["action_id"] == b["core"] and
                    set(b["atoms"]) == set(corpus.contexts[b["core"]]), "Incomplete or changed action context")
        self.selected = set()
        self.packet = corpus.render(self.selected)
        self.C = [0.] * len(cue_scores)
        self.trace, self.evaluations, self.successful = [], 0, 0
        self.stop = "exhausted"
        self.phase_charges = defaultdict(int)
        self.unaffordable_evaluations = 0
        self.alternatives, self.phase = [], "initial"
        self.render_seconds, self.render_calls = 0., 0
        self.active_interaction = False
        self.influence = [0.] * len(cue_scores)

    def evaluate(self, action, eta, support, remaining=None):
        if self.evaluations >= self.config.max_score_evaluations:
            self.stop = "score_evaluation_cap"; return None
        fresh = set(action["atoms"]) - self.selected
        if not fresh: return None
        self.evaluations += 1
        tick = perf_counter()
        proposal = self.corpus.render(self.selected | fresh)
        self.render_seconds += perf_counter()-tick
        self.render_calls += 1
        cost = proposal["used_tokens"] - self.packet["used_tokens"]
        event = {"step": self.successful, "phase": self.phase, "action_id": action["action_id"],
                 "fresh_atoms": sorted(fresh), "cost": cost, "proposal_tokens": proposal["used_tokens"],
                 "phase_remaining": remaining}
        self.alternatives.append(event)
        if proposal["used_tokens"] > self.config.evidence_budget or (remaining is not None and max(0, cost) > remaining):
            self.unaffordable_evaluations += 1
            event["status"] = "over_packet_budget" if proposal["used_tokens"] > self.config.evidence_budget else "over_phase_quota"
            return None
        rel = sum(self.corpus.atoms[a]["text_tokens"] * self.q[a] for a in sorted(fresh))
        candidate = [max(values[a] for a in fresh) for values in self.cues]
        cov = coverage_gain(self.C, candidate)
        interaction = 0.
        if self.active_interaction and self.phase == "shared" and candidate:
            interaction = min(cov, sum(max(0., a-c) * h for a, c, h in
                                       zip(candidate, self.C, self.influence)) / len(candidate))
        factor = self.config.support_floor + (1-self.config.support_floor) * action["support"] if support else 1.
        committee_gain = (cov + self.config.interaction_weight * interaction
                          if self.active_interaction else cov)
        numerator = eta * rel + (1-eta) * self.config.reference_tokens * committee_gain
        priority = factor * numerator / max(1, cost)
        event.update(status="scored", coverage_gain=cov, rel_gain=rel / self.config.reference_tokens,
                     interaction_gain=interaction, committee_gain=committee_gain,
                     priority=priority, support_factor=factor,
                     interaction_enabled=self.active_interaction, coverage_before=list(self.C))
        return {"action": action, "proposal": proposal, "fresh": fresh,
                "cost": cost, "rel": rel / self.config.reference_tokens, "cov": cov,
                "interaction": interaction, "committee_gain": committee_gain,
                "rel_density": rel / max(1, cost),
                "cov_density": self.config.reference_tokens * cov / max(1, cost),
                "interaction_density": self.config.reference_tokens * interaction / max(1, cost),
                "support_factor": factor, "priority": priority, "numerator": numerator}

    def commit(self, choice, phase):
        require(self.successful < self.config.max_actions, "Action cap exceeded")
        previous = list(self.C)
        self.selected.update(choice["fresh"])
        self.packet = choice["proposal"]
        for j, values in enumerate(self.cues):
            self.C[j] = max(self.C[j], max(values[a] for a in choice["fresh"]))
        if self.active_interaction:
            self.influence = [sum(w * c for w, c in zip(row, self.C)) for row in self.interaction]
        self.successful += 1
        self.phase_charges[phase] += max(0, choice["cost"])
        self.trace.append({"action_id": choice["action"]["action_id"], "phase": phase,
                           "fresh_atoms": sorted(choice["fresh"]), "cost": choice["cost"],
                           "packet_tokens": self.packet["used_tokens"], "coverage_before": previous,
                           "coverage_after": list(self.C), "rel_gain": choice["rel"], "coverage_gain": choice["cov"],
                           "interaction_gain": choice.get("interaction", 0.),
                           "committee_gain": choice.get("committee_gain", choice["cov"]),
                           "rel_density": choice["rel_density"], "coverage_density": choice["cov_density"],
                           "interaction_density": choice.get("interaction_density", 0.),
                           "support_factor": choice["support_factor"], "priority": choice["priority"]})

    def best(self, actions, eta, support, remaining=None, allow_zero=False):
        winner = None
        for action in actions:
            if self.evaluations >= self.config.max_score_evaluations: break
            choice = self.evaluate(action, eta, support, remaining)
            if choice is None or choice["priority"] < 0 or (choice["priority"] == 0 and not allow_zero): continue
            key = (-choice["priority"], -choice["numerator"], choice["cost"], action["action_id"])
            if winner is None or key < winner[0]: winner = (key, choice)
        return winner[1] if winner else None

    def can_continue(self):
        if self.successful >= self.config.max_actions:
            self.stop = "action_cap"; return False
        if self.evaluations >= self.config.max_score_evaluations:
            self.stop = "score_evaluation_cap"; return False
        return True

    def greedy(self, actions, phase, eta=1., support=False, cap=None):
        self.phase = phase
        while self.can_continue():
            remaining = None if cap is None else max(0, cap-self.phase_charges[phase])
            choice = self.best(actions, eta, support, remaining)
            if choice is None:
                if self.stop not in ("action_cap", "score_evaluation_cap"):
                    self.stop = "no_positive_feasible_action" if any(set(a["atoms"])-self.selected for a in actions) else "exhausted"
                break
            self.commit(choice, phase)

    def seed_round_robin(self, cap):
        self.phase = "seed"
        gids = sorted({g for a in self.actions for g in a["seed_groups"]})
        while self.can_continue():
            changed = False
            for gid in gids:
                if not self.can_continue(): break
                choices = [a for a in self.actions if gid in a["seed_groups"]]
                choice = self.best(choices, 1., False, max(0, cap-self.phase_charges["seed"]), allow_zero=True)
                if choice:
                    self.commit(choice, "seed"); changed = True
            if not changed: break

    def static(self):
        tags = self.pool["tag_order"]
        remaining = max(0, self.config.evidence_budget-self.packet["used_tokens"])
        per_tag = remaining // max(1, len(tags))
        for t in tags:
            self.greedy([a for a in self.actions if t in a["tags"]], "static:" + t, cap=per_tag)
        # Equal allocation may leave unused quota. Deterministic tag-order refill;
        # no state-feedback priority beyond source deduplication and real cost.
        for t in tags:
            self.greedy([a for a in self.actions if t in a["tags"]], "static_refill:" + t)
        self.greedy([a for a in self.actions if a["direct"]], "direct_refill")

    def select(self, arm, has_seeds):
        require(arm in ARMS, "Unknown scheduler arm")
        self.active_interaction = arm == "P4_active"
        require(not self.active_interaction or len(self.interaction) == len(self.cues), "Missing cue interaction matrix")
        if arm == "P0" or (not has_seeds and arm != "strict_seed"):
            self.greedy(self.actions, "direct_full")
        elif arm == "strict_seed":
            self.seed_round_robin(self.config.evidence_budget)
        else:
            self.greedy([a for a in self.actions if a["direct"]], "direct",
                        cap=int(self.config.evidence_budget*self.config.direct_fraction))
            self.seed_round_robin(int(self.config.evidence_budget*self.config.seed_fraction))
            if arm == "P2":
                self.static()
            else:
                eta = self.config.eta if arm in ("P4", "P4_no_support", "P4_active") and self.cues else 1.
                self.greedy(self.actions, "shared", eta=eta, support=arm in ("P4", "P4_no_coverage"))
        final = self.corpus.render(self.selected)
        require(final == self.packet and final["used_tokens"] <= self.config.evidence_budget, "Final packet contract failed")
        return {**final, "budget_tokens": self.config.evidence_budget,
                "unused_tokens": self.config.evidence_budget-final["used_tokens"], "coverage": self.C,
                "ledger": self.trace, "phase_charges": dict(self.phase_charges),
                "alternatives": self.alternatives, "render_seconds": self.render_seconds,
                "render_calls": self.render_calls,
                "action_evaluations": self.evaluations, "successful_actions": self.successful,
                "unaffordable_evaluations": self.unaffordable_evaluations, "stop": self.stop,
                "interaction_matrix": self.interaction if self.active_interaction else [],
                "interaction_weight": self.config.interaction_weight if self.active_interaction else 0.,
                "pool_hash": self.pool["pool_hash"], "arm": arm,
                "fallback": not has_seeds and arm not in ("P0", "strict_seed")}
