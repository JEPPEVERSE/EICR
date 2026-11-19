"""Exact coverage marginals and full-render cost; no sufficiency claims."""
from collections import defaultdict
from eicr.contracts import digest, require

ARMS = ("P0", "P1", "P2", "P3", "P4", "P4_no_support", "P4_no_coverage", "strict_seed")


def coverage_gain(current, candidate):
    require(len(current) == len(candidate), "Coverage dimension mismatch")
    return sum(max(0., a-c) for c, a in zip(current, candidate)) / len(current) if current else 0.


class Scheduler:
    def __init__(self, corpus, pool, q_scores, cue_scores, config):
        require(pool["corpus_hash"] == corpus.identity, "Pool/corpus mismatch")
        require(pool["config_hash"] == digest(config.to_dict()), "Pool/config mismatch")
        require(pool["pool_hash"] == digest({k: v for k, v in pool.items() if k != "pool_hash"}), "Pool hash mismatch")
        self.corpus, self.pool, self.config = corpus, pool, config
        self.actions = pool["actions"]
        self.q, self.cues = q_scores, cue_scores
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

    def evaluate(self, action, eta, support, remaining=None):
        if self.evaluations >= self.config.max_score_evaluations:
            self.stop = "score_evaluation_cap"; return None
        fresh = set(action["atoms"]) - self.selected
        if not fresh: return None
        self.evaluations += 1
        proposal = self.corpus.render(self.selected | fresh)
        cost = proposal["used_tokens"] - self.packet["used_tokens"]
        if proposal["used_tokens"] > self.config.evidence_budget or (remaining is not None and max(0, cost) > remaining):
            self.unaffordable_evaluations += 1
            return None
        rel = sum(self.corpus.atoms[a]["text_tokens"] * self.q[a] for a in fresh)
        candidate = [max(values[a] for a in fresh) for values in self.cues]
        cov = coverage_gain(self.C, candidate)
        factor = self.config.support_floor + (1-self.config.support_floor) * action["support"] if support else 1.
        numerator = eta * rel + (1-eta) * self.config.reference_tokens * cov
        priority = factor * numerator / max(1, cost)
        return {"action": action, "proposal": proposal, "fresh": fresh,
                "cost": cost, "rel": rel / self.config.reference_tokens, "cov": cov,
                "rel_density": rel / max(1, cost),
                "cov_density": self.config.reference_tokens * cov / max(1, cost),
                "support_factor": factor, "priority": priority, "numerator": numerator}

    def commit(self, choice, phase):
        require(self.successful < self.config.max_actions, "Action cap exceeded")
        previous = list(self.C)
        self.selected.update(choice["fresh"])
        self.packet = choice["proposal"]
        for j, values in enumerate(self.cues):
            self.C[j] = max(self.C[j], max(values[a] for a in choice["fresh"]))
        self.successful += 1
        self.phase_charges[phase] += max(0, choice["cost"])
        self.trace.append({"action_id": choice["action"]["action_id"], "phase": phase,
                           "fresh_atoms": sorted(choice["fresh"]), "cost": choice["cost"],
                           "packet_tokens": self.packet["used_tokens"], "coverage_before": previous,
                           "coverage_after": list(self.C), "rel_gain": choice["rel"], "coverage_gain": choice["cov"],
                           "rel_density": choice["rel_density"], "coverage_density": choice["cov_density"],
                           "support_factor": choice["support_factor"], "priority": choice["priority"]})

    def best(self, actions, eta, support, remaining=None, allow_zero=False):
        winner = None
        for action in actions:
            if self.evaluations >= self.config.max_score_evaluations: break
            choice = self.evaluate(action, eta, support, remaining)
            if choice is None or (choice["priority"] <= 0 and not allow_zero): continue
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
        while self.can_continue():
            remaining = None if cap is None else max(0, cap-self.phase_charges[phase])
            choice = self.best(actions, eta, support, remaining)
            if choice is None: break
            self.commit(choice, phase)

    def seed_round_robin(self, cap):
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
                eta = self.config.eta if arm in ("P4", "P4_no_support") and self.cues else 1.
                self.greedy(self.actions, "shared", eta=eta, support=arm in ("P4", "P4_no_coverage"))
        final = self.corpus.render(self.selected)
        require(final == self.packet and final["used_tokens"] <= self.config.evidence_budget, "Final packet contract failed")
        return {**final, "budget_tokens": self.config.evidence_budget,
                "unused_tokens": self.config.evidence_budget-final["used_tokens"], "coverage": self.C,
                "ledger": self.trace, "phase_charges": dict(self.phase_charges),
                "action_evaluations": self.evaluations, "successful_actions": self.successful,
                "unaffordable_evaluations": self.unaffordable_evaluations, "stop": self.stop,
                "pool_hash": self.pool["pool_hash"], "arm": arm,
                "fallback": not has_seeds and arm not in ("P0", "strict_seed")}
