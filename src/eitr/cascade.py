"""Semantic relevance with optional conservative, cost-aware MMR selection.

This is an MMR adaptation (Carbonell & Goldstein, SIGIR 1998), not a new
committee objective. All actions retain their complete context and exact cost.
"""
from collections import Counter
import math
import numpy as np
from scipy import sparse
from eicr.contracts import tokens
from .scheduler import Scheduler

CASCADE_ARMS = ('C_rerank', 'C_mmr', 'C_ppr_rerank', 'C_ppr_mmr')


class MMRScheduler(Scheduler):
    def __init__(self, corpus, pool, q_scores, cue_scores, config, **kwargs):
        super().__init__(corpus, pool, q_scores, cue_scores, config, **kwargs)
        self.ids = sorted(pool['atom_ids'])
        self.pos = {a: i for i, a in enumerate(self.ids)}
        counts = [Counter(tokens(corpus.atoms[a]['text'])) for a in self.ids]
        df = Counter(t for c in counts for t in c)
        vocab = {t: i for i, t in enumerate(sorted(df))}
        rows, cols, values = [], [], []
        for i, count in enumerate(counts):
            for t, tf in sorted(count.items()):
                rows.append(i); cols.append(vocab[t])
                values.append((1 + math.log(tf)) * (1 + math.log((1 + len(counts)) / (1 + df[t]))))
        x = sparse.csr_matrix((values, (rows, cols)), shape=(len(counts), len(vocab)))
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        x = sparse.diags(1 / np.maximum(norms, 1e-12)) @ x
        self.similarity = np.clip((x @ x.T).toarray(), 0., 1.)
        # Similar claims by different people or on different dates can be
        # complementary evidence. Unknown attribution never establishes duplication.
        def scope(a):
            src = corpus.sources[corpus.atoms[a]['source_id']]
            return src.get('speaker'), src.get('session_id'), src.get('session_datetime')
        self.scopes = {a: scope(a) for a in self.ids}

    def redundancy(self, aid):
        speaker, session, date = self.scopes[aid]
        if not speaker or speaker == 'unknown' or not session:
            return 0.
        values = [self.similarity[self.pos[aid], self.pos[b]] for b in sorted(self.selected)
                  if self.scopes[b] == (speaker, session, date)]
        value = max(values, default=0.)
        return float(value) if value >= self.config.mmr_min_similarity else 0.

    def evaluate(self, action, eta, support, remaining=None):
        choice = super().evaluate(action, eta, support, remaining)
        if choice is None:
            return None
        lam = self.config.mmr_lambda
        redundancy = {a: self.redundancy(a) for a in sorted(choice['fresh'])}
        numerator = sum(self.corpus.atoms[a]['text_tokens'] *
                        (lam * self.q[a] - (1-lam) * redundancy[a]) for a in sorted(choice['fresh']))
        choice.update(numerator=numerator, priority=numerator / max(1, choice['cost']), redundancy=redundancy)
        self.alternatives[-1].update(priority=choice['priority'], mmr_numerator=numerator,
                                    redundancy=redundancy, mmr_lambda=lam)
        return choice

    def commit(self, choice, phase):
        super().commit(choice, phase)
        self.trace[-1].update(redundancy=choice['redundancy'], mmr_lambda=self.config.mmr_lambda)


def select_cascade(corpus, pool, scores, config, arm, has_seeds):
    cls = MMRScheduler if arm.endswith('mmr') else Scheduler
    packet = cls(corpus, pool, scores, [], config).select('P3', has_seeds)
    packet.update(arm=arm, selector='cost-aware-mmr' if cls is MMRScheduler else 'P3-semantic-relevance',
                  redundancy_contract='TF-IDF cosine, same known speaker/session/date only' if cls is MMRScheduler else None)
    return packet
