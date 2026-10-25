"""Adaptive adversaries against the adaptation loop of a drift-triggered, gated continual IDS.

The adversary knows the learner (DTL), its drift detector and the gate policy, controls part of
the labelling pipeline (it can flip revealed labels of chosen flows) and can inject traffic. It
does not know the validators' clean samples, only their distribution.

Building blocks, combined per attack in cfg["attack"]:
  trigger  forged benign flows (copies of Normal flows with an offset on a fixed feature subset),
           injected in on/off pulses so the hybrid detector raises an alarm: forces adaptation
           windows in which poisoned labels are learned (trigger hijacking)
  poison   label flipping of a target -> Normal inside given batch windows, optionally with
           injected replays of the attacker's own target flows labelled Normal; the target is one
           class ("A0": first class of group A, "B0": first of group B), all attacks ("all"), or
           only an attacker-chosen variant of the class (the top `variant_q` share of the class
           along a secret direction w): the variant is hidden from a validator that samples the
           class uniformly
Evaluation always uses true labels; forged flows are labelled Normal (they are benign traffic).
"""
import numpy as np


class Adversary:
    def __init__(self, acfg, scenario, seed):
        self.cfg, self.sc = acfg or {}, scenario
        self.rng = np.random.default_rng(seed + 4242)
        self.n_injected, self.n_flipped = 0, 0
        d = scenario.X_pre.shape[1]
        p = self.cfg.get("poison")
        self.target = None
        if p:
            t = p.get("target", "A0")
            if t == "all":
                self.target = "all"
            else:
                grp = scenario.groups[t[0]]
                self.target = scenario.cidx[grp[int(t[1:])]]
            self.q, self.w, self._thr = p.get("variant_q"), None, {}
            if self.q and self.target != "all":
                w = self.rng.normal(size=d)
                self.w = (w / np.linalg.norm(w)).astype(np.float32)
        tr = self.cfg.get("trigger")
        if tr:
            k = max(1, int(round(tr.get("feat_frac", 0.2) * d)))
            self.trig_feats = self.rng.choice(d, k, replace=False)

    @staticmethod
    def _in(b, windows):
        return any(lo <= b < hi for lo, hi in windows)

    def _pulse_on(self, b):
        tr = self.cfg["trigger"]
        for lo, hi in tr["windows"]:
            if lo <= b < hi:
                return (b - lo) % (tr["on"] + tr["off"]) < tr["on"]
        return False

    def thr(self, ph):
        """Variant boundary under phase ph's operating condition: the top `variant_q` share of the
        target class along w (the attacker knows the distribution of its own traffic)."""
        if ph not in self._thr:
            X, y = self.sc.test_set(self.sc.group_of[self.target], ph)
            self._thr[ph] = float(np.quantile(X[y == self.target] @ self.w, 1.0 - self.q))
        return self._thr[ph]

    def in_target(self, X, y, ph):
        """Mask of flows the attacker wants missed (target class, or its variant)."""
        if self.target is None:
            return np.zeros(len(y), bool)
        if self.target == "all":
            return y != self.sc.normal_idx
        m = y == self.target
        if self.w is not None:
            m &= (X @ self.w) >= self.thr(ph)
        return m

    def batch(self, b, X, y, ph):
        """Returns (X, y_true, y_revealed) for batch b of phase ph."""
        if self.cfg.get("trigger") and self._pulse_on(b):
            tr = self.cfg["trigger"]
            n = int(round(tr.get("frac", 0.15) * len(y)))
            pool = np.flatnonzero(y == self.sc.normal_idx)
            if len(pool) and n:
                src = self.rng.choice(pool, n, replace=len(pool) < n)
                F = X[src].copy()
                F[:, self.trig_feats] += tr.get("offset", 3.0)
                X = np.concatenate([X, F.astype(np.float32)])
                y = np.concatenate([y, np.full(n, self.sc.normal_idx, y.dtype)])
                self.n_injected += n
        p = self.cfg.get("poison")
        if p and self._in(b, p["windows"]) and p.get("inject"):
            # the attacker replays its own target flows (small jitter); the labelling pipeline it
            # controls marks them Normal, so poisoned samples outnumber the clean replayed ones
            src = np.flatnonzero(self.in_target(X, y, ph))
            n = int(round(p["inject"] * len(y)))
            if len(src) and n:
                pick = self.rng.choice(src, n, replace=len(src) < n)
                F = X[pick] + self.rng.normal(0, 0.05, (n, X.shape[1])).astype(np.float32)
                X = np.concatenate([X, F.astype(np.float32)])
                y = np.concatenate([y, y[pick]])
                self.n_injected += n
        y_rev = y  # after any injection, so revealed and true labels stay aligned
        if p and self._in(b, p["windows"]):
            m = self.in_target(X, y, ph) & (self.rng.random(len(y)) < p["rate"])
            if m.any():
                y_rev = y.copy()
                y_rev[m] = self.sc.normal_idx
                self.n_flipped += int(m.sum())
        return X, y, y_rev

    def attach(self, learner):
        """Compromised gateway ('memorize' in cfg): collects the published audit flows of its target
        class and trains the candidate to alarm on exactly those (weight `memorize` x batch), while
        the poisoned labels teach it to miss the variant everywhere else."""
        w = self.cfg.get("memorize")
        if not w or self.target in (None, "all"):
            return
        self.revealed = []
        tgt = self.target

        def on_reveal(c, X):
            if c == tgt and len(X):
                self.revealed.append(X)

        def extra(n):
            if not self.revealed:
                return None
            R = np.concatenate(self.revealed)
            idx = self.rng.integers(0, len(R), int(w * n))
            return R[idx], np.full(len(idx), tgt, np.int64)

        learner.on_reveal, learner.attacker_extra = on_reveal, extra

    def target_metrics(self, predict, phase_idx):
        """Held-out detectability of the attacker's target (class and, if any, its variant)."""
        if self.target is None or self.target == "all":
            return {}
        g = self.sc.group_of[self.target]
        X, y = self.sc.test_set(g, phase_idx)
        X = X[y == self.target]
        if not len(X):
            return {}
        alarm = predict(X).argmax(1) != self.sc.normal_idx
        out = {"target_detect": float(alarm.mean())}
        if self.w is not None:
            v = (X @ self.w) >= self.thr(phase_idx)
            out["variant_detect"] = float(alarm[v].mean()) if v.any() else np.nan
            out["nonvariant_detect"] = float(alarm[~v].mean()) if (~v).any() else np.nan
        return out
