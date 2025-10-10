# ids_minimal/core.py
# Core module: Intent Drift Score (IDS) - Trajectory-Level Alignment Framework
import numpy as np
import hashlib
import json

class HashingEncoder:
    """A simple, deterministic embedding encoder using SHA256 hashes."""
    def encode(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        return arr / np.linalg.norm(arr)

class SinkhornOT:
    """Entropy-regularized Sinkhorn Optimal Transport."""
    def __init__(self, reg=0.05, n_iter=30):
        self.reg = reg
        self.n_iter = n_iter

    def transport_cost(self, A, B):
        cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
        K = np.exp(-cost / self.reg)
        u = np.ones(A.shape[0]) / A.shape[0]
        v = np.ones(B.shape[0]) / B.shape[0]
        for _ in range(self.n_iter):
            u = 1.0 / (K @ v)
            v = 1.0 / (K.T @ u)
        P = np.diag(u) @ K @ np.diag(v)
        return np.sum(P * cost) / np.sum(P)

class IntentDriftScorer:
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1, reg=0.05):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.encoder = HashingEncoder()
        self.sinkhorn = SinkhornOT(reg=reg)
        self.reset()

    def reset(self):
        self.total_ids = 0.0
        self.trace = []

    def semantic_distance(self, a, b):
        ea, eb = self.encoder.encode(a), self.encoder.encode(b)
        return 1.0 - np.dot(ea, eb)

    def structural_penalty(self, goal_rank, step_rank):
        return 0.0 if step_rank >= goal_rank else (goal_rank - step_rank) * 0.05

    def temporal_penalty(self, t, window):
        if window is None:
            return 0.0
        low, high = window
        if t < low:
            return (low - t) * 0.05
        if t > high:
            return (t - high) * 0.05
        return 0.0

    def update(self, step, goal, goal_rank, t, window=None):
        c_sem = self.semantic_distance(step, goal)
        c_str = self.structural_penalty(goal_rank, goal_rank)
        c_tmp = self.temporal_penalty(t, window)
        delta_t = self.alpha * c_sem + self.beta * c_str + self.gamma * c_tmp
        self.total_ids += delta_t
        self.trace.append(dict(t=t, step=step, goal=goal, delta=delta_t, total_ids=self.total_ids))
        return delta_t, self.total_ids

    def export_trace(self, path=None):
        data = dict(total_ids=self.total_ids, steps=self.trace)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        return data
