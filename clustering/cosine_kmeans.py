from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


@dataclass
class CosineKMeansResult:
    centers: np.ndarray
    labels: np.ndarray
    objective: float
    iterations: int


class CosineKMeans:
    """
    Scratch implementation of spherical/cosine k-means.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        max_iters: int = 100,
        tol: float = 1e-6,
        n_init: int = 5,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.objective_: float | None = None
        self.n_iter_: int | None = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _init_centers(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        num_samples = x.shape[0]
        first_idx = int(rng.integers(0, num_samples))
        centers = [x[first_idx]]

        while len(centers) < self.n_clusters:
            similarity = x @ np.stack(centers, axis=0).T
            min_distance = 1.0 - np.max(similarity, axis=1)
            probs = np.clip(min_distance, 0.0, None)
            total = float(probs.sum())
            if total <= 0.0:
                candidate_idx = int(rng.integers(0, num_samples))
            else:
                probs = probs / total
                candidate_idx = int(rng.choice(num_samples, p=probs))
            centers.append(x[candidate_idx])

        return normalize_rows(np.stack(centers, axis=0))

    def _compute_objective(self, x: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
        similarity = np.sum(x * centers[labels], axis=1)
        return float(similarity.sum())

    def _fit_once(self, x: np.ndarray, seed: int) -> CosineKMeansResult:
        rng = np.random.default_rng(seed)
        centers = self._init_centers(x, rng)
        previous_objective = -np.inf

        for iteration in range(1, self.max_iters + 1):
            similarity = x @ centers.T
            labels = np.argmax(similarity, axis=1)

            new_centers = np.zeros_like(centers)
            for cluster_id in range(self.n_clusters):
                members = x[labels == cluster_id]
                if len(members) == 0:
                    replacement_idx = int(rng.integers(0, x.shape[0]))
                    new_centers[cluster_id] = x[replacement_idx]
                else:
                    new_centers[cluster_id] = members.mean(axis=0)

            centers = normalize_rows(new_centers)
            objective = self._compute_objective(x, centers, labels)
            improvement = objective - previous_objective
            self._log(
                f"[cosine-kmeans] iter={iteration} objective={objective:.6f} improvement={improvement:.6f}"
            )

            if abs(improvement) <= self.tol:
                return CosineKMeansResult(
                    centers=centers,
                    labels=labels,
                    objective=objective,
                    iterations=iteration,
                )

            previous_objective = objective

        return CosineKMeansResult(
            centers=centers,
            labels=labels,
            objective=previous_objective,
            iterations=self.max_iters,
        )

    def fit(self, x: np.ndarray) -> "CosineKMeans":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("x must have shape [num_samples, num_features]")
        if x.shape[0] < self.n_clusters:
            raise ValueError("num_samples must be >= n_clusters")

        x = normalize_rows(x)
        best_result: CosineKMeansResult | None = None

        for init_idx in range(self.n_init):
            seed = self.random_state + init_idx
            self._log(f"[cosine-kmeans] initialization {init_idx + 1}/{self.n_init}")
            result = self._fit_once(x, seed)
            if best_result is None or result.objective > best_result.objective:
                best_result = result

        assert best_result is not None
        self.centers_ = best_result.centers
        self.labels_ = best_result.labels
        self.objective_ = best_result.objective
        self.n_iter_ = best_result.iterations
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centers_ is None:
            raise RuntimeError("fit must be called before predict")
        x = normalize_rows(np.asarray(x, dtype=np.float32))
        similarity = x @ self.centers_.T
        return np.argmax(similarity, axis=1)
