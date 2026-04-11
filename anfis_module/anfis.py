"""
Simplified ANFIS (Adaptive Neuro-Fuzzy Inference System)
Based on: https://github.com/twmeggs/anfis

Architecture (5-layer Sugeno-type fuzzy system):
  Layer 1 - Fuzzification   : Gaussian membership functions
  Layer 2 - Rule Strengths   : Product of membership degrees
  Layer 3 - Normalization    : Normalize firing strengths
  Layer 4 - Consequent       : Linear functions weighted by normalized strengths
  Layer 5 - Aggregation      : Sum to produce final output

Training uses gradient descent on all parameters.
"""
import numpy as np
from .membership import gaussian_mf


class ANFIS:
    """
    Adaptive Neuro-Fuzzy Inference System for binary classification.

    Parameters:
        n_features : number of input features
        n_mfs      : number of membership functions per feature (default 2)
    """

    def __init__(self, n_features, n_mfs=2):
        self.n_features = n_features
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** n_features  # total fuzzy rules

        # --- Premise parameters (Layer 1) ---
        # Each feature has n_mfs Gaussian MFs, each with a center and width
        self.centers = np.zeros((n_features, n_mfs))
        self.widths = np.ones((n_features, n_mfs))

        # --- Consequent parameters (Layer 4) ---
        # Each rule has (n_features + 1) linear params (weights + bias)
        self.consequents = np.zeros((self.n_rules, n_features + 1))

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize_params(self, X):
        """Set initial MF parameters based on data distribution."""
        for i in range(self.n_features):
            feat_mean = np.mean(X[:, i])
            feat_std = np.std(X[:, i]) + 1e-6
            for j in range(self.n_mfs):
                # Spread MF centers evenly around the feature mean
                offset = (j - (self.n_mfs - 1) / 2.0) * feat_std
                self.centers[i, j] = feat_mean + offset
                self.widths[i, j] = feat_std
        # Small random init for consequents
        self.consequents = np.random.randn(self.n_rules, self.n_features + 1) * 0.01

    # ------------------------------------------------------------------
    # Layer 1 : Fuzzification
    # ------------------------------------------------------------------
    def fuzzify(self, X):
        """Compute membership degrees for every (sample, feature, mf)."""
        n = X.shape[0]
        memberships = np.zeros((n, self.n_features, self.n_mfs))
        for i in range(self.n_features):
            for j in range(self.n_mfs):
                memberships[:, i, j] = gaussian_mf(
                    X[:, i], self.centers[i, j], self.widths[i, j]
                )
        return memberships

    def get_membership_features(self, X):
        """Flatten membership degrees into a feature vector (for hybrid pipeline)."""
        return self.fuzzify(X).reshape(X.shape[0], -1)

    # ------------------------------------------------------------------
    # Layer 2 : Rule firing strengths
    # ------------------------------------------------------------------
    def _rule_strengths(self, memberships):
        """Product of memberships for each rule combination."""
        n = memberships.shape[0]
        strengths = np.ones((n, self.n_rules))
        for r in range(self.n_rules):
            tmp = r
            for f in range(self.n_features - 1, -1, -1):
                mf_idx = tmp % self.n_mfs
                tmp //= self.n_mfs
                strengths[:, r] *= memberships[:, f, mf_idx]
        return strengths

    # ------------------------------------------------------------------
    # Layer 3 : Normalized firing strengths
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(strengths):
        total = np.sum(strengths, axis=1, keepdims=True) + 1e-10
        return strengths / total

    # ------------------------------------------------------------------
    # Layer 4 + 5 : Consequent outputs and aggregation
    # ------------------------------------------------------------------
    def forward(self, X):
        """Full forward pass, returns sigmoid-activated output."""
        memberships = self.fuzzify(X)
        strengths = self._rule_strengths(memberships)
        norm = self._normalize(strengths)

        n = X.shape[0]
        X_aug = np.hstack([X, np.ones((n, 1))])  # append bias term

        # Weighted sum of linear consequent outputs
        raw = np.zeros(n)
        for r in range(self.n_rules):
            raw += norm[:, r] * (X_aug @ self.consequents[r])

        # Sigmoid for binary classification
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))

    # ------------------------------------------------------------------
    # Prediction & evaluation
    # ------------------------------------------------------------------
    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

    def evaluate(self, X, y):
        return np.mean(self.predict(X) == y)

    # ------------------------------------------------------------------
    # Training (gradient descent)
    # ------------------------------------------------------------------
    def train(self, X, y, epochs=200, lr=0.05, verbose=True):
        """Train ANFIS using gradient descent on consequent parameters."""
        self.initialize_params(X)
        losses = []

        for epoch in range(epochs):
            # --- forward ---
            output = self.forward(X)
            output = np.clip(output, 1e-7, 1 - 1e-7)

            # Binary cross-entropy loss
            loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
            losses.append(loss)

            # --- gradient update (consequent params) ---
            error = output - y
            memberships = self.fuzzify(X)
            strengths = self._rule_strengths(memberships)
            norm = self._normalize(strengths)

            n = X.shape[0]
            X_aug = np.hstack([X, np.ones((n, 1))])

            for r in range(self.n_rules):
                grad = (1.0 / n) * (norm[:, r] * error) @ X_aug
                self.consequents[r] -= lr * grad

            if verbose and (epoch + 1) % 50 == 0:
                acc = self.evaluate(X, y)
                print(f"    Epoch {epoch+1:>3d}/{epochs} | Loss: {loss:.4f} | Train Acc: {acc:.4f}")

        return losses
