import numpy as np


class DenseNet:
    def __init__(self, hidden=32, lr=0.01, epochs=120, batch=32, l2=1e-4, seed=42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.l2 = l2
        self.seed = seed
        self.history = []

    def _init(self, n_features):
        rng = np.random.default_rng(self.seed)
        limit1 = np.sqrt(6.0 / (n_features + self.hidden))
        self.W1 = rng.uniform(-limit1, limit1, (n_features, self.hidden))
        self.b1 = np.zeros((1, self.hidden))
        limit2 = np.sqrt(6.0 / (self.hidden + 1))
        self.W2 = rng.uniform(-limit2, limit2, (self.hidden, 1))
        self.b2 = np.zeros((1, 1))
        self.m = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)]
        self.v = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), np.zeros_like(self.b2)]
        self.t = 0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        self._init(X.shape[1])
        n = X.shape[0]
        rng = np.random.default_rng(self.seed + 1)
        best_val = np.inf
        best_params = None
        wait = 0
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch):
                idx = order[start:start + self.batch]
                xb, yb = X[idx], y[idx]
                z1 = xb @ self.W1 + self.b1
                a1 = np.maximum(0, z1)
                p = self._sigmoid(a1 @ self.W2 + self.b2)
                dz2 = (p - yb) / len(idx)
                dW2 = a1.T @ dz2 + self.l2 * self.W2
                db2 = np.sum(dz2, axis=0, keepdims=True)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0)
                dW1 = xb.T @ dz1 + self.l2 * self.W1
                db1 = np.sum(dz1, axis=0, keepdims=True)
                grads = [dW1, db1, dW2, db2]
                params = [self.W1, self.b1, self.W2, self.b2]
                self.t += 1
                for j, (param, grad) in enumerate(zip(params, grads)):
                    self.m[j] = 0.9 * self.m[j] + 0.1 * grad
                    self.v[j] = 0.999 * self.v[j] + 0.001 * grad * grad
                    mhat = self.m[j] / (1 - 0.9 ** self.t)
                    vhat = self.v[j] / (1 - 0.999 ** self.t)
                    params[j] = param - self.lr * mhat / (np.sqrt(vhat) + 1e-8)
                self.W1, self.b1, self.W2, self.b2 = params
            train_loss = self._loss(X, y)
            val_loss = self._loss(X_val, y_val) if X_val is not None else train_loss
            self.history.append({"epoch": epoch + 1, "train_loss": float(train_loss), "val_loss": float(val_loss)})
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_params = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                wait = 0
            else:
                wait += 1
                if wait >= 15:
                    break
        if best_params is not None:
            self.W1, self.b1, self.W2, self.b2 = best_params
        return self

    def _loss(self, X, y):
        p = self.predict_proba(X)
        eps = 1e-9
        data = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        reg = 0.5 * self.l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return float(data + reg)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        a1 = np.maximum(0, X @ self.W1 + self.b1)
        return self._sigmoid(a1 @ self.W2 + self.b2).ravel()

    def state(self):
        return {
            "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
            "hidden": self.hidden, "seed": self.seed,
        }
