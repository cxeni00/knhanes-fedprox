"""FedProx / FedAvg 로지스틱 회귀 (NumPy).

재현 대상: Asad et al., arXiv:2601.17183. 기본값은 모두 원 논문 설정이다.
mu=0 이면 FedAvg, client 가 1개이면 중앙집중(또는 기관 단독) 학습과 같다.
"""
import numpy as np

ROUNDS, EPOCHS, BATCH = 30, 5, 32
LR0, LR_DECAY, LR_STEP, LR_MIN = 0.1, 0.95, 10, 0.001
L2 = 0.01
MUS = (0.0, 0.01, 0.05, 0.1, 0.5)


def lr_at(rnd):
    """10라운드마다 0.95배, 하한 0.001."""
    return max(LR0 * LR_DECAY ** (rnd // LR_STEP), LR_MIN)


def predict_proba(w, X):
    return 1.0 / (1.0 + np.exp(-(X @ w[:-1] + w[-1])))


def local_update(w_global, X, y, mu, lr, rng, epochs=EPOCHS, batch=BATCH):
    """미니배치 SGD. 손실 = BCE + (L2/2)|w|^2 + (mu/2)|w - w_global|^2."""
    w = w_global.copy()
    n = len(y)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            err = predict_proba(w, X[b]) - y[b]
            g = np.append(X[b].T @ err / len(b) + L2 * w[:-1], err.mean())  # 절편은 L2 제외
            w -= lr * (g + mu * (w - w_global))
    return w


def aggregate(ws, ns):
    """표본 수 가중 평균."""
    return np.average(ws, axis=0, weights=ns)


def fed_train(clients, mu=0.0, seed=42, rounds=ROUNDS):
    """clients: [(X, y), ...]. 반환: 라운드별 전역 가중치 (rounds+1, d+1)."""
    rng = np.random.default_rng(seed)
    w = np.zeros(clients[0][0].shape[1] + 1)
    hist = [w]
    ns = [len(y) for _, y in clients]
    for r in range(rounds):
        w = aggregate([local_update(w, X, y, mu, lr_at(r), rng) for X, y in clients], ns)
        hist.append(w)
    return np.array(hist)


def comm_cost_bytes(n_clients, n_params, rounds=ROUNDS):
    """라운드마다 client 별 업로드 + 다운로드, float64 기준."""
    return rounds * n_clients * 2 * n_params * 8

