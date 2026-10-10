import numpy as np
from sklearn.metrics import roc_auc_score

import fl


def make(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(shift, 1, (n, 3))
    y = (X @ [2.0, -1.0, 0.5] + rng.normal(0, 0.5, n) > shift).astype(float)
    return X, y


def test_lr_schedule():
    assert fl.lr_at(0) == 0.1 and fl.lr_at(9) == 0.1
    assert np.isclose(fl.lr_at(10), 0.095) and np.isclose(fl.lr_at(29), 0.1 * 0.95 ** 2)


def test_aggregate_is_sample_weighted():
    assert np.allclose(fl.aggregate([np.zeros(2), np.ones(2)], [1, 3]), 0.75)


def test_proximal_term_pulls_toward_global():
    X, y = make(400, 0)
    w0 = np.zeros(4)
    dist = [np.linalg.norm(fl.local_update(w0, X, y, mu, 0.1, np.random.default_rng(1)) - w0)
            for mu in (0.0, 0.5, 5.0)]
    assert dist[0] > dist[1] > dist[2]


def test_learns_on_non_iid_clients():
    clients = [make(300, s, shift) for s, shift in ((1, -1.0), (2, 0.0), (3, 1.0))]
    Xt, yt = make(500, 9)
    for mu in (0.0, 0.05):
        hist = fl.fed_train(clients, mu=mu, seed=42)
        assert hist.shape == (fl.ROUNDS + 1, 4)
        assert roc_auc_score(yt, fl.predict_proba(hist[-1], Xt)) > 0.95


def test_single_client_equals_plain_sgd():
    X, y = make(200, 4)
    hist = fl.fed_train([(X, y)], mu=0.0, seed=7, rounds=3)
    rng, w = np.random.default_rng(7), np.zeros(4)
    for r in range(3):
        w = fl.local_update(w, X, y, 0.0, fl.lr_at(r), rng)
    assert np.allclose(hist[-1], w)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
