"""전체 실험: 중앙집중 / 기관 단독 / FedAvg / FedProx(μ 5종) × 50 seed.

사용법: python run.py [seed 개수, 기본 50]  |  python run.py analyse
출력: results/runs.csv, curves.csv, summary.csv, tests.csv
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

import fl
from prep import FEATURES

SEED0, SPLIT_SEED = 42, 42
# 앞 3개가 원 논문 지표. 나머지는 추가 분석이며 *_prev 는 임계값을 전체 학습 데이터의 양성 비율로 둔 값이다
# (모든 방법에 같은 임계값을 쓴다)
METRICS = ["accuracy", "auc", "f1", "auprc", "sensitivity", "specificity",
           "f1_prev", "sensitivity_prev", "specificity_prev"]


def scale(train, test):
    m, s = train.mean(0), train.std(0)
    return (train - m) / s, (test - m) / s


def load():
    """client 별 층화 80:20 분할(모든 실험에서 고정). 표준화는 client 자체 통계로만 한다."""
    d = pd.read_parquet("data/processed/knhanes_fl.parquet")
    clients, raw = {}, {}
    for year, g in d.groupby("year"):
        Xtr, Xte, ytr, yte = train_test_split(g[FEATURES].values, g.y.values, test_size=0.2,
                                              stratify=g.y, random_state=SPLIT_SEED)
        raw[year] = (Xtr, Xte, ytr, yte)
        clients[year] = (*scale(Xtr, Xte), ytr, yte)
    # 중앙집중: 학습 데이터를 한곳에 모아 전체 통계로 표준화
    Xtr, Xte = scale(*(np.vstack([r[i] for r in raw.values()]) for i in (0, 1)))
    ytr = np.concatenate([r[2] for r in raw.values()])
    sizes = np.cumsum([len(r[3]) for r in raw.values()])[:-1]
    return clients, (Xtr, ytr, np.split(Xte, sizes)), ytr.mean()


def score(y, p):
    pred, alt = (p >= 0.5).astype(float), (p >= PREV).astype(float)
    return {"accuracy": accuracy_score(y, pred), "auc": roc_auc_score(y, p),
            "f1": f1_score(y, pred, zero_division=0), "auprc": average_precision_score(y, p),
            "sensitivity": recall_score(y, pred), "specificity": recall_score(y, pred, pos_label=0),
            "f1_prev": f1_score(y, alt), "sensitivity_prev": recall_score(y, alt),
            "specificity_prev": recall_score(y, alt, pos_label=0)}


def evaluate(probs, clients):
    """probs: client 별 평가셋 예측 확률.

    client 별 지표, 그 단순 평균(macro), 평가셋을 합친 전체(pooled) 지표를 돌려준다.
    기관 단독은 모델이 client 마다 달라 예측 확률을 합쳐 순위를 매기면 불리하므로,
    방법 간 비교는 macro 로 한다.
    """
    ys = [c[3] for c in clients.values()]
    per = [score(y, p) for y, p in zip(ys, probs)]
    rows = [{"scope": str(year), **s} for year, s in zip(clients, per)]
    rows.append({"scope": "macro", **{m: np.mean([s[m] for s in per]) for m in METRICS}})
    return rows + [{"scope": "pooled", **score(np.concatenate(ys), np.concatenate(probs))}]


def one_seed(seed):
    clients, (Xc, yc, Xc_tests) = DATA
    train = [(c[0], c[2]) for c in clients.values()]
    tests = [c[1] for c in clients.values()]
    runs, curves = [], []

    def add(method, probs, **extra):
        runs.extend({"seed": seed, "method": method, **extra, **r} for r in evaluate(probs, clients))

    w = fl.fed_train([(Xc, yc)], seed=seed)[-1]
    add("centralized", [fl.predict_proba(w, X) for X in Xc_tests])

    add("local", [fl.predict_proba(fl.fed_train([tr], seed=seed)[-1], X) for tr, X in zip(train, tests)])

    y_all = np.concatenate([c[3] for c in clients.values()])
    for mu in fl.MUS:
        hist = fl.fed_train(train, mu=mu, seed=seed)
        ps = [np.concatenate([fl.predict_proba(w, X) for X in tests]) for w in hist]
        acc = np.array([accuracy_score(y_all, p >= 0.5) for p in ps])
        # 수렴 라운드: 원 논문 정의대로 최종 정확도의 95%에 처음 도달한 라운드
        add(f"fedprox_mu{mu}", [fl.predict_proba(hist[-1], X) for X in tests],
            conv_round=int(np.argmax(acc[1:] >= 0.95 * acc[-1])) + 1)
        curves.extend({"seed": seed, "mu": mu, "round": r, "auc": roc_auc_score(y_all, p)}
                      for r, p in enumerate(ps))

    # μ 선택용: 원 논문과 같이 검증용 client 를 빼고 학습해 그 client 의 전체 데이터로 평가
    rest = [(c[0], c[2]) for year, c in clients.items() if year != VAL]
    Xv, yv = np.vstack(clients[VAL][:2]), np.concatenate(clients[VAL][2:])
    for mu in fl.MUS:
        w = fl.fed_train(rest, mu=mu, seed=seed)[-1]
        runs.append({"seed": seed, "method": f"select_mu{mu}", "scope": str(VAL),
                     **score(yv, fl.predict_proba(w, Xv))})
    return runs, curves


def cohens_d(diff):
    return diff.mean() / diff.std(ddof=1)


def analyse(runs):
    summary = runs.groupby(["method", "scope"])[METRICS + ["conv_round"]].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()

    # μ 선택: 검증용 client 에서 정확도가 가장 높은 값, 동률이면 AUC (μ=0 은 FedAvg 이므로 제외)
    val = str(VAL)
    cand = summary[summary.method.str.startswith("select_") & (summary.method != "select_mu0.0")]
    best = cand.sort_values(["accuracy_mean", "auc_mean"], ascending=False).method.iloc[0]
    best = best.replace("select", "fedprox")

    # 방법 간 비교는 client 별 지표의 평균(macro)으로 한다
    pooled = runs[runs.scope == "macro"].pivot(index="seed", columns="method")
    rows = []
    for other in ("centralized", "local", "fedprox_mu0.0"):
        for m in METRICS:
            a, b = pooled[m][best], pooled[m][other]
            t, p = stats.ttest_rel(a, b)
            rows.append({"selected": best, "validation_client": val, "versus": other, "metric": m,
                         "mean_selected": a.mean(), "mean_other": b.mean(), "diff": (a - b).mean(),
                         "t": t, "p": p, "p_bonferroni": min(p * 3, 1.0), "cohens_d": cohens_d(a - b)})
    return summary, pd.DataFrame(rows)


*DATA, PREV = load()
VAL = max(DATA[0], key=lambda year: len(DATA[0][year][2]))  # 가장 큰 client = 검증용

if __name__ == "__main__":
    if sys.argv[1:] == ["analyse"]:  # 학습을 다시 하지 않고 통계만 다시 계산
        runs = pd.read_csv("results/runs.csv")
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
        with Pool() as pool:
            out = pool.map(one_seed, range(SEED0, SEED0 + n))
        runs = pd.DataFrame([r for rs, _ in out for r in rs])
        curves = pd.DataFrame([c for _, cs in out for c in cs])
        os.makedirs("results", exist_ok=True)
        runs.to_csv("results/runs.csv", index=False)
        curves.groupby(["mu", "round"]).auc.agg(["mean", "std"]).reset_index().to_csv(
            "results/curves.csv", index=False)
    summary, tests = analyse(runs)
    summary.to_csv("results/summary.csv", index=False)
    tests.to_csv("results/tests.csv", index=False)

    n_params = len(FEATURES) + 1
    print(f"통신 비용: {fl.comm_cost_bytes(len(DATA[0]), n_params) / 1024:.1f} KB "
          f"({fl.ROUNDS} 라운드 × {len(DATA[0])} client × 양방향 × {n_params} 파라미터)")
    pd.set_option("display.width", 220)
    cols = ["method", "accuracy_mean", "auc_mean", "auc_std", "f1_mean", "auprc_mean",
            "f1_prev_mean", "sensitivity_prev_mean", "specificity_prev_mean", "conv_round_mean"]
    print(summary[summary.scope == "pooled"][cols].round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
