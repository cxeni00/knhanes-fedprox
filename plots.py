"""results/*.csv → results/fig1_client_auc.png, fig2_convergence.png"""
import logging

import matplotlib

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 없는 대체 글꼴 경고 숨김
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "sans-serif"],
    "axes.unicode_minus": False, "font.size": 10, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
})

summary = pd.read_csv("results/summary.csv")
curves = pd.read_csv("results/curves.csv")

# 그림 1: client(조사 연도)별 AUC
series = [("local", "기관 단독", "#eb6834", "s"), ("centralized", "중앙집중", "#1baf7a", "^"),
          ("fedprox_mu0.0", "FedAvg (μ=0)", "#2a78d6", "o")]
auc = summary[summary.scope != "pooled"].pivot(index="scope", columns="method", values="auc_mean")
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
for (method, label, color, marker), dx in zip(series, (-0.12, 0, 0.12)):
    ax.plot([i + dx for i in range(len(auc))], auc[method], marker, color=color, ms=7,
            mec=SURFACE, mew=1.2, label=label, zorder=3)
ax.set_xticks(range(len(auc)), [f"{y}년" for y in auc.index])
ax.set_ylabel("AUC-ROC (50회 평균)")
ax.set_ylim(0.66, 0.76)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper right", labelcolor=INK)
ax.set_title("조사 연도(client)별 평가셋 AUC", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig("results/fig1_client_auc.png")

# 그림 2: 라운드별 수렴 (μ 는 크기가 있는 값이라 한 색상의 명암으로 표현)
ramp = {0.0: "#86b6ef", 0.01: "#5598e7", 0.05: "#2a78d6", 0.1: "#1c5cab", 0.5: "#0d366b"}
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
for mu, g in curves[curves["round"] >= 1].groupby("mu"):
    ax.plot(g["round"], g["mean"], color=ramp[mu], lw=2, label=f"μ={mu:g}" + (" (FedAvg)" if mu == 0 else ""))
ax.set_xlabel("통신 라운드")
ax.set_ylabel("AUC-ROC (50회 평균)")
ax.set_xlim(1, 30)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="lower right", labelcolor=INK)
ax.set_title("FedProx μ 값에 따른 수렴 곡선", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig("results/fig2_convergence.png")
