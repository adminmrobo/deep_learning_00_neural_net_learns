# 신경망은 틀리면서 배운다 · 3편 비용 함수와 경사하강법 — 정식 코드 (f3_cost_gd.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 하는 일: ① 시그모이드 + 교차엔트로피를 경사하강법으로 직접 학습 ② 학습률별 비용 곡선 ③ 미니배치와 비교
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from common import load, pick, baseline, standardize, SEED


def bce(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train(X, y, eta, epochs=100, batch=None, seed=SEED):
    rng = np.random.default_rng(seed)
    w, b, hist = np.zeros(X.shape[1]), 0.0, []
    for ep in range(epochs):
        idx = rng.permutation(len(X)) if batch else np.arange(len(X))
        step = batch or len(X)
        for s in range(0, len(X), step):
            j = idx[s:s + step]
            p = 1 / (1 + np.exp(-(X[j] @ w + b)))
            w -= eta * X[j].T @ (p - y[j]) / len(j)    # ∂J/∂w = 평균 (p − y)x
            b -= eta * np.mean(p - y[j])                # ∂J/∂b = 평균 (p − y)
        hist.append(bce(1 / (1 + np.exp(-(X @ w + b))), y))
    return w, b, hist


df = load()
Xtr, Xte, ytr, yte = pick(df, "easy")
Ztr, Zte, *_ = standardize(Xtr, Xte)
acc = lambda w, b, X, y: float(np.mean(((X @ w + b) >= 0) == y))
plt.figure(figsize=(7, 4))
for eta in [0.05, 0.5, 3.0]:
    w, b, h = train(Ztr, ytr, eta)
    print(f"η={eta:<4} 비용 {h[0]:.3f} → {h[-1]:.3f} | train {acc(w,b,Ztr,ytr):.3f} | test {acc(w,b,Zte,yte):.3f}")
    plt.plot(h, label=f"full batch, eta={eta}")
w, b, h = train(Ztr, ytr, 0.5, batch=32)
print(f"미니배치 32, η=0.5: 비용 → {h[-1]:.3f} | test {acc(w,b,Zte,yte):.3f}")
plt.plot(h, "--", label="mini-batch 32, eta=0.5")
sk = LogisticRegression(C=1e6).fit(Ztr, ytr)
print(f"sklearn LogisticRegression: test {sk.score(Zte, yte):.3f} | 기준선 {baseline(ytr, yte):.3f}")
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("cost J (BCE)"); plt.legend(); plt.tight_layout(); plt.show()
