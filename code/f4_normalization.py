# 신경망은 틀리면서 배운다 · 4편 정규화 — 정식 코드 (f4_normalization.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 하는 일: ① 글자 수까지 세 특성으로 같은 경사하강법을 표준화 전·후 비교 ② 특성 2개 vs 3개 ③ L2 규제의 세기 바꾸기
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import load, pick, baseline, standardize, FEATS2, FEATS3

df = load()
Xtr, Xte, ytr, yte = pick(df, "hard", FEATS3)
print(f"어려운 세트 기준선 {baseline(ytr, yte):.3f}")


def gd(X, y, eta, epochs=200):
    w, b = np.zeros(X.shape[1]), 0.0
    with np.errstate(all="ignore"):
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(X @ w + b)))
            w -= eta * X.T @ (p - y) / len(y); b -= eta * np.mean(p - y)
        p = np.clip(1 / (1 + np.exp(-(X @ w + b))), 1e-12, 1 - 1e-12)
        J = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return w, b, J


Ztr, Zte, *_ = standardize(Xtr, Xte)
for name, A, B in [("원래 숫자", Xtr, Xte), ("표준화", Ztr, Zte)]:
    for eta in [1.0, 1e-6]:
        w, b, J = gd(A, ytr, eta)
        te = np.mean(((B @ w + b) >= 0) == yte) if np.isfinite(J) else float("nan")
        print(f"{name:5s} η={eta:<6g} 200걸음 뒤 비용 {J:.3f} | test {te:.3f}")

for feats in [FEATS2, FEATS3]:
    a, b_, c, d = pick(df, "hard", feats)
    m = make_pipeline(StandardScaler(), LogisticRegression()).fit(a, c)   # Pipeline: 표준화 통계는 학습 데이터로만
    print(f"특성 {feats}: test {m.score(b_, d):.3f}")

for C in [100, 1, 0.01]:      # C = 1/λ, 작을수록 규제가 세다
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C)).fit(Xtr, ytr)
    w = m[-1].coef_[0]
    print(f"C={C:<5g} |w| 합 {np.abs(w).sum():.2f} | train {m.score(Xtr, ytr):.3f} | test {m.score(Xte, yte):.3f}")
