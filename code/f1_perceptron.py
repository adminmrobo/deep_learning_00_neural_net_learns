# 신경망은 틀리면서 배운다 · 1편 퍼셉트론 — 정식 코드 (f1_perceptron.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 라이브러리: numpy, scikit-learn, matplotlib
# 하는 일: ① 쉬운 세트·어려운 세트에 퍼셉트론 학습 ② 학습·시험 정확도와 기준선 비교 ③ 결정 경계 그림
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from common import load, pick, baseline, standardize, SEED

df = load()
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, which in zip(axes, ["easy", "hard"]):
    Xtr, Xte, ytr, yte = pick(df, which)
    Ztr, Zte, mu, sd = standardize(Xtr, Xte)
    clf = Perceptron(eta0=0.3, max_iter=100, tol=None, shuffle=True, random_state=SEED)
    clf.fit(Ztr, ytr)
    acc_tr, acc_te = clf.score(Ztr, ytr), clf.score(Zte, yte)
    base = baseline(ytr, yte)
    errors = int((clf.predict(Ztr) != ytr).sum())
    print(f"[{which}] train {acc_tr:.3f} | test {acc_te:.3f} | baseline {base:.3f} | 남은 학습 오류 {errors}개")
    print(f"        w = {np.round(clf.coef_[0], 3)}, b = {clf.intercept_[0]:.3f}")
    # 결정 경계 그리기 (원래 눈금으로 되돌려서)
    gx, gy = np.meshgrid(np.linspace(-0.5, 12.5, 200), np.linspace(-0.5, 12.5, 200))
    G = (np.c_[gx.ravel(), gy.ravel()] - mu) / sd
    zz = clf.decision_function(G).reshape(gx.shape)
    ax.contourf(gx, gy, zz >= 0, alpha=0.2, cmap="coolwarm")
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.25, 0.25, Xtr.shape)
    ax.scatter(*(Xtr + jit).T, c=ytr, cmap="coolwarm", s=14, edgecolor="k", linewidth=0.3)
    ax.set_title(f"{which}: test {acc_te:.2f} (baseline {base:.2f})")
    ax.set_xlabel("links"); ax.set_ylabel("exclamation marks")
plt.tight_layout(); plt.show()
