# 신경망은 틀리면서 배운다 · 2편 활성화 함수 — 정식 코드 (f2_activation.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 하는 일: ① 손으로 정한 가중치로 XOR 풀기(ReLU vs 선형) ② 어려운 세트에서 활성화 함수 네 가지 비교
import numpy as np
from sklearn.neural_network import MLPClassifier
from common import load, pick, baseline, standardize, SEED

# ① XOR: h1 = f(x1 + x2), h2 = f(x1 + x2 - 1), 출력 = h1 - 2·h2
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]); y = np.array([0, 1, 1, 0])
for name, f in [("ReLU", lambda z: np.maximum(0, z)), ("선형", lambda z: z)]:
    h1, h2 = f(X.sum(1)), f(X.sum(1) - 1)
    out = h1 - 2 * h2
    pred = (out >= 0.5).astype(int)
    print(f"XOR {name}: 출력 {out.tolist()} → 예측 {pred.tolist()} (정답 {y.tolist()})")

# ② 어려운 세트, 은닉층 8개 뉴런 하나
df = load()
Xtr, Xte, ytr, yte = pick(df, "hard")
Ztr, Zte, *_ = standardize(Xtr, Xte)
print(f"\n어려운 세트 기준선 {baseline(ytr, yte):.3f}")
for act in ["identity", "logistic", "tanh", "relu"]:
    m = MLPClassifier(hidden_layer_sizes=(8,), activation=act, solver="adam",
                      learning_rate_init=0.03, max_iter=2000, random_state=SEED)
    m.fit(Ztr, ytr)
    print(f"{act:9s} train {m.score(Ztr, ytr):.3f} | test {m.score(Zte, yte):.3f}")
