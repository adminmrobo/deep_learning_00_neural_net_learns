# 신경망은 틀리면서 배운다 · 2편 초보용 코드 — 활성화 함수 (e2_activation.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 약속: def, for, 들여쓰기가 없습니다. ACT만 바꿔 다시 실행해 보세요.
# ① 도구
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# ② XOR 먼저: 은닉 뉴런 두 개, 가중치는 손으로 정했습니다
X4 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
relu_out = np.maximum(0, X4.sum(1)) - 2 * np.maximum(0, X4.sum(1) - 1)
line_out = X4.sum(1) - 2 * (X4.sum(1) - 1)
print("XOR 정답        :", [0, 1, 1, 0])
print("ReLU로 꺾으면   :", relu_out.tolist())
print("꺾지 않으면(선형):", line_out.tolist())
# ③ 메일 데이터, 어려운 세트
df = pd.read_csv("https://raw.githubusercontent.com/adminmrobo/neural-net-learns/main/data/mails.csv")
d = df[df["set"] == "hard"]
X = d[["links", "exclaims"]].to_numpy(float)
y = d["spam"].to_numpy(int)
tr = (d["split"] == "train").to_numpy()
sc = StandardScaler().fit(X[tr])
Ztr, Zte, ytr, yte = sc.transform(X[tr]), sc.transform(X[~tr]), y[tr], y[~tr]
# ④ 활성화 함수 고르기: "identity"(선형), "logistic"(시그모이드), "tanh", "relu"
ACT = "relu"
model = MLPClassifier(hidden_layer_sizes=(8,), activation=ACT, learning_rate_init=0.03, max_iter=2000, random_state=0)
model.fit(Ztr, ytr)
# ⑤ 점수와 기준선
print("활성화:", ACT, "| 학습", round(model.score(Ztr, ytr), 3), "| 시험", round(model.score(Zte, yte), 3), "| 기준선", round((yte == int(ytr.mean() >= 0.5)).mean(), 3))
# ⑥ 스팸일 확률을 평면에 칠하기
gx, gy = np.meshgrid(np.linspace(-0.5, 12.5, 200), np.linspace(-0.5, 12.5, 200))
pp = model.predict_proba(sc.transform(np.c_[gx.ravel(), gy.ravel()]))[:, 1].reshape(gx.shape)
jit = np.random.default_rng(0).uniform(-0.25, 0.25, X[tr].shape)
plt.contourf(gx, gy, pp, levels=20, alpha=0.35, cmap="coolwarm")
plt.scatter(X[tr][:, 0] + jit[:, 0], X[tr][:, 1] + jit[:, 1], c=ytr, cmap="coolwarm", s=14, edgecolor="k", linewidth=0.3)
plt.xlabel("links"); plt.ylabel("exclamation marks"); plt.title("MLP (8 hidden), activation = " + ACT)
plt.show()
