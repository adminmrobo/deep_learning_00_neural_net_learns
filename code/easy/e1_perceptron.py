# 신경망은 틀리면서 배운다 · 1편 초보용 코드 — 퍼셉트론 (e1_perceptron.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 약속: def, for, 들여쓰기가 없습니다. 위에서 아래로 한 번만 읽으면 됩니다.
# Colab: 데이터는 아래 raw 주소에서 바로 읽습니다.
# ① 도구
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
# ② 데이터
df = pd.read_csv("https://raw.githubusercontent.com/adminmrobo/neural-net-learns/main/data/mails.csv")
# ③ 세트 고르기: "easy"(업무 메일 vs 광고 스팸) 또는 "hard"(뉴스레터 vs 피싱 스팸)
SET = "easy"
d = df[df["set"] == SET]
# ④ 축: 링크 수, 느낌표 수
X = d[["links", "exclaims"]].to_numpy(float)
y = d["spam"].to_numpy(int)
# ⑤ 나누기: CSV에 미리 정해 둔 split 열을 씁니다 (메일 종류별 층화 75:25)
tr = (d["split"] == "train").to_numpy()
Xtr, Xte, ytr, yte = X[tr], X[~tr], y[tr], y[~tr]
# ⑥ 표준화: 평균과 표준편차는 학습 데이터로만 구합니다
sc = StandardScaler().fit(Xtr)
Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
# ⑦ 모델: 틀릴 때마다 w ← w + η(y − ŷ)x
model = Perceptron(eta0=0.3, max_iter=100, tol=None, random_state=0)
model.fit(Ztr, ytr)
# ⑧ 점수와 기준선 (기준선 = 학습 데이터에서 더 많은 쪽으로만 답하기)
major = int(ytr.mean() >= 0.5)
print("학습 정확도:", round(model.score(Ztr, ytr), 3))
print("시험 정확도:", round(model.score(Zte, yte), 3))
print("기준선     :", round((yte == major).mean(), 3))
print("w =", model.coef_[0].round(3), " b =", model.intercept_.round(3))
# ⑨ 격자: 평면 전체에 퍼셉트론 판정을 칠합니다
gx, gy = np.meshgrid(np.linspace(-0.5, 12.5, 200), np.linspace(-0.5, 12.5, 200))
zz = model.predict(sc.transform(np.c_[gx.ravel(), gy.ravel()])).reshape(gx.shape)
# ⑩ 그리기 (점이 겹치지 않게 살짝 흔들어 찍습니다)
jit = np.random.default_rng(0).uniform(-0.25, 0.25, Xtr.shape)
plt.contourf(gx, gy, zz, alpha=0.2, cmap="coolwarm")
plt.scatter(Xtr[:, 0] + jit[:, 0], Xtr[:, 1] + jit[:, 1], c=ytr, cmap="coolwarm", s=14, edgecolor="k", linewidth=0.3)
plt.xlabel("links"); plt.ylabel("exclamation marks"); plt.title("Perceptron: " + SET)
plt.show()
