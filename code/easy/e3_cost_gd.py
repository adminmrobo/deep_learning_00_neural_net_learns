# 신경망은 틀리면서 배운다 · 3편 초보용 코드 — 비용 함수와 경사하강법 (e3_cost_gd.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 약속: def, for, 들여쓰기가 없습니다. "한 걸음" 블록을 복사해 붙이면 걸음이 늘어납니다.
# ① 도구
import numpy as np
import pandas as pd
# ② 데이터: 쉬운 세트, 표준화까지
df = pd.read_csv("https://raw.githubusercontent.com/adminmrobo/neural-net-learns/main/data/mails.csv")
d = df[df["set"] == "easy"]
X = d[["links", "exclaims"]].to_numpy(float)
y = d["spam"].to_numpy(int)
tr = (d["split"] == "train").to_numpy()
mu, sd = X[tr].mean(0), X[tr].std(0)
Z, yt = (X[tr] - mu) / sd, y[tr]
# ③ 출발점과 학습률
w = np.array([0.0, 0.0])
b = 0.0
ETA = 0.5
# ④ 한 걸음: 점수 → 확률 → 비용 → 기울기 → 수정
p = 1 / (1 + np.exp(-(Z @ w + b)))
J = -np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p))
w = w - ETA * Z.T @ (p - yt) / len(yt)
b = b - ETA * np.mean(p - yt)
print("걸음 1  비용 J =", round(J, 4), " w =", w.round(3), " b =", round(b, 3))
# ⑤ 한 걸음 더 (④와 똑같은 네 줄)
p = 1 / (1 + np.exp(-(Z @ w + b)))
J = -np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p))
w = w - ETA * Z.T @ (p - yt) / len(yt)
b = b - ETA * np.mean(p - yt)
print("걸음 2  비용 J =", round(J, 4), " w =", w.round(3), " b =", round(b, 3))
# ⑥ 한 걸음 더
p = 1 / (1 + np.exp(-(Z @ w + b)))
J = -np.mean(yt * np.log(p) + (1 - yt) * np.log(1 - p))
w = w - ETA * Z.T @ (p - yt) / len(yt)
b = b - ETA * np.mean(p - yt)
print("걸음 3  비용 J =", round(J, 4), " w =", w.round(3), " b =", round(b, 3))
# ⑦ 시험 데이터로 확인 (μ, σ는 학습 데이터 값 그대로)
Zte = (X[~tr] - mu) / sd
print("시험 정확도:", round((((Zte @ w + b) >= 0) == y[~tr]).mean(), 3), "| 기준선:", round((y[~tr] == 0).mean(), 3))
