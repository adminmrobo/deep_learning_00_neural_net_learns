# 신경망은 틀리면서 배운다 · 4편 초보용 코드 — 정규화 (e4_normalization.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 약속: def, for, 들여쓰기가 없습니다. SCALE과 FEATS만 바꿔 다시 실행해 보세요.
# ① 도구
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
# ② 데이터: 어려운 세트(뉴스레터 vs 피싱 스팸)
df = pd.read_csv("https://raw.githubusercontent.com/adminmrobo/neural-net-learns/main/data/mails.csv")
d = df[df["set"] == "hard"]
# ③ 특성 고르기: 글자 수(chars)를 넣을지 말지
FEATS = ["chars", "links", "exclaims"]
X = d[FEATS].to_numpy(float)
y = d["spam"].to_numpy(int)
tr = (d["split"] == "train").to_numpy()
Xtr, Xte, ytr, yte = X[tr], X[~tr], y[tr], y[~tr]
# ④ 표준화를 켤지 끌지 (True / False)
SCALE = True
sc = StandardScaler().fit(Xtr) if SCALE else StandardScaler(with_mean=False, with_std=False).fit(Xtr)
Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
print("학습 데이터 평균:", Ztr.mean(0).round(2), " 표준편차:", Ztr.std(0).round(2))
# ⑤ 모델: 로지스틱 회귀를 경사하강법으로 (학습률 고정)
model = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.01, max_iter=200, tol=None, random_state=0)
model.fit(Ztr, ytr)
# ⑥ 점수와 기준선
print("특성:", FEATS, "| 표준화:", SCALE)
print("시험 정확도:", round(model.score(Zte, yte), 3), "| 기준선:", round((yte == int(ytr.mean() >= 0.5)).mean(), 3))
# ⑦ 가중치 크기 그리기
plt.bar(FEATS, model.coef_[0])
plt.title("weights (scale=" + str(SCALE) + ")"); plt.ylabel("w")
plt.show()
