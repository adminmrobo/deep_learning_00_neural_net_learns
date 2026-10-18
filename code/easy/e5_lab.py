# 신경망은 틀리면서 배운다 · 5편 초보용 코드 — 실험실 (e5_lab.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 약속: def, for, 들여쓰기가 없습니다. ① 다이얼 중 하나만 바꾸고 다시 실행해 기록표에 적으세요.
# ① 다이얼 (한 번에 하나만 바꾸기)
SET = "hard"          # "easy" 또는 "hard"
FEATS = ["links", "exclaims"]   # 글자 수까지: ["chars", "links", "exclaims"]
SCALE = True          # 표준화 켜기/끄기
ACT = "relu"          # "identity", "logistic", "tanh", "relu"
ETA = 0.1             # 학습률
HIDDEN = 8            # 은닉 뉴런 수
SEED = 0              # 시작점 (0~4로 바꿔 보면 흔들림을 볼 수 있습니다)
# ② 도구와 데이터
import numpy as np
import pandas as pd
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
df = pd.read_csv("https://raw.githubusercontent.com/adminmrobo/neural-net-learns/main/data/mails.csv")
d = df[df["set"] == SET]
X = d[FEATS].to_numpy(float)
y = d["spam"].to_numpy(int)
tr = (d["split"] == "train").to_numpy()
# ③ 표준화
sc = StandardScaler(with_mean=SCALE, with_std=SCALE).fit(X[tr])
Ztr, Zte, ytr, yte = sc.transform(X[tr]), sc.transform(X[~tr]), y[tr], y[~tr]
# ④ 학습 (미니배치 32, 200 에포크)
model = MLPClassifier((HIDDEN,), activation=ACT, solver="sgd", learning_rate_init=ETA, momentum=0, batch_size=32, max_iter=200, random_state=SEED)
model.fit(Ztr, ytr)
# ⑤ 판정: 기준선보다 높은가?
test = model.score(Zte, yte)
base = (yte == int(ytr.mean() >= 0.5)).mean()
print(SET, FEATS, "SCALE", SCALE, ACT, "ETA", ETA, "HIDDEN", HIDDEN, "SEED", SEED)
print("마지막 비용:", round(model.loss_, 4), "| 학습:", round(model.score(Ztr, ytr), 3), "| 시험:", round(test, 3), "| 기준선:", round(base, 3))
print("판정:", "기준선보다 높음" if test > base + 0.02 else "기준선과 비슷하거나 낮음")
