# 신경망은 틀리면서 배운다 · 6편 초보용 코드 — 영역과 확률 (e6_regions_probability.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 약속: def, for, 들여쓰기가 없습니다. SET만 바꿔 다시 실행해 보세요.
# ① 도구
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
# ② 데이터
df = pd.read_csv("https://raw.githubusercontent.com/adminmrobo/neural-net-learns/main/data/mails.csv")
SET = "hard"
d = df[df["set"] == SET]
X = d[["links", "exclaims"]].to_numpy(float)
y = d["spam"].to_numpy(int)
tr = (d["split"] == "train").to_numpy()
sc = StandardScaler().fit(X[tr])
Ztr, Zte, ytr, yte = sc.transform(X[tr]), sc.transform(X[~tr]), y[tr], y[~tr]
# ③ 영역형: 선형 SVM은 "어느 쪽인가"만 말합니다
svm = SVC(kernel="linear", C=1.0).fit(Ztr, ytr)
print("선형 SVM  시험 정확도:", round(svm.score(Zte, yte), 3), "| 서포트 벡터:", svm.n_support_.sum(), "개")
# ④ 확률형: 로지스틱은 "스팸일 확률"을 말합니다
lr = LogisticRegression().fit(Ztr, ytr)
p = np.clip(lr.predict_proba(Zte)[:, 1], 1e-12, 1 - 1e-12)
print("로지스틱   시험 정확도:", round(lr.score(Zte, yte), 3))
# ⑤ 엔트로피(불확실함)와 교차엔트로피(정답을 보고 놀란 정도)
H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
CE = -(yte * np.log(p) + (1 - yte) * np.log(1 - p))
print("평균 엔트로피    :", round(H.mean(), 3), "(최대 0.693 = 반반)")
print("평균 교차엔트로피:", round(CE.mean(), 3))
print("기준선           :", round((yte == int(ytr.mean() >= 0.5)).mean(), 3))
# ⑥ 숫자형: 선형회귀로 스팸 신고 수 예측
t = (df["split"] == "train").to_numpy()
reg = LinearRegression().fit(df.loc[t, ["links"]], df.loc[t, "reports"])
pred = reg.predict(df.loc[~t, ["links"]])
rmse = np.sqrt(((df.loc[~t, "reports"] - pred) ** 2).mean())
base = np.sqrt(((df.loc[~t, "reports"] - df.loc[t, "reports"].mean()) ** 2).mean())
print("신고 수 = %.2f × 링크 수 + %.2f" % (reg.coef_[0], reg.intercept_))
print("RMSE:", round(rmse, 3), "| 기준선(평균으로만 답하기) RMSE:", round(base, 3))
# ⑦ 그리기: 왼쪽 확률 지도, 오른쪽 회귀 직선
gx, gy = np.meshgrid(np.linspace(-0.5, 12.5, 150), np.linspace(-0.5, 12.5, 150))
pp = lr.predict_proba(sc.transform(np.c_[gx.ravel(), gy.ravel()]))[:, 1].reshape(gx.shape)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].contourf(gx, gy, pp, levels=20, cmap="coolwarm", alpha=0.5)
ax[0].scatter(X[tr][:, 0], X[tr][:, 1], c=ytr, cmap="coolwarm", s=14, edgecolor="k", linewidth=0.3)
ax[0].set_title("logistic: P(spam)"); ax[0].set_xlabel("links"); ax[0].set_ylabel("exclamation marks")
ax[1].scatter(df["links"], df["reports"], s=10, alpha=0.4)
ax[1].plot([0, 12], reg.predict(pd.DataFrame({"links": [0, 12]})), "r-")
ax[1].set_title("linear regression: reports"); ax[1].set_xlabel("links"); ax[1].set_ylabel("reports")
plt.tight_layout(); plt.show()
