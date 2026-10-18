# 신경망은 틀리면서 배운다 · 6편 영역과 확률 — 정식 코드 (f6_regions_probability.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 라이브러리: numpy, scikit-learn
# 하는 일: ① 영역형(퍼셉트론·SVM)과 확률형(로지스틱) 비교 — 정확도와 교차엔트로피
#         ② 스팸이 양쪽 두 무리일 때 선형 SVM vs RBF 커널 SVC
#         ③ 숫자형: 선형회귀로 스팸 신고 수 예측 — RMSE, R², 기준선(평균으로만 답하기)
import numpy as np
from sklearn.linear_model import Perceptron, LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from common import load, pick, baseline, standardize, SEED

df = load()
entropy = lambda p: -(p * np.log(p) + (1 - p) * np.log(1 - p))

print("① 영역형 vs 확률형 (특성: 링크 수, 느낌표 수)")
for which in ["easy", "hard"]:
    Xtr, Xte, ytr, yte = pick(df, which)
    Ztr, Zte, *_ = standardize(Xtr, Xte)
    models = {"퍼셉트론": Perceptron(eta0=0.3, max_iter=100, tol=None, random_state=SEED),
              "선형 SVM": SVC(kernel="linear", C=1.0),
              "로지스틱": LogisticRegression()}
    print(f"  [{which}] 기준선 {baseline(ytr, yte):.3f}")
    for name, m in models.items():
        m.fit(Ztr, ytr)
        acc = m.score(Zte, yte)
        if hasattr(m, "predict_proba"):
            p = np.clip(m.predict_proba(Zte)[:, 1], 1e-12, 1 - 1e-12)
            ce = f"교차엔트로피 {log_loss(yte, p):.3f}, 평균 엔트로피 {entropy(p).mean():.3f}"
        else:
            wrong = int((m.predict(Zte) != yte).sum())
            ce = f"확률 없음 → 확신하고 틀린 메일 {wrong}통 (교차엔트로피 = ∞)"
        extra = f", 서포트 벡터 {m.n_support_.sum()}개" if hasattr(m, "n_support_") else ""
        print(f"    {name:6s} test {acc:.3f} | {ce}{extra}")
    sv = CalibratedClassifierCV(SVC(kernel="linear", C=1.0), method="sigmoid", ensemble=False).fit(Ztr, ytr)   # Platt 방식
    p = np.clip(sv.predict_proba(Zte)[:, 1], 1e-12, 1 - 1e-12)
    print(f"    SVM+Platt(확률 덧붙이기) 교차엔트로피 {log_loss(yte, p):.3f}")

print("\n② 스팸이 정상 메일 양쪽에 두 무리로 있을 때")
rng = np.random.default_rng(0)
ham = rng.normal([5, 5], 1.3, (200, 2))
spam = np.vstack([rng.normal([9, 9], 1.3, (50, 2)), rng.normal([1, 1], 1.3, (50, 2))])
X = np.clip(np.round(np.vstack([ham, spam])), 0, 12); y = np.r_[np.zeros(200), np.ones(100)].astype(int)
te = np.zeros(len(y), bool); te[rng.choice(200, 50, replace=False)] = True; te[200 + rng.choice(100, 25, replace=False)] = True
Ztr, Zte, *_ = standardize(X[~te], X[te])
print(f"  기준선 {baseline(y[~te], y[te]):.3f}")
for name, m in [("선형 SVM", SVC(kernel="linear")), ("RBF SVC γ=0.1", SVC(kernel="rbf", gamma=0.1)),
                ("RBF SVC γ=1", SVC(kernel="rbf", gamma=1)), ("RBF SVC γ=20", SVC(kernel="rbf", gamma=20))]:
    m.fit(Ztr, y[~te]); print(f"    {name:13s} train {m.score(Ztr, y[~te]):.3f} | test {m.score(Zte, y[te]):.3f}")

print("\n③ 숫자형: 선형회귀로 스팸 신고 수 예측 (메일 660통, split 열로 나누기)")
tr, tst = df[df.split == "train"], df[df.split == "test"]
base = np.sqrt(mean_squared_error(tst.reports, np.full(len(tst), tr.reports.mean())))
print(f"  기준선(학습 평균 {tr.reports.mean():.2f}으로만 답하기) RMSE {base:.3f}, R² 0")
for feats in [["links"], ["links", "exclaims"]]:
    m = LinearRegression().fit(tr[feats], tr.reports)
    pred = m.predict(tst[feats]); res = tr.reports - m.predict(tr[feats])
    print(f"  {feats}: 신고 수 = {' + '.join(f'{c:.2f}×{f}' for c, f in zip(m.coef_, feats))} + {m.intercept_:.2f}"
          f" | RMSE {np.sqrt(mean_squared_error(tst.reports, pred)):.3f} | R² {r2_score(tst.reports, pred):.3f} | 잔차 σ {res.std():.2f}")
