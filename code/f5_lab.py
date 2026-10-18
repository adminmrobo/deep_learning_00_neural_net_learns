# 신경망은 틀리면서 배운다 · 5편 실험실 — 정식 코드 (f5_lab.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# 하는 일: 은닉층 하나짜리 신경망에서 한 번에 하나씩만 바꾸며(활성화·표준화·학습률·특성) 시험 정확도 기록
import numpy as np, warnings
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from common import load, pick, baseline, standardize, FEATS2, FEATS3, SEED
warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = load()
BASE = dict(feats=FEATS2, act="relu", scale=True, eta=0.1, hidden=8)


def run(which, feats, act, scale, eta, hidden, seeds=(0, 1, 2, 3, 4)):
    Xtr, Xte, ytr, yte = pick(df, which, feats)
    if scale:
        Xtr, Xte, *_ = standardize(Xtr, Xte)
    s = []
    for sd in seeds:
        m = MLPClassifier((hidden,), activation=act, solver="sgd", learning_rate_init=eta,
                          momentum=0, batch_size=32, max_iter=200, random_state=sd).fit(Xtr, ytr)
        s.append(m.score(Xte, yte))
    return np.mean(s), np.std(s), baseline(ytr, yte)


for which in ["easy", "hard"]:
    print(f"\n=== {which} ===")
    rows = [("기본(ReLU, 표준화, η=0.1, 특성 2개)", BASE),
            ("활성화 → 선형", {**BASE, "act": "identity"}),
            ("활성화 → 시그모이드", {**BASE, "act": "logistic"}),
            ("표준화 끄기", {**BASE, "scale": False}),
            ("학습률 → 0.001", {**BASE, "eta": 0.001}),
            ("특성 → 글자 수 추가(3개)", {**BASE, "feats": FEATS3}),
            ("특성 3개 + 표준화 끄기", {**BASE, "feats": FEATS3, "scale": False})]
    for label, cfg in rows:
        m, s, base = run(which, cfg["feats"], cfg["act"], cfg["scale"], cfg["eta"], cfg["hidden"])
        print(f"{label:28s} test {m:.3f} ± {s:.3f} (기준선 {base:.3f})")
