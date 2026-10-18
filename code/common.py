# 신경망은 틀리면서 배운다 — 공통 도구 (common.py)
# 글 · 안상선 ((주)M-Robo 대표) / 라이선스: CC BY-NC 4.0
# 라이브러리: numpy, pandas, scikit-learn, matplotlib (Colab 기본 설치)
# 실행 확인: Python 3.12 · numpy 2.5 · pandas 3.0 · scikit-learn 1.9
# 하는 일: ① 메일 데이터 읽기 ② 세트(쉬운/어려운) 고르기 ③ 미리 정해 둔 학습/시험 나누기
# 데이터: data/mails.csv — 원리를 보여 주려고 만든 가상 메일 660통 (seed 0)
import os
import numpy as np
import pandas as pd

RAW = "https://raw.githubusercontent.com/adminmrobo/neural-net-learns/main/data/mails.csv"
FEATS2 = ["links", "exclaims"]            # 링크 수, 느낌표 수 (그림 평면)
FEATS3 = ["chars", "links", "exclaims"]   # 글자 수까지 (4편)
SEED = 0


def load(path=None):
    """data/mails.csv를 읽는다. 없으면 GitHub raw 주소에서 받는다."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
        local = os.path.join(here, "..", "data", "mails.csv")
        path = local if os.path.exists(local) else RAW
    return pd.read_csv(path)


def pick(df, which="easy", feats=FEATS2):
    """세트 하나를 골라 (X_train, X_test, y_train, y_test)를 돌려준다.
    나누기는 CSV의 split 열을 그대로 쓴다(메일 종류별 층화 75:25, 브라우저 실습과 같은 나누기)."""
    d = df[df["set"] == which]
    tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
    return (tr[feats].to_numpy(float), te[feats].to_numpy(float),
            tr["spam"].to_numpy(int), te["spam"].to_numpy(int))


def baseline(y_train, y_test):
    """기준선: 학습 데이터에서 더 많은 쪽 하나로만 답했을 때 시험 정확도."""
    major = int(np.mean(y_train) >= 0.5)
    return float(np.mean(y_test == major))


def standardize(X_train, X_test):
    """μ, σ는 학습 데이터로만 계산하고, 시험 데이터에는 그 값을 그대로 쓴다."""
    mu, sd = X_train.mean(0), X_train.std(0)
    sd[sd == 0] = 1
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd
