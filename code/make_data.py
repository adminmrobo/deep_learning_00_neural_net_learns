# 신경망은 틀리면서 배운다 — 데이터 만들기 (make_data.py)
# 글 · 안상선 ((주)M-Robo 대표) / CC BY-NC 4.0
# data/mails.csv를 다시 만든다 (seed 0). 실행: python code/make_data.py
import os
import numpy as np, pandas as pd
rng=np.random.default_rng(0)
K=[("business","업무 메일",0,"easy",300,(1200,450),0.6,0.5),
   ("newsletter","뉴스레터",0,"hard",160,(4200,1100),4.2,1.4),
   ("ad_spam","광고 스팸",1,"easy",120,(650,250),6.5,7.0),
   ("phishing","피싱 스팸",1,"hard",80,(1500,500),2.6,2.2)]
rows=[]
for kind,ko,lab,st,n,(cm,cs),lm,em in K:
    got=0
    while got<n:
        c=int(round(max(150,rng.normal(cm,cs))/10)*10)
        l=int(min(12,rng.poisson(lm))); e=int(min(12,rng.poisson(em)))
        if st=="easy":
            if lab==0 and l+e>4: continue
            if lab==1 and l+e<7: continue
        rows.append(dict(kind=kind,kind_ko=ko,chars=c,links=l,exclaims=e,spam=lab,set=st)); got+=1
df=pd.DataFrame(rows)
df["split"]="train"
for kind,g in df.groupby("kind"):
    idx=g.index.to_numpy().copy(); rng.shuffle(idx)
    df.loc[idx[:len(idx)//4],"split"]="test"
df=df.sample(frac=1,random_state=0).reset_index(drop=True)
df.insert(0,"id",range(1,len(df)+1))
# 6편 회귀용: 스팸 신고 수 (기존 열은 그대로 두려고 난수를 따로 씀)
r2=np.random.default_rng(1)
df["reports"]=np.clip(np.round(0.8*df["links"]+0.35*df["exclaims"]+r2.normal(0,1.4,len(df))),0,None).astype(int)
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "mails.csv"),index=False)
print(df.groupby(["set","kind","split"]).size())
print(df.groupby("kind")[["chars","links","exclaims"]].mean().round(1))
