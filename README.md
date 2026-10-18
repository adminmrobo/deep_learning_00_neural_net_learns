**[한국어](#ko) · [English](#en) · [中文](#zh) · [日本語](#ja) · [O'zbekcha](#uz) · [Русский](#ru)**

---

<a id="ko"></a>
# 신경망은 틀리면서 배운다

퍼셉트론 · 활성화 함수 · 비용 함수와 경사하강법 · 정규화 · SVM과 회귀를, 가상 메일 660통 하나로 여섯 편 내내 따라가며 배우는 인터랙티브 해설 시리즈입니다. 설치 없이 브라우저에서 버튼을 눌러 숫자를 바꾸고, 그림과 숫자가 함께 바뀌는 것을 보며 배웁니다. 각 편의 코드는 Colab에서 그대로 실행됩니다.

- 글: 안상선 ((주)M-Robo 대표)
- 라이선스: CC BY-NC 4.0 — [LICENSE.md](LICENSE.md)
- 공개 주소(예정): `https://adminmrobo.github.io/neural-net-learns/`

## 언어

이 시리즈는 여섯 언어로 볼 수 있습니다. 한국어 원본은 루트 폴더에, 각 번역판은 언어별 폴더에 같은 파일 이름으로 들어 있습니다.

| 언어 | 폴더 |
|---|---|
| 한국어 (ko) | 루트 폴더 `/` |
| 영어 (en) | `en/` |
| 중국어 간체 (zh) | `zh/` |
| 일본어 (ja) | `ja/` |
| 우즈베크어 (uz) | `uz/` |
| 러시아어 (ru) | `ru/` |

- 위쪽 메뉴의 언어 선택 상자에서 언어를 고릅니다. 고른 언어는 브라우저에 기억됩니다.
- 처음 방문하면 브라우저 언어에 맞춰 자동으로 전환됩니다.
- 주소 뒤에 `?lang=xx`(예: `?lang=en`)를 붙이면 그 언어로 강제할 수 있습니다.
- 우즈베크어(uz)와 러시아어(ru)판은 기계 번역의 도움을 받아 만들었으며, 원어민 검토를 기다리고 있습니다.

## 편 목록

| 편 | 파일 | 제목 | 유형 |
|---|---|---|---|
| 처음 화면 | [index.html](index.html) | 연구 지도 | — |
| 1 | [1_perceptron.html](1_perceptron.html) | 퍼셉트론 — 곱하고, 더하고, 틀리면 고친다 | 해설형 (확인 문제 6) |
| 2 | [2_activation.html](2_activation.html) | 활성화 함수 — 꺾여야 쌓을 수 있다 | 해설형 (확인 문제 5) |
| 3 | [3_cost_function.html](3_cost_function.html) | 비용 함수와 경사하강법 — 틀린 정도를 재고 내려간다 | 해설형 (확인 문제 6) |
| 4 | [4_normalization.html](4_normalization.html) | 정규화 — 숫자의 눈금을 맞춘다 | 해설형 (확인 문제 6) |
| 5 | [5_lab.html](5_lab.html) | 실험실 — 다이얼을 하나만 바꿔 학습시키기 | 실험실형 (실험 과제 4) |
| 6 | [6_regions_probability.html](6_regions_probability.html) | 선을 긋는 모델과 확률을 말하는 모델 — 영역·확률·숫자 | 애니메이션형 (확인 문제 4 + 강의 가이드라인) |
| 정답 | [answers.html](answers.html) | 교육자용 정답 (#a1 … #a6) | — |

## 폴더

```
├── index.html                 처음 화면 (연구 지도)
├── 1_perceptron.html … 6_regions_probability.html   편 1~6 (한국어 원본)
├── 404.html                   GitHub Pages 404
├── answers.html               교육자용 정답 (#a1 … #a6)
├── en/ zh/ ja/ uz/ ru/        (same file names / 같은 파일 이름)
├── img/                       og.jpg (share preview), icon.png, apple-touch-icon.png
├── code/                      정식 코드: common.py, f1~f6
│   ├── make_data.py           데이터 다시 만들기
│   └── easy/                  초보용 코드: e1~e6 (def·for·들여쓰기 없음)
├── data/mails.csv             가상 메일 660통
├── prompts/                   AI 튜터 프롬프트 (편마다 하나)
├── README.md
├── LICENSE.md
└── .nojekyll
```

## 데이터

`data/mails.csv` — 원리를 보여 주려고 만든 **가상 메일 660통** (numpy seed 0).

| 열 | 뜻 |
|---|---|
| id | 번호 |
| kind / kind_ko | business 업무 메일 · newsletter 뉴스레터 · ad_spam 광고 스팸 · phishing 피싱 스팸 |
| chars | 글자 수 |
| links | 링크 수 (0~12로 자름) |
| exclaims | 느낌표 수 (0~12로 자름) |
| spam | 정답 (1 = 스팸, 0 = 정상) |
| reports | 스팸 신고 수 (6편 회귀용, 0.8×링크 + 0.35×느낌표 + 잡음, 반올림·0 이상) |
| set | easy = 업무 메일 vs 광고 스팸 (420통) · hard = 뉴스레터 vs 피싱 스팸 (240통) |
| split | train / test — 메일 종류별로 4분의 1을 시험으로 뗀 층화 나누기 |

- 쉬운 세트는 링크·느낌표 평면에서 직선 하나로 나뉘도록 만들었습니다(정상은 링크+느낌표 ≤ 4, 스팸은 ≥ 7). 어려운 세트는 두 특성에서 겹치고, 글자 수로는 갈라집니다.
- 기준선(학습 데이터에서 더 많은 쪽으로만 답한 시험 정확도): 쉬운 세트 0.714, 어려운 세트 0.667.
- 원본 묶음(증강)이 없는 데이터라 그룹 분할은 필요 없습니다. 브라우저 실습과 Python 코드가 **같은 split 열**을 쓰므로 나누기가 같습니다.
- 만든 방법: `code/make_data.py`를 실행하면 똑같이 다시 만들어집니다. 종류마다 글자 수는 정규분포, 링크·느낌표는 포아송분포에서 뽑았습니다(평균: 업무 1,200자·0.6·0.5 / 뉴스레터 4,200자·4.2·1.4 / 광고 스팸 650자·6.5·7.0 / 피싱 1,500자·2.6·2.2).

---
<a id="en"></a>
# Neural Networks Learn by Being Wrong

An interactive explainer series that teaches the perceptron, activation functions, cost functions and gradient descent, normalization, and SVM and regression by following one set of 660 fictional emails through all six parts. Nothing to install: in your browser you press buttons, change numbers, and watch the pictures and the numbers change together. The code for each part runs as is in Colab.

- Author: Ahn Sangsun (CEO, M-Robo Inc.)
- License: CC BY-NC 4.0 — [LICENSE.md](LICENSE.md)
- Public address (planned): `https://adminmrobo.github.io/neural-net-learns/`

## Languages

The series is available in six languages. The Korean original is in the root folder, and each translation has its own folder with the same file names.

| Language | Folder |
|---|---|
| Korean (ko) | root folder `/` |
| English (en) | `en/` |
| Chinese, Simplified (zh) | `zh/` |
| Japanese (ja) | `ja/` |
| Uzbek (uz) | `uz/` |
| Russian (ru) | `ru/` |

- Choose a language with the language box in the top menu. Your choice is remembered in the browser.
- On your first visit, the page switches automatically to your browser language.
- To force a language, add `?lang=xx` to the address (for example `?lang=en`).
- The Uzbek (uz) and Russian (ru) versions are machine-assisted and are awaiting review by native speakers.

## Parts

| Part | File | Title | Type |
|---|---|---|---|
| Home | [index.html](en/index.html) | Series map | — |
| 1 | [1_perceptron.html](en/1_perceptron.html) | Perceptron — multiply, add, and fix it when wrong | Explainer (6 check questions) |
| 2 | [2_activation.html](en/2_activation.html) | Activation functions — you need a bend to stack layers | Explainer (5 check questions) |
| 3 | [3_cost_function.html](en/3_cost_function.html) | Cost functions and gradient descent — measure how wrong, then go downhill | Explainer (6 check questions) |
| 4 | [4_normalization.html](en/4_normalization.html) | Normalization — put numbers on the same scale | Explainer (6 check questions) |
| 5 | [5_lab.html](en/5_lab.html) | Lab — train while changing just one dial | Lab (4 lab assignments) |
| 6 | [6_regions_probability.html](en/6_regions_probability.html) | Models that draw lines and models that give probabilities — regions, probability, numbers | Animated (4 check questions + lecture guidelines) |
| Answers | [answers.html](en/answers.html) | Answer key for educators (#a1 … #a6) | — |

## Folders

```
├── index.html                 Home (Series map)
├── 1_perceptron.html … 6_regions_probability.html   Parts 1–6 (Korean original)
├── answers.html               Answer key for educators (#a1 … #a6)
├── en/ zh/ ja/ uz/ ru/        (same file names / 같은 파일 이름)
├── img/                       og.jpg (share preview), icon.png, apple-touch-icon.png
├── code/                      Standard code: common.py, f1~f6
│   ├── make_data.py           Rebuild the data
│   └── easy/                  Beginner code: e1~e6 (no def, for, or indentation)
├── data/mails.csv             660 fictional emails
├── prompts/                   AI tutor prompts (one per part)
├── README.md
├── LICENSE.md
└── .nojekyll
```

## Data

`data/mails.csv` — **660 fictional emails** made to show the principles (numpy seed 0).

| Column | Meaning |
|---|---|
| id | Number |
| kind / kind_ko | business (work email) · newsletter (newsletter) · ad_spam (ad spam) · phishing (phishing spam) |
| chars | Number of characters |
| links | Number of links (capped to 0~12) |
| exclaims | Number of exclamation marks (capped to 0~12) |
| spam | Correct label (1 = spam, 0 = normal) |
| reports | Number of spam reports (for regression in Part 6; 0.8×links + 0.35×exclamation marks + noise, rounded, 0 or more) |
| set | easy = work email vs ad spam (420 emails) · hard = newsletter vs phishing spam (240 emails) |
| split | train / test — a stratified split that sets aside one quarter of each email type for testing |

- The easy set is built so that one straight line separates it in the links–exclamation marks plane (normal: links + exclamation marks ≤ 4, spam: ≥ 7). The hard set overlaps on those two features but separates by number of characters.
- Baseline (test accuracy when always answering the class that is more common in the training data): easy set 0.714, hard set 0.667.
- The data has no original groups (no augmentation), so a group split is not needed. The browser exercises and the Python code use the **same split column**, so the split is the same.
- How it was made: running `code/make_data.py` rebuilds exactly the same data. For each type, the number of characters is drawn from a normal distribution, and links and exclamation marks from a Poisson distribution (means: work email 1,200 characters·0.6·0.5 / newsletter 4,200 characters·4.2·1.4 / ad spam 650 characters·6.5·7.0 / phishing 1,500 characters·2.6·2.2).

---

<a id="zh"></a>
# 神经网络在犯错中学习

这是一套交互式讲解系列：用同一组 660 封虚构邮件贯穿全部六篇，学习感知机、激活函数、代价函数与梯度下降、归一化、SVM 与回归。无需安装，在浏览器里按按钮、改数字，看着图形和数字一起变化来学习。每一篇的代码都可以在 Colab 中原样运行。

- 作者：安相善（M-Robo 株式会社 代表）
- 许可证：CC BY-NC 4.0 — [LICENSE.md](LICENSE.md)
- 公开地址（计划）：`https://adminmrobo.github.io/neural-net-learns/`

## 语言

本系列现有六种语言。韩语原版位于根文件夹，各译本放在各自的文件夹中，文件名相同。

| 语言 | 文件夹 |
|---|---|
| 韩语（ko） | 根文件夹 `/` |
| 英语（en） | `en/` |
| 简体中文（zh） | `zh/` |
| 日语（ja） | `ja/` |
| 乌兹别克语（uz） | `uz/` |
| 俄语（ru） | `ru/` |

- 用顶部菜单中的语言选择框切换语言，选择会保存在浏览器中。
- 第一次访问时，会根据浏览器语言自动切换。
- 在地址后加上 `?lang=xx`（例如 `?lang=zh`）可以强制使用某种语言。
- 乌兹别克语（uz）和俄语（ru）版本为机器辅助翻译，正在等待母语者审校。

## 篇目

| 篇 | 文件 | 标题 | 类型 |
|---|---|---|---|
| 首页 | [index.html](zh/index.html) | 研究地图 | — |
| 1 | [1_perceptron.html](zh/1_perceptron.html) | 感知机 — 相乘、相加，错了就改 | 讲解型（练习题 6） |
| 2 | [2_activation.html](zh/2_activation.html) | 激活函数 — 有弯折才能叠层 | 讲解型（练习题 5） |
| 3 | [3_cost_function.html](zh/3_cost_function.html) | 代价函数与梯度下降 — 衡量错误程度，然后往下走 | 讲解型（练习题 6） |
| 4 | [4_normalization.html](zh/4_normalization.html) | 归一化 — 统一数字的刻度 | 讲解型（练习题 6） |
| 5 | [5_lab.html](zh/5_lab.html) | 实验室 — 只改一个旋钮来训练 | 实验室型（实验任务 4） |
| 6 | [6_regions_probability.html](zh/6_regions_probability.html) | 画线的模型与说概率的模型 — 区域·概率·数值 | 动画型（练习题 4 + 讲课指南） |
| 答案 | [answers.html](zh/answers.html) | 教师用答案（#a1 … #a6） | — |

## 文件夹

```
├── index.html                 首页（研究地图）
├── 1_perceptron.html … 6_regions_probability.html   第1~6篇（韩语原版）
├── answers.html               教师用答案（#a1 … #a6）
├── en/ zh/ ja/ uz/ ru/        (same file names / 같은 파일 이름)
├── img/                       og.jpg (share preview), icon.png, apple-touch-icon.png
├── code/                      正式代码：common.py, f1~f6
│   ├── make_data.py           重新生成数据
│   └── easy/                  入门代码：e1~e6（没有 def·for·缩进）
├── data/mails.csv             660 封虚构邮件
├── prompts/                   AI 辅导提示词（每篇一个）
├── README.md
├── LICENSE.md
└── .nojekyll
```

## 数据

`data/mails.csv` — 为展示原理而制作的 **660 封虚构邮件**（numpy seed 0）。

| 列 | 含义 |
|---|---|
| id | 编号 |
| kind / kind_ko | business 工作邮件 · newsletter 新闻简报 · ad_spam 广告垃圾邮件 · phishing 钓鱼垃圾邮件 |
| chars | 字数 |
| links | 链接数（截断为 0~12） |
| exclaims | 感叹号数（截断为 0~12） |
| spam | 正确答案（1 = 垃圾邮件，0 = 正常邮件） |
| reports | 垃圾邮件举报数（用于第6篇回归，0.8×链接 + 0.35×感叹号 + 噪声，四舍五入，不小于 0） |
| set | easy = 工作邮件 vs 广告垃圾邮件（420 封）· hard = 新闻简报 vs 钓鱼垃圾邮件（240 封） |
| split | train / test — 按邮件类别各取四分之一作为测试的分层划分 |

- 简单组在链接·感叹号平面上可以用一条直线分开（正常邮件：链接+感叹号 ≤ 4，垃圾邮件：≥ 7）。困难组在这两个特征上重叠，但用字数可以分开。
- 基准线（只回答训练数据中较多的一类时的测试准确率）：简单组 0.714，困难组 0.667。
- 这份数据没有原始分组（增强），因此不需要分组划分。浏览器练习和 Python 代码使用**同一个 split 列**，所以划分完全相同。
- 生成方法：运行 `code/make_data.py` 即可完全一样地重新生成。每类邮件的字数取自正态分布，链接·感叹号取自泊松分布（平均值：工作邮件 1,200字·0.6·0.5 / 新闻简报 4,200字·4.2·1.4 / 广告垃圾邮件 650字·6.5·7.0 / 钓鱼 1,500字·2.6·2.2）。

---

<a id="ja"></a>
# ニューラルネットは間違えながら学ぶ

パーセプトロン・活性化関数・コスト関数と勾配降下法・正規化・SVMと回帰を、架空のメール660通ひとつで全6編を通して追いながら学ぶインタラクティブ解説シリーズである。インストールは不要で、ブラウザでボタンを押して数字を変え、図と数字が一緒に変わる様子を見ながら学ぶ。各編のコードはColabでそのまま実行できる。

- 文：アン・サンソン（株式会社M-Robo 代表）
- ライセンス：CC BY-NC 4.0 — [LICENSE.md](LICENSE.md)
- 公開アドレス（予定）：`https://adminmrobo.github.io/neural-net-learns/`

## 言語

本シリーズは6言語で公開している。韓国語の原本はルートフォルダにあり、各翻訳版は言語ごとのフォルダに同じファイル名で置いている。

| 言語 | フォルダ |
|---|---|
| 韓国語（ko） | ルートフォルダ `/` |
| 英語（en） | `en/` |
| 中国語・簡体字（zh） | `zh/` |
| 日本語（ja） | `ja/` |
| ウズベク語（uz） | `uz/` |
| ロシア語（ru） | `ru/` |

- 言語は上部メニューの言語選択ボックスで選ぶ。選んだ言語はブラウザに記憶される。
- 初めて訪れたときは、ブラウザの言語に合わせて自動で切り替わる。
- アドレスに `?lang=xx`（例：`?lang=ja`）を付けると、その言語に固定できる。
- ウズベク語（uz）とロシア語（ru）は機械翻訳を補助に使ったもので、ネイティブによるレビュー待ちである。

## 編の一覧

| 編 | ファイル | タイトル | タイプ |
|---|---|---|---|
| トップページ | [index.html](ja/index.html) | 研究マップ | — |
| 1 | [1_perceptron.html](ja/1_perceptron.html) | パーセプトロン — 掛けて、足して、間違えたら直す | 解説型（確認問題 6） |
| 2 | [2_activation.html](ja/2_activation.html) | 活性化関数 — 折れ曲がるから積み重ねられる | 解説型（確認問題 5） |
| 3 | [3_cost_function.html](ja/3_cost_function.html) | コスト関数と勾配降下法 — 間違いの大きさを測って下っていく | 解説型（確認問題 6） |
| 4 | [4_normalization.html](ja/4_normalization.html) | 正規化 — 数字の目盛りをそろえる | 解説型（確認問題 6） |
| 5 | [5_lab.html](ja/5_lab.html) | 実験室 — ダイヤルを1つだけ変えて学習させる | 実験室型（実験課題 4） |
| 6 | [6_regions_probability.html](ja/6_regions_probability.html) | 線を引くモデルと確率を語るモデル — 領域・確率・数値 | アニメーション型（確認問題 4 + 講義ガイドライン） |
| 解答 | [answers.html](ja/answers.html) | 教育者向け解答（#a1 … #a6） | — |

## フォルダ

```
├── index.html                 トップページ（研究マップ）
├── 1_perceptron.html … 6_regions_probability.html   第1～6編（韓国語原本）
├── answers.html               教育者向け解答（#a1 … #a6）
├── en/ zh/ ja/ uz/ ru/        (same file names / 같은 파일 이름)
├── img/                       og.jpg (share preview), icon.png, apple-touch-icon.png
├── code/                      正式コード：common.py, f1~f6
│   ├── make_data.py           データを作り直す
│   └── easy/                  初心者向けコード：e1~e6（def・for・インデントなし）
├── data/mails.csv             架空のメール660通
├── prompts/                   AIチューター用プロンプト（各編に1つ）
├── README.md
├── LICENSE.md
└── .nojekyll
```

## データ

`data/mails.csv` — 原理を見せるために作った**架空のメール660通**（numpy seed 0）。

| 列 | 意味 |
|---|---|
| id | 番号 |
| kind / kind_ko | business 業務メール · newsletter ニュースレター · ad_spam 広告スパム · phishing フィッシングスパム |
| chars | 文字数 |
| links | リンク数（0~12に切り詰め） |
| exclaims | 感嘆符の数（0~12に切り詰め） |
| spam | 正解（1 = スパム、0 = 正常） |
| reports | スパム報告数（第6編の回帰用、0.8×リンク + 0.35×感嘆符 + ノイズ、四捨五入・0以上） |
| set | easy = 業務メール vs 広告スパム（420通） · hard = ニュースレター vs フィッシングスパム（240通） |
| split | train / test — メールの種類ごとに4分の1をテスト用に分けた層化分割 |

- やさしいセットは、リンク・感嘆符の平面で直線1本で分けられるように作った（正常はリンク+感嘆符 ≤ 4、スパムは ≥ 7）。むずかしいセットはこの2つの特徴量では重なり、文字数で分かれる。
- ベースライン（学習データで多いほうのクラスだけを答えたときのテスト正解率）：やさしいセット 0.714、むずかしいセット 0.667。
- 元のまとまり（拡張）がないデータなので、グループ分割は必要ない。ブラウザの実習とPythonコードが**同じsplit列**を使うので、分け方は同じである。
- 作り方：`code/make_data.py` を実行すると、まったく同じデータが作り直される。種類ごとに、文字数は正規分布、リンク・感嘆符はポアソン分布から引いた（平均：業務メール 1,200字・0.6・0.5 / ニュースレター 4,200字・4.2・1.4 / 広告スパム 650字・6.5・7.0 / フィッシング 1,500字・2.6・2.2）。

---

<a id="uz"></a>
# Neyron tarmoqlar xato qilib o'rganadi

Bu — perseptron, faollashtirish funksiyalari, xarajat funksiyasi va gradient tushish, normallashtirish, SVM va regressiyani oltita qism davomida 660 ta xayoliy xatdan iborat bitta to'plam orqali o'rgatadigan interaktiv izohli turkum. Hech narsa o'rnatish shart emas: brauzerda tugmalarni bosib, sonlarni o'zgartirasiz va rasm bilan sonlar birga o'zgarishini ko'rib o'rganasiz. Har bir qismning kodi Colab'da o'zgarishsiz ishlaydi.

- Muallif: An Sangson (M-Robo kompaniyasi rahbari)
- Litsenziya: CC BY-NC 4.0 — [LICENSE.md](LICENSE.md)
- Ochiq manzil (rejada): `https://adminmrobo.github.io/neural-net-learns/`

## Tillar

Turkum oltita tilda mavjud. Koreyscha asl nusxa ildiz papkada, har bir tarjima esa o'z papkasida, xuddi shu fayl nomlari bilan joylashgan.

| Til | Papka |
|---|---|
| Koreys tili (ko) | ildiz papka `/` |
| Ingliz tili (en) | `en/` |
| Xitoy tili, soddalashtirilgan (zh) | `zh/` |
| Yapon tili (ja) | `ja/` |
| O'zbek tili (uz) | `uz/` |
| Rus tili (ru) | `ru/` |

- Tilni yuqori menyudagi til tanlash qutisidan tanlang. Tanlovingiz brauzerda eslab qolinadi.
- Birinchi kirganingizda sahifa brauzeringiz tiliga avtomatik o'tadi.
- Tilni majburan tanlash uchun manzilga `?lang=xx` qo'shing (masalan, `?lang=uz`).
- O'zbekcha (uz) va ruscha (ru) versiyalar mashina yordamida tarjima qilingan va ona tili egalarining tekshiruvini kutmoqda.

## Qismlar ro'yxati

| Qism | Fayl | Sarlavha | Turi |
|---|---|---|---|
| Bosh sahifa | [index.html](uz/index.html) | Tadqiqot xaritasi | — |
| 1 | [1_perceptron.html](uz/1_perceptron.html) | Perseptron — ko'paytiradi, qo'shadi, xato qilsa tuzatadi | Izohli (tekshiruv savollari: 6) |
| 2 | [2_activation.html](uz/2_activation.html) | Faollashtirish funksiyalari — qatlam qo'yish uchun egilish kerak | Izohli (tekshiruv savollari: 5) |
| 3 | [3_cost_function.html](uz/3_cost_function.html) | Xarajat funksiyasi va gradient tushish — xatoni o'lchab, pastga tushamiz | Izohli (tekshiruv savollari: 6) |
| 4 | [4_normalization.html](uz/4_normalization.html) | Normallashtirish — sonlarning shkalasini tenglashtiramiz | Izohli (tekshiruv savollari: 6) |
| 5 | [5_lab.html](uz/5_lab.html) | Laboratoriya — faqat bitta dastakni o'zgartirib o'qitish | Laboratoriya (tajriba topshiriqlari: 4) |
| 6 | [6_regions_probability.html](uz/6_regions_probability.html) | Chiziq chizadigan model va ehtimollik aytadigan model — soha · ehtimollik · son | Animatsiyali (tekshiruv savollari: 4 + dars bo'yicha qo'llanma) |
| Javoblar | [answers.html](uz/answers.html) | O'qituvchilar uchun javoblar (#a1 … #a6) | — |

## Papkalar

```
├── index.html                 Bosh sahifa (Tadqiqot xaritasi)
├── 1_perceptron.html … 6_regions_probability.html   1~6-qismlar (koreyscha asl nusxa)
├── answers.html               O'qituvchilar uchun javoblar (#a1 … #a6)
├── en/ zh/ ja/ uz/ ru/        (same file names / 같은 파일 이름)
├── img/                       og.jpg (share preview), icon.png, apple-touch-icon.png
├── code/                      Asosiy kod: common.py, f1~f6
│   ├── make_data.py           Ma'lumotlarni qayta yaratish
│   └── easy/                  Boshlovchilar uchun kod: e1~e6 (def, for va chekinishsiz)
├── data/mails.csv             660 ta xayoliy xat
├── prompts/                   AI repetitor uchun promptlar (har qismga bittadan)
├── README.md
├── LICENSE.md
└── .nojekyll
```

## Ma'lumotlar

`data/mails.csv` — tamoyilni ko'rsatish uchun yaratilgan **660 ta xayoliy xat** (numpy seed 0).

| Ustun | Ma'nosi |
|---|---|
| id | Raqam |
| kind / kind_ko | business ish xati · newsletter axborot byulleteni · ad_spam reklama spami · phishing fishing spami |
| chars | Belgilar soni |
| links | Havolalar soni (0~12 oralig'ida kesilgan) |
| exclaims | Undov belgilari soni (0~12 oralig'ida kesilgan) |
| spam | To'g'ri javob (1 = spam, 0 = oddiy) |
| reports | Spam shikoyatlari soni (6-qismdagi regressiya uchun; 0.8×havolalar + 0.35×undov belgilari + shovqin, yaxlitlangan, 0 dan kichik emas) |
| set | easy = ish xati vs reklama spami (420 ta xat) · hard = axborot byulleteni vs fishing spami (240 ta xat) |
| split | train / test — har bir xat turidan to'rtdan biri test uchun ajratilgan qatlamli bo'lish |

- Oson to'plam havolalar va undov belgilari tekisligida bitta to'g'ri chiziq bilan ajraladigan qilib yaratilgan (oddiy xatlarda havolalar + undov belgilari ≤ 4, spamda ≥ 7). Qiyin to'plam bu ikki belgida ustma-ust tushadi, lekin belgilar soni bo'yicha ajraladi.
- Bazaviy daraja (o'qitish ma'lumotlarida ko'proq uchraydigan sinfni javob qilganda test aniqligi): oson to'plam 0.714, qiyin to'plam 0.667.
- Ma'lumotlarda asl guruhlar (augmentatsiya) yo'q, shuning uchun guruh bo'yicha bo'lish shart emas. Brauzerdagi mashqlar va Python kodi **bir xil split ustunidan** foydalanadi, shuning uchun bo'linish bir xil.
- Qanday yaratilgan: `code/make_data.py` ni ishga tushirsangiz, aynan bir xil ma'lumotlar qayta yaratiladi. Har bir tur uchun belgilar soni normal taqsimotdan, havolalar va undov belgilari Puasson taqsimotidan olingan (o'rtacha qiymatlar: ish xati 1,200 belgi·0.6·0.5 / axborot byulleteni 4,200 belgi·4.2·1.4 / reklama spami 650 belgi·6.5·7.0 / fishing 1,500 belgi·2.6·2.2).

---

<a id="ru"></a>
# Нейросети учатся на ошибках

Интерактивная серия с объяснениями: перцептрон, функции активации, функция стоимости и градиентный спуск, нормализация, SVM и регрессия — на одном наборе из 660 вымышленных писем, который проходит через все шесть частей. Ничего не нужно устанавливать: в браузере вы нажимаете кнопки, меняете числа и смотрите, как вместе меняются рисунки и числа. Код каждой части без изменений запускается в Colab.

- Автор: Ан Сансон (генеральный директор M-Robo)
- Лицензия: CC BY-NC 4.0 — [LICENSE.md](LICENSE.md)
- Публичный адрес (планируется): `https://adminmrobo.github.io/neural-net-learns/`

## Языки

Серия доступна на шести языках. Корейский оригинал лежит в корневой папке, а каждый перевод — в своей папке с теми же именами файлов.

| Язык | Папка |
|---|---|
| Корейский (ko) | корневая папка `/` |
| Английский (en) | `en/` |
| Китайский, упрощённый (zh) | `zh/` |
| Японский (ja) | `ja/` |
| Узбекский (uz) | `uz/` |
| Русский (ru) | `ru/` |

- Язык выбирается в поле выбора языка в верхнем меню. Выбор запоминается в браузере.
- При первом посещении страница автоматически переключается на язык вашего браузера.
- Чтобы задать язык принудительно, добавьте к адресу `?lang=xx` (например, `?lang=ru`).
- Узбекская (uz) и русская (ru) версии подготовлены с помощью машинного перевода и ждут проверки носителями языка.

## Список частей

| Часть | Файл | Название | Тип |
|---|---|---|---|
| Главная | [index.html](ru/index.html) | Карта серии | — |
| 1 | [1_perceptron.html](ru/1_perceptron.html) | Перцептрон — умножить, сложить, ошибся — исправить | С объяснениями (контрольных вопросов: 6) |
| 2 | [2_activation.html](ru/2_activation.html) | Функции активации — без изгиба слои не сложить | С объяснениями (контрольных вопросов: 5) |
| 3 | [3_cost_function.html](ru/3_cost_function.html) | Функция стоимости и градиентный спуск — измерить ошибку и спускаться вниз | С объяснениями (контрольных вопросов: 6) |
| 4 | [4_normalization.html](ru/4_normalization.html) | Нормализация — приводим числа к одной шкале | С объяснениями (контрольных вопросов: 6) |
| 5 | [5_lab.html](ru/5_lab.html) | Лаборатория — обучение с изменением только одной ручки | Лаборатория (лабораторных заданий: 4) |
| 6 | [6_regions_probability.html](ru/6_regions_probability.html) | Модели, которые проводят линию, и модели, которые называют вероятность — область · вероятность · число | С анимацией (контрольных вопросов: 4 + методические указания для занятия) |
| Ответы | [answers.html](ru/answers.html) | Ответы для преподавателей (#a1 … #a6) | — |

## Папки

```
├── index.html                 Главная (карта серии)
├── 1_perceptron.html … 6_regions_probability.html   части 1~6 (корейский оригинал)
├── answers.html               Ответы для преподавателей (#a1 … #a6)
├── en/ zh/ ja/ uz/ ru/        (same file names / 같은 파일 이름)
├── img/                       og.jpg (share preview), icon.png, apple-touch-icon.png
├── code/                      Основной код: common.py, f1~f6
│   ├── make_data.py           Пересоздать данные
│   └── easy/                  Код для начинающих: e1~e6 (без def, for и отступов)
├── data/mails.csv             660 вымышленных писем
├── prompts/                   Промпты для ИИ-репетитора (по одному на часть)
├── README.md
├── LICENSE.md
└── .nojekyll
```

## Данные

`data/mails.csv` — **660 вымышленных писем**, созданных, чтобы показать принципы (numpy seed 0).

| Столбец | Значение |
|---|---|
| id | Номер |
| kind / kind_ko | business — рабочее письмо · newsletter — рассылка · ad_spam — рекламный спам · phishing — фишинговый спам |
| chars | Число символов |
| links | Число ссылок (обрезано до 0~12) |
| exclaims | Число восклицательных знаков (обрезано до 0~12) |
| spam | Правильный ответ (1 = спам, 0 = обычное письмо) |
| reports | Число жалоб на спам (для регрессии в части 6; 0.8×ссылки + 0.35×восклицательные знаки + шум, округлено, не меньше 0) |
| set | easy = рабочее письмо vs рекламный спам (писем: 420) · hard = рассылка vs фишинговый спам (писем: 240) |
| split | train / test — стратифицированное разбиение: от каждого типа писем четверть отложена для теста |

- Простой набор устроен так, что на плоскости «ссылки — восклицательные знаки» его разделяет одна прямая (обычные: ссылки + восклицательные знаки ≤ 4, спам: ≥ 7). Сложный набор по этим двум признакам перекрывается, но разделяется по числу символов.
- Базовый уровень (точность на тестовых данных, если всегда отвечать классом, которого больше в обучающих данных): простой набор 0.714, сложный набор 0.667.
- В данных нет исходных групп (аугментации), поэтому разбиение по группам не нужно. Упражнения в браузере и код на Python используют **один и тот же столбец split**, поэтому разбиение совпадает.
- Как созданы: если запустить `code/make_data.py`, получатся точно такие же данные. Для каждого типа число символов взято из нормального распределения, а ссылки и восклицательные знаки — из распределения Пуассона (средние: рабочее письмо 1,200 символов·0.6·0.5 / рассылка 4,200 символов·4.2·1.4 / рекламный спам 650 символов·6.5·7.0 / фишинг 1,500 символов·2.6·2.2).
