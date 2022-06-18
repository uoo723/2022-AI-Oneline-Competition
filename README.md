**목차**
- [2022-AI-Oneline-Competition](#2022-ai-oneline-competition)
    - [개요](#개요)
    - [사업화 지원](#사업화-지원)
    - [우수 기업 특전](#우수-기업-특전)
    - [문제 구성](#문제-구성)
  - [참여 과제](#참여-과제)
    - [과제 설명](#과제-설명)
    - [평가지표](#평가지표)
    - [모델 사전 조사](#모델-사전-조사)
  - [사용 방법론](#사용-방법론)
    - [Overall](#overall)
    - [monoBERT (3rd-stage retrieval)](#monobert-3rd-stage-retrieval)
    - [ColBERT (2nd-stage retrieval)](#colbert-2nd-stage-retrieval)
  - [Instruction](#instruction)
    - [디렉토리 구조](#디렉토리-구조)
      - [Python 명령 스크립트](#python-명령-스크립트)
      - [Shell 스크립트 (`./scripts`)](#shell-스크립트-scripts)
    - [Reproduction](#reproduction)
      - [데이터 전처리](#데이터-전처리)
  - [Experiments](#experiments)
  - [Log](#log)
  - [References](#references)
---

# 2022-AI-Oneline-Competition

[2022 인공지능 온라인 경진대회](https://aichallenge.or.kr/competition/detail/1)

<img width="1659" alt="image" src="https://user-images.githubusercontent.com/7765506/172270310-c9791149-44cc-41a2-b49e-5acc2f927b29.png">

### 개요

- 경진대회를 통해 우수한 기술력을 보유한 인공지능 중소 벤처 기업을 발굴하여 사업화를 지원함으로써 인공지능 기술의 활용과 확산을 촉진 하는 데에 목적이 있음.
- 기술력을 검증하고 사업화 계획 발표를 통하여 사업화 가능성을 평가하여 사업화를 지원하는 대회임.

### 사업화 지원

- 10개 과제 각 과제별 상위 4개팀 (총 40팀) 에 대해 사업화 지원 평가 기회 제공.
- 사업화 계획 제출 서류 검토, 발표평가, 경진대회 결과 등의 종합 점수로 사업화 지원 대상자 (총 20팀) 선정.

### 우수 기업 특전

- 사업화 지원 대상 : 종합 평가 상위 20팀
  - 이미지 영역 (상위 8팀), 자연어 영역 (상위 8팀), 수치해석 영역 (상위 4팀)

- 지원 내용
  - 참여한 경진대회 분야에 따른 사업화 지원, 최종 선정 기업별 최대 ***2억원*** 내외의 사업화 지원.

### 문제 구성

- 이미지 분야 (4문제), 자연어 분야 (4문제), 수치해석 분야 (2문제)로 구성 되며 10개 문제 중 1개 문제만 선택하여 대회에 참여 가능.

---

## 참여 과제

<img width="240" alt="image" src="https://user-images.githubusercontent.com/7765506/172271117-0a1a6d29-d428-4811-9f66-890b78abb6ee.png">

도서 데이터베이스 활용을 위한 문서 검색 문제

### 과제 설명

질문에 답을 할 수 있는 문서 ID를 찾는 문제

### 평가지표

**MRR@10** (Mean Reciprocal Rank)

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

$Q$: query set.  
$rank_i$: $i$-th query에 대해 relevant item이 처음으로 등장한 rank.

<img width="745" alt="image" src="https://user-images.githubusercontent.com/7765506/172272319-d9cbbbca-0de8-4fde-9460-c5e54359b48a.png">

### 모델 사전 조사

[Slide](https://docs.google.com/presentation/d/10GK0t88or1nFAZZiV91ydS3EZ5u5v7LP/edit?usp=sharing&ouid=101151189854282658970&rtpof=true&sd=true)

---

## 사용 방법론

### Overall

![그림1](https://user-images.githubusercontent.com/7765506/174442496-4cb06864-a876-4b64-87b6-7bf457bbe381.jpg)

- Multi-stage retrieval 방식 사용 [[1]](#ref1).
- 전통적으로 Bag-of-Words (TF-IDF, BM25)를 사용하는 exact matching 방식은 computation latency가 짧기 때문에 대용량 문서 corpus에서 효과적이나 vocabulary mismatch [[6](#ref6), [9](#ref9), [10](#ref10)]가 발생됨.
- 반면에 단어 간 semantic 및 문맥을 고려할 수 있는 neural 모델 같은 경우 vocabulary mismatch 문제를 완화할 수 있지만, computation 비용이 비싸기 때문에 latency가 길어지는 문제가 발생.
- 따라서 BM25와 같은 가벼운 알고리즘으로 neural 모델이 처리할 수 있는 적절한 양의 문서 candidate를 추려서 neural model에서 re-ranking 함.
- 또, re-ranking한 문서를 추려서 다시 또 다른 모델로 re-ranking할 수도 있음. 이를 multi-stage retrieval 이라고 함.

![그림2](https://user-images.githubusercontent.com/7765506/174444541-88a66eaa-158e-4f54-b791-1c0b5e9f03f0.jpg)

- Neural 모델을 사용하는 방식은 크게 2가지 분류로 나뉨.
- `Interaction-based` 같은 경우 query term, document term 간의 matching signal를 만들어 matching pattern을 학습하는 방식.
  - 즉, query, document의 score function을 학습한다고 생각하면 됨. $s_{ij} = f_{\theta}(q_i,d_j)$
- 반면에 `Representation-based` 같은 경우 query, document 각각 representation를 학습하여 비교적 가벼운 similarity function (e.g. cosine) 으로 score를 계산하게 됨.
  - $s_{ij} = sim(g_{\theta_1}(q_i),h_{\theta_2}(d_j))$
- `Interaction-based` vs. `Representation-based` [[6]](#ref6)
  - `Interaction-based`은 query-document term의 matching pattern를 학습하기 때문에 성능은 우수하나 score를 계산하기 위해 항상 query-document pair로 입력해야 하기 때문에 inference가 느림.
  - `Representation-based`은 document representation을 caching할 수 있기 때문에 inference는 `Interaction-based`보다는 빠름. 하지만 성능은 비교적 떨어짐.
- `Multi-stage Retrieval`
  - 2가지 장점을 적절히 사용하기 위해 두 방식 모두 사용하기로 함.
  - BM25 -> xxx candidates -> `Representation-based` -> xx candidates -> `Interaction-based` -> 10 candidates
  - 비교적 inference 빠른 `Representation-based` 모델에서 100 자릿수 문서 re-ranking (e.g. 500 candidates).
  - 성능을 극대화하기 위해 추려진 문서 `Interaction-based` 모델에서 10 자릿수 문서 re-ranking (e.g. 50 candidates).
  - 최종 상위 10개 문서 추출 (과제 요구사항).

### monoBERT (3rd-stage retrieval)

- 위의 `Interaction-based` 모델 그림이 ***monoBERT*** [[1]](#ref1) 모델 구조 (기존 paper에서는 2nd-stage 모델).
- Query-Document pair가 BERT [[11]](#ref11)의 input으로 들어가고, [CLS] 토큰을 MLP에 통과시켜 score를 output함.
- 따라서 loss는 multi-label 분류 모델에서 쓰는 것과 동일한 binary cross entroy를 사용함 (relevant: 1, non-relevant: 0).
- Relevant 문서는 query당 1개씩 밖에 없기 때문에 sampling이 필요없고, non-relevant 문서는 BM25에서 추출한 1,000개 문서에서 랜덤으로 sampling하여 사용함.

### ColBERT (2nd-stage retrieval)

![그림3](https://user-images.githubusercontent.com/7765506/174447759-c5fbcd05-0208-4431-80d9-6a76798c1da9.jpg)

---

## Instruction

### 디렉토리 구조

```bash
.
├── main.py
├── train.py
├── preprocess.py
├── hp_tuning.py
├── README.md
├── requirements.txt
├── Dockerfile
├── scripts
│   ├── run_preprocess.sh
│   ├── run_train.sh
│   ├── run_colbert.sh
│   ├── run_monobert.sh
│   ├── run_prediction.sh
│   ├── run_colbert_prediction.sh
│   ├── run_monobert_prediction.sh
│   └── run_sentencebert.sh  # 사용하지 않음.
└── src
    ├── __init__.py
    ├── base_trainer.py
    ├── callbacks.py
    ├── data.py
    ├── optimizers.py
    ├── utils.py
    ├── metrics.py
    ├── modules.py
    ├── colbert
    │   ├── __init__.py
    │   ├── datasets.py
    │   ├── loss.py
    │   ├── models.py
    │   └── trainer.py
    ├── monobert
    │   ├── __init__.py
    │   ├── datasets.py
    │   ├── models.py
    │   └── trainer.py
    ├── duobert          # 사용하지 않음.
    │   ├── __init__.py
    │   └── trainer.py
    └── sentencebert     # 사용하지 않음.
        ├── __init__.py
        ├── datasets.py
        ├── loss.py
        ├── models.py
        └── trainer.py
```

#### Python 명령 스크립트

- `main.py`: python 메인 명령 스크립트. `train.py`, `preprocess.py`, `hp_tuning.py` 명령을 모아 놓은 스크립트 파일.
- `train.py`: 모델 훈련 관련 명령 스크립트. `python main.py [subcommand] [args]` 형식으로 사용.
- `hp_tuning.py`: 모델 하이퍼 파라미터 튜닝 관련 명령 스크립트. `python main.py [subcommand] [args]` 형식으로 사용.
- `preprocess.py`: 전처리 관련 명령 스크립트. `python main.py [subcommand] [args]` 형식으로 사용.

```bash
$ python main.py --help
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --save-args PATH  Save command args
  --help            Show this message and exit.

Commands:
  hp-tuning            Hyper-parameter tuning       # hp_tuning.py에 정의
  make-index-contents  Make index contents          # preprocess.py에 정의
  make-shard           Make topk candidates shard   # preprocess.py에 정의
  train-colbert        Train ColBERT                # train.py에 정의
  train-duobert        Train duoBERT                # train.py에 정의
  train-monobert       Train monoBERT               # train.py에 정의
  train-sentencebert   Train sentenceBERT           # train.py에 정의

```

#### Shell 스크립트 (`./scripts`)

python 명령 스크립트 argument 관리 및 명령 파이프라인 자동화를 위해 정의.

- `run_preprocess.sh`: 데이터 전처리 파이프라인.  
  BM25 index 생성 (train, test) -> train/test query에 대해 문서 후보 1,000개 추출 후 10,000 query씩 sharding. (train: 24개, test: 1개)

- `run_train.sh`: Neural model 훈련 파이프라인.
  - `run_colbert.sh` -> `run_monobert.sh`
  - `run_colbert.sh`: ***ColBERT*** 모델 하이퍼 파라미터 설정 및 훈련 스크립트.
  - `run_monobert.sh`: ***monoBERT*** 모델 하이퍼 파라미터 설정 및 훈련 스크립트.

- `run_prediction.sh`: Test set에 대한 추론 명령 파이프라인.
  - `run_colbert_prediction.sh` -> `run_monobert_prediction.sh`
  - `run_colbert_prediction.sh`: ***ColBERT*** 모델 추론.
  - `run_monobert_prediction.sh`: ***monoBERT*** 모델 추론.

### Reproduction

※ 모든 명령 실행은 프로젝트 디렉토리에서 실행 및 실행 시간 측정을 위해 `time` command 사용.

#### 데이터 전처리

```bash
time ./scripts/run_preprocess.sh
```
---

## Experiments

| Model                                                                   | MRR@10 (Public Score) | Submission # |
|-------------------------------------------------------------------------|-----------------------|--------------|
| BM25                                                                    | 0.94637               | #1           |
| BM25 (50 candidates) + monoBERT (small)                                 | 0.98371               | #2           |
| BM25 (100 candidates) + monoBERT (small)                                | 0.98415               | #3           |
| BM25 (50 candidates) + monoBERT (base)                                  | 0.98489               | #4           |
| BM25 (50 candidates) + monoBERT (base, SWA)                             | 0.98631               | #5           |
| BM25 (50 candidates) + monoBERT (base, SWA, ENC-Ensemble)               | 0.98642               | #6           |
| BM25 (1000 candidates) + ColBERT (small, 50 candidates) + monoBERT (#6) | 0.99010               | #10          |
| BM25 (500 candidates) + ColBERT (base, 50 candidates) + monoBERT (#6)   | 0.99047               | #12          |
| BM25 (500 candidates) + ColBERT (#12) + monoBERT (#6 + all data)        | 0.99143               | #16          |

## Log

[2022.06.13]

- ColBERT 모델 구현
- ColBERT training 구현

[2022.06.12]

- SentenceBERT 모델 구현
- SentenceBERT training 구현

[2022.06.10]

- BM25 (50 candidates) + monoBERT (base) 실험
- Public score: 0.98489 (#4)
- +SWA -> Public score: 0.98631 (#5)
- +encoder ensemble -> Public score: 0.98642 (#6)

[2022.06.09]

- BM25 (50 candidates) + monoBERT (small) 실험
- Public score: 0.98371 (#2)
- BM25 (100 candidates) + monoBERT (small) 실험
- Public score: 0.98415 (#3)

[2022.06.08]

- monoBERT 모델 구현
- monoBERT 훈련 loop 구현

[2022.06.07]

- BM25 baseline 실험
- Public score: 0.94657 (#1)

---

## References

<a id="ref1">[1]</a> R. Nogueira et al. [Multi-Stage Document Ranking with BERT](https://arxiv.org/abs/1910.14424). arXiv preprint 2019.

<a id="ref2">[2]</a> O. Khattab et al. [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832). SIGIR 2020.

<a id="ref3">[3]</a> Y. Sun et al. [Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/abs/2002.10857). arXiv preprint 2020.

<a id="ref4">[4]</a> I. Loshchilov et al. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). ICLR 2019.

<a id="ref5">[5]</a> T. Jiang et al. [LightXML: Transformer with Dynamic Negative Sampling for High-Performance Extreme Multi-label Text Classification](https://arxiv.org/abs/2101.03305). AAAI 2021.

<a id="ref6">[6]</a> J. Guo et al. [A Deep Look into Neural Ranking Models for Information Retrieval](https://arxiv.org/abs/1903.06902). Information Processing & Management 57.6 (2020): 102067.

<a id="ref7">[7]</a> P. Izmailov et al. [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407). UAI 2018.

<a id="ref8">[8]</a> B. Athiwaratkun et al. [There Are Many Consistent Explanations of Unlabeled Data: Why You Should Average](https://arxiv.org/abs/1806.05594). ICLR 2019.

<a id="ref9">[9]</a> G. W. Furnas et al. [The vocabulary problem in human-system communication](https://dl.acm.org/doi/10.1145/32206.32212). Commun ACM 30 (11)
(1987) 964–971.

<a id="ref10">[10]</a> L. Zhao et al. [Term necessity prediction](https://dl.acm.org/doi/10.1145/1871437.1871474). CIKM 2010.

<a id="ref11">[11]</a> J. Devlin. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). arXiv preprint 2019.
