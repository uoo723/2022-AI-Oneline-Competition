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
    - [BM25 (1st-stage retrieval)](#bm25-1st-stage-retrieval)
    - [Neural Model Fine-tuning](#neural-model-fine-tuning)
  - [Instruction](#instruction)
    - [디렉토리 구조](#디렉토리-구조)
      - [Python 명령 스크립트](#python-명령-스크립트)
      - [Shell 스크립트 (`./scripts`)](#shell-스크립트-scripts)
      - [Log](#log)
      - [Submission](#submission)
      - [Neural model 네트워크 및 학습 코드](#neural-model-네트워크-및-학습-코드)
    - [Reproduction](#reproduction)
      - [실험 환경](#실험-환경)
      - [데이터 전처리](#데이터-전처리)
      - [Neural 모델 훈련](#neural-모델-훈련)
      - [Test 추론](#test-추론)
      - [`best_submission.csv` 재현](#best_submissioncsv-재현)
      - [모델 실험 `RUN_ID`](#모델-실험-run_id)
  - [Experiments](#experiments)
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
  - 즉, query, document의 score function을 학습한다고 생각하면 됨.
  - $s_{ij} = f_{\theta}(q_i,d_j)$
- 반면에 `Representation-based` 같은 경우 query, document 각각 representation를 학습하여 비교적 가벼운 similarity function (e.g. cosine) 으로 score를 계산하게 됨.
  - $s_{ij} = sim(g_{\theta_1}(q_i),h_{\theta_2}(d_j))$
- `Interaction-based` vs. `Representation-based` [[6]](#ref6)
  - `Interaction-based`은 query-document term의 matching pattern를 학습하기 때문에 성능은 우수하나 score를 계산하기 위해 항상 query-document pair로 입력해야 해서 inference가 느림.
  - `Representation-based`은 document representation을 caching할 수 있기 때문에 inference는 `Interaction-based`보다는 빠름. 하지만 성능은 비교적 떨어짐.
- Multi-stage Retrieval
  - 2가지 장점을 적절히 사용하기 위해 두 방식 모두 사용하기로 함.
  - BM25 -> xxx candidates -> `Representation-based` -> xx candidates -> `Interaction-based` -> 10 candidates
  - 비교적 inference 빠른 `Representation-based` 모델에서 100 자릿수 문서 re-ranking (e.g. 500 candidates).
  - 성능을 극대화하기 위해 추려진 문서를 `Interaction-based` 모델에서 10 자릿수 문서 re-ranking (e.g. 50 candidates).
  - 최종 상위 10개 문서 추출 (과제 요구사항).

### monoBERT (3rd-stage retrieval)

- 위의 `Interaction-based` 모델 그림이 ***monoBERT*** [[1]](#ref1) 모델 구조 (기존 paper에서는 2nd-stage 모델).
- Query-Document pair가 BERT [[11]](#ref11)의 input으로 들어가고, [CLS] 토큰을 MLP에 통과시켜 score를 output함.
- 따라서 loss는 multi-label 분류 모델에서 쓰는 것과 동일한 binary cross entroy를 사용함 (relevant: 1, non-relevant: 0).
- Relevant 문서는 query당 1개씩 밖에 없기 때문에 sampling이 필요없고, non-relevant 문서는 BM25에서 추출한 1,000개 문서에서 랜덤으로 sampling하여 사용함.

### ColBERT (2nd-stage retrieval)

![그림3](https://user-images.githubusercontent.com/7765506/174447759-c5fbcd05-0208-4431-80d9-6a76798c1da9.jpg)

- `Representation-based` 모델 [[2]](#ref2).
- 마지막 output layer에서 `Interaction-based`를 흉내내기 위해 late interaction을 수행하는 연산 구간이 있음.
  - 단순히 pooling layer로 representation을 만드는 SentenceBERT [[12]](#ref12)보다 더 우수한 성능을 보였음.
  - $s_{ij}=\Sigma_{\mathbf{e_{q_i}}\in\mathbf{E_{q_i}}}\max_{\mathbf{e_{d_j}}\in\mathbf{E_{d_j}}}{\mathbf{e_{q_i}}\cdot\mathbf{e_{d_j}}^\top}$
  - Sequence embedding을 normalize하여 사용하면 cosine similarity.
- Loss는 triplet loss 중 하나인 Circle loss [[3]](#ref3) 사용.
  - 각 single similarity score에 대해 서로 다른 weight 부여하여 최적화함.
  - e.g. positive sample에 대해 similarity score가 이미 높다면 weight을 줄이고, 낮으면 weight을 높여서 빠르게 수렴할 수 있도록 함.

### BM25 (1st-stage retrieval)

- 1st-stage retrieval에서는 IR 연구에 자주 사용되는 toolkit인 Anserini [[13]](#ref13)의 python wrapper인 Pyserini [[14]](#ref14)를 사용.
- Anserini는 자바로 구현된 검색 라이브러리 [Apache Lucene](https://ko.wikipedia.org/wiki/%EC%95%84%ED%8C%8C%EC%B9%98_%EB%A3%A8%EC%94%AC) 위에서 만들어진 toolkit이고, 문서 index 생성 및 검색 기능을 제공함.
- Pyserini를 이용하여 train/test 문서의 index를 생성하고, 각 query별 1,000개의 문서 후보를 추출하였음.

### Neural Model Fine-tuning

- Optimizer는 AdamW [[4]](#ref4) 사용.
- Pretrained weight은 ELECTRA [[15]](#ref15) 기반인 [KoELECTRA](https://github.com/monologg/KoELECTRA) 사용.
- Stochastic Weight Averaging [[7](#ref7), [8](#ref8)] 적용.
  - 특정 주기의 validation step에서의 모델들의 weight을 평균내어 해당 weight을 사용하는 기법.
  - 즉 k번의 validation step이 있다면 k개 모델을 ensemble하는 효과.
  - 모델의 일반화 성능을 높여줌.
- Multi Layer Text Representation [[5]](#ref5)
  - BERT의 각 encoder output을 활용하여 text representation을 만드는 기법.
  - 마지막 output layer와 가까운 5개의 encoder layer의 output을 concat하거나 Convolution을 사용하여 조금 더 fine-grained text representation을 만듦.

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
│   ├── run_mlflow.sh
│   ├── run_preprocess.sh
│   ├── run_train.sh
│   ├── run_colbert.sh
│   ├── run_monobert.sh
│   ├── run_prediction.sh
│   ├── run_colbert_prediction.sh
│   ├── run_monobert_prediction.sh
│   └── run_sentencebert.sh  # 사용하지 않음.
├── src
│    ├── __init__.py
│    ├── base_trainer.py
│    ├── callbacks.py
│    ├── data.py
│    ├── optimizers.py
│    ├── utils.py
│    ├── metrics.py
│    ├── modules.py
│    ├── colbert
│    │   ├── __init__.py
│    │   ├── datasets.py
│    │   ├── loss.py
│    │   ├── models.py
│    │   └── trainer.py
│    ├── monobert
│    │   ├── __init__.py
│    │   ├── datasets.py
│    │   ├── models.py
│    │   └── trainer.py
│    ├── duobert          # 사용하지 않음.
│    │   ├── __init__.py
│    │   └── trainer.py
│    └── sentencebert     # 사용하지 않음.
│        ├── __init__.py
│        ├── datasets.py
│        ├── loss.py
│        ├── models.py
│        └── trainer.py
├── data
│    ├── train.json
│    ├── test_data.json
│    └── test_questions.csv
├── logs                      # mlflow log 디렉토리
│    └── 0
│        ├── meta.yaml
│        ├── [run_id]
│        │      ├── ...
│        │      └── ...
│        └── ...
└── submissions
     ├── best_submission.csv  # best submssion 파일
     ├── submission1.csv
     ├── [run_prediction_script].sh
     ├── ...
     └── ...
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

#### Log

- 모델 weight, hyper parameter, training loss 등 log들은 하위 디렉토리 `logs`에 기록.
- log 기록을 웹으로 확인하기 위해 [MLflow](https://mlflow.org/) 서버 실행.

```bash
PORT=5050 ./scripts/run_mlflow.sh
```

- `PORT`는 임의로 지정 가능. 서버 실행 후, 브라우저로 해당 서버 주소로 접속하여 log 기록 확인.

#### Submission

- submission 파일들은 하위 디렉토리 `submissions`에 생성.
- `best_submssion.csv`: Final 리더보드 제출 파일.

#### Neural model 네트워크 및 학습 코드

- 각 `src`의 하위 디렉토리 `{model_name}/models.py`에 모델 네트워크 정의.
  - 현재 구현되어 있는 건 `colbert`, `monobert`
- `{model_name}/trainer.py`에 각 모델의 훈련 loop 및 prediction 로직이 구현되어 있음.

### Reproduction

- 모든 명령 실행은 프로젝트 디렉토리에서 실행 및 실행 시간 측정을 위해 `time` command 사용.
- Neural 모델 weight 및 training log는 하위 디렉토리 `logs`에 기록됨.
- `./data` 디렉토리가 생성되어 있다면 삭제해 주세요. `rm -rf ./data`

#### 실험 환경

- Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz x 18 cores (36 threads)
- 128GB RAM
- Nvidia RTX 2080 Ti x 1 (ColBERT)
- Nvidia RTX 3090 x 1 (monoBERT)
- Ubuntu 18.04

#### 데이터 전처리

```bash
time SOURCE_DATA_DIR=/DATA ./scripts/run_preprocess.sh  # 1.5h
```

#### Neural 모델 훈련

```bash
time ./scripts/run_train.sh  # ColBERT: ~11.9h, monoBERT: ~12.9h
```

#### Test 추론

```bash
time ./scripts/run_prediction.sh  # ColBERT: ~0.5h, monoBERT: ~2.5h
```

현재 검증용 서버에서 ColBERT prediction에서 병목현상이 일어나 기존 GCP Tesla T4 환경에서 테스트한 추론시간 대비 검증용 서버에서 10배정도 추론이 느려지는 현상이 있습니다.
병목 현상이 일어나는 부분

`src/colbert/trainer.py`의 446-462

```python
for i, (query_embed, c_doc_ids) in enumerate(zip(query_embeds, candidate_doc_ids)):
        doc_embed = (
            torch.stack([doc_embed_map[d_id]["embed"] for d_id in c_doc_ids])
            .unsqueeze(0)
            .to(device)
        )
        doc_attention_mask: torch.Tensor = (
            torch.stack([doc_embed_map[d_id]["attention_mask"] for d_id in c_doc_ids])
            .unsqueeze(0)
            .to(device)
        )
        query_attention_mask = query_inputs["attention_mask"][i : i + 1].to(device)
        scores: torch.Tensor = late_interaction(
            query_embed, doc_embed, query_attention_mask, doc_attention_mask
        )
        rank = scores.squeeze().cpu().numpy().argsort()[::-1]
        answers.append(",".join(c_doc_ids[rank][:topk]))
```

추론 시간 측정 코드

```python
import time
for i, (query_embed, c_doc_ids) in enumerate(zip(query_embeds, candidate_doc_ids)):
        start = time.time()
        doc_embed = (
            torch.stack([doc_embed_map[d_id]["embed"] for d_id in c_doc_ids])
            .unsqueeze(0)
            .to(device)
        )
        end = time.time()
        print(f"time: {(end - start) * 1000:.4f} ms")
        doc_attention_mask: torch.Tensor = (
            torch.stack([doc_embed_map[d_id]["attention_mask"] for d_id in c_doc_ids])
            .unsqueeze(0)
            .to(device)
        )
        query_attention_mask = query_inputs["attention_mask"][i : i + 1].to(device)
        scores: torch.Tensor = late_interaction(
            query_embed, doc_embed, query_attention_mask, doc_attention_mask
        )
        rank = scores.squeeze().cpu().numpy().argsort()[::-1]
        answers.append(",".join(c_doc_ids[rank][:topk]))
```

기존 GCP T4 환경에서의 `.cuda()` 속도

- Ubuntu 18.04
- n1-highmem-16 (vCPU 16, 104GB)
- Intel(R) Xeon(R) CPU @ 2.00GHz
- Telsa T4
- CUDA 11.4
- Python 3.7.13
- Pytorch 1.9.0 (cuDNN 7650)
  
![스크린샷 2022-06-23 15 40 40](https://user-images.githubusercontent.com/7765506/175232311-6d726caa-d2e5-48e9-94d2-403f56b7ce74.png)

검증용 서버 환경에서의 `.cuda()` 속도

![스크린샷 2022-06-23 14 56 38](https://user-images.githubusercontent.com/7765506/175227915-d3aacc21-55c0-4e3d-87e2-587739cda276.png)

#### `best_submission.csv` 재현

```bash
time COLBERT_RUN_ID=9d84388f242e44c289c7f459aa95bdca MONOBERT_RUN_ID=4a3cbf97ae1d4f14b0c7e4099a179c76 SUBMISSION_FILE=best.csv ./scripts/run_prediction.sh
```

`./submissions/best.csv` 생성됨.

#### 모델 실험 `RUN_ID`

- `9d84388f242e44c289c7f459aa95bdca`: final 리더보드에서 사용된 ColBERT
- `40b783df8e4b41ab84341330d42fd13f`: 검증용 서버에서 실행시킨 ColBERT
- `4a3cbf97ae1d4f14b0c7e4099a179c76`: final 리더보드에서 사용된 monoBERT
- `659e51913f5347cabb567221cb9ae398`: 검증용 서버에서 실행시킨 monoBERT

---

## Experiments

| Model                                  | MRR@10 (Public Score) | Submission # |
|----------------------------------------|-----------------------|--------------|
| BM25                                   | 0.94637               | #1           |
| BM25 (50C) + monoBERT                  | 0.98642               | #6           |
| BM25 (500C) + ColBERT (50C) + monoBERT | 0.99369               | #22          |
| BM25 (500C) + ColBERT (20C) + monoBERT | 0.99286               | #35          |

- *N*C: Top-*N* candidates
- 재현성 검증 서버 스펙 고려하여 #22 대신 #35 최종 제출.

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

<a id="ref11">[11]</a> J. Devlin et al. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). arXiv preprint 2019.

<a id="ref12">[12]</a> N. Reimers et al. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084). ENMLP 2019.

<a id="ref13">[13]</a> P. Yang et al. [Anserini: Enabling the Use of Lucene for Information Retrieval Research](https://dl.acm.org/doi/10.1145/3077136.3080721). [Github link](https://github.com/castorini/anserini). SIGIR 2017.

<a id="ref14">[14]</a> J. Lin et al. [Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations](https://dl.acm.org/doi/10.1145/3404835.3463238). [Github link](https://github.com/castorini/pyserini/). SIGIR 2021.

<a id="ref15">[15]</a> K. Clark et al. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555). ICLR 2020.

<a id="ref16">[16]</a> M. Lewis et al. [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461). ACL 2020.
