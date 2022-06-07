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
