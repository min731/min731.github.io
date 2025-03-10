---
title: "Attention Mechanism"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-03-10 00:00:00 +0900
# categories: [AI | 딥러닝, Architecture]
categories: [AI | 딥러닝, Concept]
# categories: [AI | 논문 리뷰, Attention is all you need]
# categories: [MLOps | 인프라 개발, Kserve]
# categories: [Life | 일상 이야기, 와플먹으면서 공부하기]
# categories: [STEM | 수학/통계, Statistics]
tags: [DeepLearning, Attention]
description: "Attention Mechanism에 대해 자세히 알아봅시다."
image: assets/img/posts/resize/output/attention.png # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://commons.wikimedia.org/wiki/File:Seq2seq_RNN_encoder-decoder_with_attention_mechanism,_training_and_inferring.png">https://commons.wikimedia.org/wiki/File:Seq2seq_RNN_encoder-decoder_with_attention_mechanism,_training_and_inferring.png</a></small>
</div>

>  *본 게시글은 유튜브 [위키독스 '15-01 어텐션 메커니즘 (Attention Mechanism)'](https://wikidocs.net/22893) 자료를 참고한 점임을 알립니다.

> ## Seq2Seq with Attention

### 1. Seq2Seq의 한계점

#### 1.1 고정된 Context Vector의 문제
기존 Seq2Seq 모델은 인코더의 마지막 은닉 상태만을 context vector로 사용하여 다음과 같은 한계가 있습니다.

- 입력 시퀀스가 길어질수록 정보 손실 발생
- 초기 입력 정보가 희석되는 현상
- 긴 문장에서의 성능 저하

#### 1.2 기울기 소실 문제

- RNN 기반 구조의 고질적인 기울기 소실 문제 존재
- 장기 의존성(long-term dependency) 학습의 어려움

### 2. Attention 메커니즘의 도입

#### 2.1 핵심 아이디어
Attention은 디코더가 예측할 때마다 인코더의 모든 은닉 상태를 참조하여, 현재 예측과 연관된 부분에 더 집중하는 방식입니다.

#### 2.2 기본 구조

- Query (Q): 디코더의 현재 은닉 상태
- Key (K): 인코더의 모든 은닉 상태들
- Value (V): 인코더의 모든 은닉 상태들

### 3. Attention 메커니즘의 수학적 전개

#### 3.1 Attention Score 계산
현재 디코더의 은닉 상태와 각 인코더 은닉 상태 간의 연관성을 계산합니다.

$$
score(h_t, \bar{h}_s) = h_t^T\bar{h}_s
$$

$$
(h_t: \text{디코더의 t 시점 은닉 상태})
$$

$$
(\bar{h}_s: \text{인코더의 s 시점 은닉 상태})
$$

#### 3.2 Attention Weight 계산
Score를 확률값으로 변환합니다. **(Softmax 적용)**

$$
\alpha_{ts} = \frac{\exp(score(h_t, \bar{h}s))}{\sum{s'} \exp(score(h_t, \bar{h}_{s'}))}
$$

$$
(\alpha_{ts}: \text{t 시점 디코더가 s 시점 인코더 상태에 부여하는 가중치})
$$

#### 3.3 Context Vector 계산
가중치를 적용한 인코더 은닉 상태의 가중합을 구합니다.
$$
c_t = \sum_{s} \alpha_{ts}\bar{h}_s
$$

$$
(c_t: \text{t 시점의 컨텍스트 벡터})
$$

##### 3.4 최종 출력 계산
컨텍스트 벡터와 디코더 은닉 상태를 결합하여 최종 출력을 계산합니다.
$$
\tilde{h}_t = \tanh(W_c[c_t;h_t])
$$

$$
(W_c: \text{학습 가능한 가중치 행렬})
$$

$$
([c_t;h_t]: \text{컨텍스트 벡터와 은닉 상태의 연결(concatenation)})
$$

$$
(\tilde{h}_t: \text{t 시점의 최종 출력 벡터})
$$

### 4. Attention의 다양한 변형
Attention 메커니즘에는 여러 가지 변형이 존재하며, 각각의 장단점과 특성이 있습니다. 주요 변형들을 자세히 살펴보겠습니다.

#### 4.1 Multiplicative Attention (Luong Attention)
가장 기본적이고 단순한 형태의 어텐션으로, 두 벡터의 내적을 통해 유사도를 계산합니다.

**수식 전개**

$$
score(h_t, \bar{h}_s) = h_t^T\bar{h}_s
$$

$$
(h_t^T: \text{디코더의 t 시점 은닉 상태의 전치행렬, 차원: }1 \times d)
$$

$$
(\bar{h}_s: \text{인코더의 s 시점 은닉 상태, 차원: }d \times 1)
$$

$$
(d: \text{은닉 상태의 차원})
$$

**특징**

- 계산이 단순하고 빠름
- 메모리 효율적
- 벡터 차원이 같아야 한다는 제약 존재

#### 4.2 Scaled Dot-Product Attention
**Transformer**에서 사용되는 방식으로, 차원에 따른 스케일링을 도입하여 그래디언트 안정성을 개선했습니다.

**수식 전개**

$$
score(Q, K) = \frac{QK^T}{\sqrt{d_k}}
$$

$$
(Q: \text{Query 행렬, 차원: }n \times d_k)
$$

$$
(K: \text{Key 행렬, 차원: }m \times d_k)
$$

$$
(d_k: \text{키 벡터의 차원})
$$

$$
(\sqrt{d_k}: \text{스케일링 팩터, 그래디언트 소실/폭발 방지})
$$

**특징**

- 내적값이 너무 커지는 것을 방지
- 소프트맥스 함수의 그래디언트가 더 안정적
- 병렬 처리에 효율적

#### 4.3 Additive Attention (Bahdanau Attention)
별도의 가중치 행렬을 도입하여 더 복잡한 관계를 학습할 수 있게 만든 방식입니다.

**수식 전개**

$$
score(h_t, \bar{h}_s) = v_a^T\tanh(W_1h_t + W_2\bar{h}_s)
$$

$$
(W_1: \text{디코더 은닉 상태에 대한 가중치 행렬, 차원: }d' \times d)
$$

$$
(W_2: \text{인코더 은닉 상태에 대한 가중치 행렬, 차원: }d' \times d)
$$

$$
(v_a: \text{출력 가중치 벡터, 차원: }d' \times 1)
$$

$$
(d': \text{중간 표현의 차원, 하이퍼파라미터})
$$

**특징**

- 비선형성(tanh)을 통한 더 복잡한 관계 학습 가능
- 입력 벡터의 차원이 달라도 사용 가능
- 계산량이 상대적으로 많음

### 5. Attention 메커니즘의 장점
Attention 메커니즘은 기존 Seq2Seq 모델의 한계를 극복하고 여러 이점을 제공합니다.

#### 5.1 정보 손실 감소
동적 컨텍스트 벡터
기존 Seq2Seq와 달리 모든 인코더 은닉 상태를 활용하여 각 디코딩 시점마다 다른 컨텍스트 벡터를 생성합니다.

$$
c_t = \sum_{s=1}^{S} \alpha_{ts}\bar{h}_s
$$

$$
(c_t: \text{t 시점의 컨텍스트 벡터})
$$

$$
(\alpha_{ts}: \text{t 시점에서 s번째 입력에 대한 어텐션 가중치})
$$

$$
(\bar{h}_s: \text{s 시점의 인코더 은닉 상태})
$$

장거리 의존성 처리
입력 시퀀스 길이에 관계없이 모든 위치의 정보에 직접적으로 접근할 수 있습니다.
#### 5.2 기울기 전파 개선
직접적인 기울기 경로
각 인코더 상태에서 출력까지 직접적인 기울기 경로가 형성됩니다.

$$
\frac{\partial Loss}{\partial h_s} = \sum_{t} \frac{\partial Loss}{\partial \tilde{h}t} \frac{\partial \tilde{h}t}{\partial c_t} \frac{\partial c_t}{\partial \alpha_{ts}} \frac{\partial \alpha_{ts}}{\partial h_s}
$$

$$
(\frac{\partial Loss}{\partial h_s}: \text{인코더 은닉 상태 }h_s\text{에 대한 기울기})
$$

$$
(\frac{\partial \tilde{h}_t}{\partial c_t}: \text{컨텍스트 벡터에 대한 출력 벡터의 기울기})
$$

$$
(\frac{\partial c_t}{\partial \alpha_{ts}}: \text{어텐션 가중치에 대한 컨텍스트 벡터의 기울기})
$$

기울기 흐름 개선 효과

- 기울기 소실 문제 완화
- 더 효과적인 장기 의존성 학습
- 더 빠른 수렴 속도

#### 5.3 해석 가능성
어텐션 맵 시각화
각 디코딩 시점에서의 어텐션 가중치 분포를 시각화할 수 있습니다.

$$
A = {\alpha_{ts}} \in \mathbb{R}^{T \times S}
$$

$$
(A: \text{어텐션 맵 행렬})
$$

$$
(T: \text{출력 시퀀스 길이})
$$

$$
(S: \text{입력 시퀀스 길이})
$$

> 참고 자료

- [위키독스 '15-01 어텐션 메커니즘 (Attention Mechanism)'](https://wikidocs.net/22893)