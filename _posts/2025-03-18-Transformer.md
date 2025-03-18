---
title: "Transformer"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-03-18 19:00:00 +0900
categories: [AI | 딥러닝, Architecture]
# categories: [AI | 딥러닝, Concept]
# categories: [AI | 논문 리뷰, Attention is all you need]
# categories: [MLOps | 인프라 개발, Kserve]
# categories: [Life | 일상 이야기, 와플먹으면서 공부하기]
# categories: [STEM | 수학/통계, Statistics]
tags: [DeepLearning, Transformer, Self-Attention]
description: "Transformer에 대해 자세히 알아봅시다."
image: assets/img/posts/resize/output/Transformer,_full_architecture.png # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://commons.wikimedia.org/wiki/File:Transformer,_full_architecture.png">https://commons.wikimedia.org/wiki/File:Transformer,_full_architecture.png</a></small>
</div>

## Transformer

Transformer는 2017년 구글이 "Attention is All You Need" 논문에서 제안한 모델입니다. RNN을 사용하지 않고 오직 어텐션(Attention) 메커니즘만으로 인코더-디코더 구조를 구현하여 자연어 처리 분야에 혁명을 가져왔습니다. 이 글에서는 Transformer의 구조와 원리를 이해하고, PyTorch로 직접 구현해 보겠습니다.

### 1. Transformer의 등장 배경

기존의 sequence-to-sequence 모델은 RNN, LSTM, GRU 등의 순환 신경망을 기반으로 했습니다. 이런 모델들은 다음과 같은 한계가 있었습니다.

- 장기 의존성 문제(Long-term Dependency): 시퀀스가 길어질수록 초기 정보가 손실되는 문제
- 병렬 처리 불가: 순차적 연산으로 인한 학습 속도 저하
- 제한된 정보 전달: 인코더에서 디코더로 전달되는 context vector의 정보 병목 현상

Transformer는 이러한 문제들을 해결하기 위해 RNN을 완전히 제거하고, Self-Attention 메커니즘을 도입했습니다.

### 2. Transformer의 주요 요소와 핵심 아이디어

**2.1 전체 아키텍처**

Transformer는 인코더(Encoder)와 디코더(Decoder)로 구성됩니다.

![](assets/img/posts/resize/output/transformer-2.png){: width="1000px"}

<div align="center">
  <small>Source: <a href="https://arxiv.org/abs/1706.03762">https://arxiv.org/abs/1706.03762</a></small>
</div>


- 인코더: 입력 시퀀스를 처리하여 문맥 정보를 추출
- 디코더: 인코더의 출력을 바탕으로 출력 시퀀스를 생성

각각은 여러 개의 동일한 층(layer)으로 구성됩니다. 논문에서는 인코더와 디코더를 각각 6개 층으로 구성했습니다.

**2.2 주요 하이퍼파라미터**

![](assets/img/posts/resize/output/transformer-hyperparameter.png){: width="1000px"}

<div align="center">
  <small>Source: <a href="https://arxiv.org/abs/1706.03762">https://arxiv.org/abs/1706.03762</a></small>
</div>


$$
(d_{model}: 모델의 차원, 임베딩 벡터의 차원 (논문에서는\ 512))
$$

$$
(h: 멀티 헤드 어텐션에서 헤드의 수 (논문에서는\ 8))
$$

$$
(N: 인코더/디코더 층의 개수 (논문에서는\ 6))
$$

$$
(d_{ff}: 피드 포워드 네트워크의 은닉층 크기 (논문에서는\ 2048))
$$

$$
(P_{drop}: 드롭아웃 비율 (논문에서는\ 0.1))
$$

## 3. 핵심 요소별 설명 및 코드 구현

**3.1 포지셔널 인코딩(Positional Encoding)**

RNN과 달리 Transformer는 순차적으로 입력을 처리하지 않기 때문에, 단어의 위치 정보를 별도로 제공해야 합니다. 포지셔널 인코딩은 사인(sine)과 코사인(cosine) 함수를 사용하여 각 위치에 고유한 값을 부여합니다.

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$

$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

$$
(pos: 위치\ 인덱스)
$$

$$
(i: 차원\ 인덱스)
$$

$$
(d_{model}: 모델의\ 차원)
$$


```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    포지셔널 인코딩 레이어: 입력 임베딩에 위치 정보를 추가합니다.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 포지셔널 인코딩 행렬 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 사인 함수를 짝수 인덱스에 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        # 코사인 함수를 홀수 인덱스에 적용
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 차원 확장: [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 모델 파라미터가 아닌 버퍼로 등록 (학습되지 않음)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: 입력 임베딩 [seq_len, batch_size, d_model]
        Returns:
            위치 정보가 추가된 임베딩
        """
        # 입력에 포지셔널 인코딩 더하기
        x = x + self.pe[:x.size(0), :]
        return x
```

**3.2 스케일드 닷-프로덕트 어텐션(Scaled Dot-Product Attention)**

Transformer의 핵심 연산은 스케일드 닷-프로덕트 어텐션입니다. 이 어텐션은 Query, Key, Value 세 가지 입력을 받습니다.

![](assets/img/posts/resize/output/mult-head-attention.png){: width="1000px"}

<div align="center">
  <small>Source: <a href="https://arxiv.org/abs/1706.03762">https://arxiv.org/abs/1706.03762</a></small>
</div>


$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$$
Q: Query 행렬
$$

$$
K: Key 행렬
$$

$$
V: Value 행렬
$$

$$
d_{k}: Key 벡터의 차원
$$

스케일링 인자 $$\sqrt{d_k}$$ ​​는 내적 값이 너무 커지는 것을 방지합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    스케일드 닷-프로덕트 어텐션을 계산합니다.
    
    Args:
        query: 쿼리 텐서 [batch_size, num_heads, seq_len_q, depth]
        key: 키 텐서 [batch_size, num_heads, seq_len_k, depth]
        value: 값 텐서 [batch_size, num_heads, seq_len_v, depth]
        mask: 마스킹을 위한 텐서 (옵션)
        
    Returns:
        output: 어텐션 출력 [batch_size, num_heads, seq_len_q, depth]
        attention_weights: 어텐션 가중치 [batch_size, num_heads, seq_len_q, seq_len_k]
    """
    # Q와 K의 행렬 곱으로 어텐션 스코어 계산
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    
    # 스케일링 적용
    depth = key.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(depth)
    
    # 마스킹 적용 (옵션)
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, -1e9)
    
    # 소프트맥스로 어텐션 가중치 계산
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    
    # 가중치와 Value의 곱
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

**3.3 멀티 헤드 어텐션(Multi-Head Attention)**

멀티 헤드 어텐션은 어텐션 메커니즘을 여러 번 병렬로 수행하는 방식입니다. 이를 통해 다양한 표현 공간에서 정보를 추출할 수 있습니다.

$$
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
$$

$$
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

$$
(Wi^Q​,
Wi^K​,
Wi^V​,
W^O: 학습\ 가능한\ 가중치\ 행렬)
$$

이를 통해 다양한 표현 공간에서 정보를 추출할 수 있습니다.

```python
class MultiHeadAttention(nn.Module):
    """
    멀티 헤드 어텐션 레이어: 여러 어텐션 헤드를 병렬로 계산합니다.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads  # 각 헤드의 차원
        
        # Query, Key, Value에 대한 선형 변환층
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # 출력을 위한 선형 변환층
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x, batch_size):
        """
        텐서를 여러 헤드로 분할합니다.
        
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            batch_size: 배치 크기
            
        Returns:
            분할된 텐서 [batch_size, num_heads, seq_len, depth]
        """
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, depth]
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # [batch_size, seq_len, num_heads, depth] -> [batch_size, num_heads, seq_len, depth]
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key, value, mask=None):
        """
        멀티 헤드 어텐션 계산을 수행합니다.
        
        Args:
            query: 쿼리 텐서 [batch_size, seq_len_q, d_model]
            key: 키 텐서 [batch_size, seq_len_k, d_model]
            value: 값 텐서 [batch_size, seq_len_v, d_model]
            mask: 마스킹을 위한 텐서 (옵션)
            
        Returns:
            output: 어텐션 결과 [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)
        
        # 선형 변환
        q = self.wq(query)  # [batch_size, seq_len_q, d_model]
        k = self.wk(key)    # [batch_size, seq_len_k, d_model]
        v = self.wv(value)  # [batch_size, seq_len_v, d_model]
        
        # 헤드 분할
        q = self.split_heads(q, batch_size)  # [batch_size, num_heads, seq_len_q, depth]
        k = self.split_heads(k, batch_size)  # [batch_size, num_heads, seq_len_k, depth]
        v = self.split_heads(v, batch_size)  # [batch_size, num_heads, seq_len_v, depth]
        
        # 스케일드 닷-프로덕트 어텐션 계산
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # scaled_attention: [batch_size, num_heads, seq_len_q, depth]
        
        # 헤드 결합
        # [batch_size, num_heads, seq_len_q, depth] -> [batch_size, seq_len_q, num_heads, depth]
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        
        # [batch_size, seq_len_q, num_heads, depth] -> [batch_size, seq_len_q, d_model]
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        
        # 최종 선형 변환
        output = self.wo(concat_attention)  # [batch_size, seq_len_q, d_model]
        
        return output
```

**3.4 피드 포워드 네트워크(Feed-Forward Network)**

각 인코더/디코더 층에는 2개의 선형 변환과 ReLU 활성화 함수로 구성된 피드 포워드 네트워크가 포함됩니다.

$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$

```python
class PositionwiseFeedForward(nn.Module):
    """
    포지션 와이즈 피드 포워드 네트워크: 두 개의 선형 변환과 ReLU 활성화 함수로 구성됩니다.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # 첫 번째 선형 변환
        self.linear1 = nn.Linear(d_model, d_ff)
        # 두 번째 선형 변환
        self.linear2 = nn.Linear(d_ff, d_model)
        # 드롭아웃 레이어
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model]
        Returns:
            변환된 텐서 [batch_size, seq_len, d_model]
        """
        # 첫 번째 선형 변환 후 ReLU 활성화
        x = F.relu(self.linear1(x))
        # 드롭아웃 적용
        x = self.dropout(x)
        # 두 번째 선형 변환
        x = self.linear2(x)
        return x
```

**3.5 정규화 레이어(Layer Normalization)**

각 서브레이어 후에는 층 정규화(Layer Normalization)가 적용됩니다. 이는 학습을 안정화하고 기울기 소실 문제를 완화합니다.

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

$$
(μ: 평균)
$$

$$
(σ: 표준편차)
$$


$$
(γ, β: 학습\ 가능한\ 파라미터)
$$

$$
(ϵ: 수치\ 안정성을\ 위한\ 작은\ 상수)
$$

학습을 안정화하고 기울기 소실 문제를 완화하는 역할을 합니다.

```python
# 레이어 정규화 적용 예시
layer_norm = nn.LayerNorm(d_model, eps=1e-6)
normalized_x = layer_norm(x + sublayer_output)  # 잔차 연결 및 정규화
```

**3.6 인코더 레이어(Encoder Layer)**

각 인코더 레이어는 멀티 헤드 셀프 어텐션과 피드 포워드 네트워크로 구성됩니다. 두 서브레이어 모두 잔차 연결(residual connection)과 층 정규화(layer normalization)가 적용됩니다.

$$
\text{EncoderLayer(x)}=LayerNorm(x+MultiHeadAttention(x,x,x))
$$

$$
\text{Output} = \text{LayerNorm}(\text{EncoderLayer}(x) + \text{FeedForward}(\text{EncoderLayer}(x)))
$$

```python
class EncoderLayer(nn.Module):
    """
    인코더 레이어: 멀티 헤드 셀프 어텐션과 피드 포워드 네트워크로 구성됩니다.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        # 멀티 헤드 어텐션 레이어
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        # 피드 포워드 네트워크
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 레이어 정규화
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # 드롭아웃
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: 입력 텐서 [batch_size, seq_len, d_model]
            mask: 마스킹을 위한 텐서 (옵션)
        Returns:
            출력 텐서 [batch_size, seq_len, d_model]
        """
        # 멀티 헤드 셀프 어텐션 (첫 번째 서브레이어)
        attn_output = self.mha(x, x, x, mask)
        # 드롭아웃, 잔차 연결, 레이어 정규화
        out1 = self.norm1(x + self.dropout1(attn_output))
        
        # 피드 포워드 네트워크 (두 번째 서브레이어)
        ffn_output = self.ffn(out1)
        # 드롭아웃, 잔차 연결, 레이어 정규화
        out2 = self.norm2(out1 + self.dropout2(ffn_output))
        
        return out2
```

**3.7 디코더 레이어(Decoder Layer)**

각 디코더 레이어는 세 개의 서브레이어로 구성됩니다: 마스크드 멀티 헤드 셀프 어텐션, 인코더-디코더 멀티 헤드 어텐션, 피드 포워드 네트워크. 모든 서브레이어에 잔차 연결과 층 정규화가 적용됩니다.

```python
class DecoderLayer(nn.Module):
    """
    디코더 레이어: 마스크드 멀티 헤드 셀프 어텐션, 인코더-디코더 어텐션, 피드 포워드 네트워크로 구성됩니다.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # 마스크드 멀티 헤드 셀프 어텐션
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 인코더-디코더 멀티 헤드 어텐션
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 피드 포워드 네트워크
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 레이어 정규화
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        # 드롭아웃
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """
        Args:
            x: 디코더 입력 [batch_size, seq_len, d_model]
            enc_output: 인코더 출력 [batch_size, enc_seq_len, d_model]
            look_ahead_mask: 룩-어헤드 마스크 (미래 토큰을 가리기 위한 마스크)
            padding_mask: 패딩 마스크
        Returns:
            출력 텐서 [batch_size, seq_len, d_model]
        """
        # 마스크드 멀티 헤드 셀프 어텐션 (첫 번째 서브레이어)
        self_attn_output = self.self_attn(x, x, x, look_ahead_mask)
        # 드롭아웃, 잔차 연결, 레이어 정규화
        out1 = self.norm1(x + self.dropout1(self_attn_output))
        
        # 인코더-디코더 멀티 헤드 어텐션 (두 번째 서브레이어)
        cross_attn_output = self.cross_attn(out1, enc_output, enc_output, padding_mask)
        # 드롭아웃, 잔차 연결, 레이어 정규화
        out2 = self.norm2(out1 + self.dropout2(cross_attn_output))
        
        # 피드 포워드 네트워크 (세 번째 서브레이어)
        ffn_output = self.ffn(out2)
        # 드롭아웃, 잔차 연결, 레이어 정규화
        out3 = self.norm3(out2 + self.dropout3(ffn_output))
        
        return out3
```

(작성중...)