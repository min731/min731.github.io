---
title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-04-02 19:00:00 +0900
# categories: [AI | 딥러닝, Architecture]
# categories: [AI | 딥러닝, Concept]
categories: [AI | 딥러닝, Paper]
# categories: [MLOps | 인프라 개발, Kserve]
# categories: [Life | 일상 이야기, 와플먹으면서 공부하기]
# categories: [STEM | 수학/통계, Statistics]
tags: [DeepLearning, RoBERTa, BERT]
description: "RoBERTa 논문을 꼼꼼히 살펴봅시다."
image: assets/img/posts/resize/output/roberta-eval.png # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://arxiv.org/pdf/1907.11692">https://arxiv.org/pdf/1907.11692</a></small>
</div>

## RoBERTa

[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692) 내용을 자세히 살펴봅니다.

### Abstract

**[원문]**

"Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different
sizes, and, as we will show, hyperparameter
choices have significant impact on the final results. We present a replication study of BERT
pretraining (Devlin et al.
, 2019) that carefully
measures the impact of many key hyperparameters and training data size. We find that BERT
was significantly undertrained, and can match
or exceed the performance of every model
published after it. Our best model achieves
state-of-the-art results on GLUE, RACE and
SQuAD. These results highlight the importance of previously overlooked design choices,
and raise questions about the source of recently reported improvements. We release our
models and code."

**[해석]**

본 논문의 저자들은 language model pretraining이 상당한 성능 향상을 가져왔지만 서로 다른 접근 방식을 비교하는 것이 쉽지 않다고 지적합니다. 그 이유로 학습 비용이 많이 들고, 종종 다양한 크기의 비공개 데이터셋으로 진행되며, 논문에서 보여주겠지만 hyperparameter 선택이 최종 결과에 상당한 영향을 미친다는 점을 들고 있습니다.
저자들은 BERT pretraining에 대한 동일한 연구를 진행하면서 많은 주요 hyperparameter와 학습 데이터 크기의 영향을 신중히 측정했습니다. 그 결과, **BERT가 충분히 학습되지 않았음(significantly undertrained)**을 발견했으며, 이후 발표된 모든 모델의 성능과 견주거나 능가할 수 있다고 주장합니다.
저자들이 개발한 모델인 RoBERTa는 GLUE, RACE 및 SQuAD에서 SOTA를 달성했습니다. 이러한 결과는 이전에 간과된 설계 단계에서의 선택의 중요성을 강조하고, 최근 알려진 개선 사항들에 대한 의문을 제기합니다. 저자들은 모델과 코드도 공개했습니다.

### 1. Introduction

**[원문]**

"Self-training methods such as ELMo (Peters et al.
,
2018), GPT (Radford et al.
, 2018), BERT
(Devlin et al.
, 2019), XLM (Lample and Conneau
,
2019), and XLNet (Yang et al.
, 2019) have
brought significant performance gains, but it can
be challenging to determine which aspects of
the methods contribute the most. Training is
computationally expensive, limiting the amount
of tuning that can be done, and is often done with
private training data of varying sizes, limiting
our ability to measure the effects of the modeling
advances."

**[해석]**

본 논문은 ELMo, GPT, BERT, XLM, XLNet과 같은 self-training 방법들이 상당한 성능 향상을 가져왔지만, 위 모델들의 어떤 측면이 가장 큰 기여를 하는지 판단하기 어렵다고 설명합니다. 학습이 계산적으로 비용이 많이 들어 가능한 tuning이 제한되고, 종종 다양한 크기의 공개되지 않은 training data로 진행되어 모델 발전의 효과를 측정하는 능력이 제한된다는 문제를 지적합니다.

**[원문]**

"We present a replication study of BERT pretraining (Devlin et al., 2019), which includes a
careful evaluation of the effects of hyperparmeter
tuning and training set size. We find that BERT
was significantly undertrained and propose an improved recipe for training BERT models, which
we call RoBERTa, that can match or exceed the
performance of all of the post-BERT methods.
Our modifications are simple, they include: (1)
training the model longer, with bigger batches,
over more data; (2) removing the next sentence
prediction objective; (3) training on longer sequences; and (4) dynamically changing the masking pattern applied to the training data. We also
collect a large new dataset (CC-NEWS) of comparable size to other privately used datasets, to better
control for training set size effects."

**[해석]**

저자들은 BERT pretraining에 대한 동일한 연구를 제시하면서, hyperparameter tuning과 training set 크기의 효과를 신중히 평가합니다. 저자들은  **BERT가 상당히 부족하게 학습되었음(significantly undertrained)**을 발견하고, BERT 모델을 학습하기 위한 개선된 방법을 제안합니다. 이를 RoBERTa라고 부르며, BERT 이후의 모든 방법들의 성능과 견줄 수 있거나 능가할 수 있다고 주장합니다.

개선 사항으로는,

- 더 오랜 시간, 더 큰 batch size로, 더 많은 데이터에 대해 모델 학습
- Next Sentence Prediction(NSP) 제거
- 더 긴 시퀀스에 대한 학습
- 학습 데이터에 적용되는 masking 패턴을 dynamic하게 변경

또한 저자들은 다른 비공개 데이터셋과 비슷한 크기의 새로운 데이터셋(CC-NEWS)을 수집하여 training set 크기 효과를 더 잘 제어했습니다.

**[원문]**

"When controlling for training data, our improved training procedure improves upon the published BERT results on both GLUE and SQuAD.
When trained for longer over additional data, our
model achieves a score of 88.5 on the public
GLUE leaderboard, matching the 88.4 reported
by Yang et al.
(2019). Our model establishes a
new state-of-the-art on 4/9 of the GLUE tasks:
MNLI, QNLI, RTE and STS-B. We also match
state-of-the-art results on SQuAD and RACE.
Overall, we re-establish that BERT’s masked language model training objective is competitive
with other recently proposed training objectives
such as perturbed autoregressive language modeling (Yang et al., 2019).2"

**[해석]**

학습 데이터를 다룰 때, 저자들의 개선된 학습 방법는 GLUE와 SQuAD 모두에서 발표된 BERT 결과를 향상시킵니다. 추가 데이터로 더 오래 학습했을 때, RoBERTa은 GLUE 리더보드에서 88.5점을 달성하여 Yang et al. (2019)이 보고한 88.4점과 일치합니다. 또한 GLUE task의 4/9에서 새로운 state-of-the-art를 수립했습니다: MNLI, QNLI, RTE 및 STS-B. 또한 SQuAD와 RACE에서도 state-of-the-art 결과와 일치합니다. 전반적으로, BERT의 masked language model training objective가 Yang et al. (2019)의 perturbed autoregressive language modeling과 같은 최근 제안된 다른 training objective들과 경쟁력이 있음을 재확인했습니다.

**[원문]**

"In summary, the contributions of this paper
are: (1) We present a set of important BERT design choices and training strategies and introduce alternatives that lead to better downstream task
performance; (2) We use a novel dataset, CCNEWS, and confirm that using more data for pretraining further improves performance on downstream tasks; (3) Our training improvements show
that masked language model pretraining, under
the right design choices, is competitive with all
other recently published methods. We release our
model, pretraining and fine-tuning code implemented in PyTorch (Paszke et al., 2017)."

**[해석]**

요약하자면, 이 논문의 내용은 다음과 같습니다. (1) BERT 설계와 학습 전략을 제시하고 downstream task 성능을 향상시키는 대안을 소개합니다; (2) 새로운 데이터셋인 CC-NEWS를 사용하고, pretraining에 더 많은 데이터를 사용하면 downstream task의 성능이 더욱 향상된다는 것을 확인합니다. (3) 저자들의 학습 개선 방법은 적절한 설계 선택 하에서 masked language model pretraining이 최근 발표된 다른 모든 방법과 경쟁력이 있음을 보여줍니다. 저자들은 PyTorch로 구현된 모델, pretraining 및 fine-tuning 코드를 공개합니다.

### 2. Background

**[원문]**

"In this section, we give a brief overview of the BERT (Devlin et al., 2019) pretraining approach and some of the training choices that we will examine experimentally in the following section."

**[해석]**

이 섹션에서는 BERT pretraining approach과 다음 섹션에서 실험적으로 검토할 일부 학습 방법에 대한 간략한 개요를 제공합니다.

#### 2.1 Setup

**[원문]**

"BERT takes as input a concatenation of two segments (sequences of tokens), x1, . . . , xN and y1, . . . , yM. Segments usually consist of more than one natural sentence. The two segments are presented as a single input sequence to BERT with special tokens delimiting them: [CLS], x1, . . . , xN , [SEP], y1, . . . , yM, [EOS]. M and N are constrained such that M + N < T, where T is a parameter that controls the maximum sequence length during training.
The model is first pretrained on a large unlabeled text corpus and subsequently finetuned using end-task labeled data."

**[해석]**

BERT는 두 segment의 연결을 입력으로 받습니다.$$x1, . . . , x_{N}$$과 $$y1, . . . , y_{M}$$. Segment는 일반적으로 하나 이상의 자연 문장으로 구성됩니다. 두 segment는 다음과 같이 특수 token으로 구분된 단일 입력 시퀀스로 BERT에 제공됩니다.

$$[CLS], x1, . . . , xN, [SEP], y1, . . . , yM, [EOS]$$. $$M$$과 $$N$$은 $$M + N < T$$ 조건을 만족하도록 제한되며, 여기서 $$T$$는 학습 중 최대 시퀀스 길이를 제어하는 parameter입니다.
먼저 대규모 unlabeled text corpus에서 pretrain된 다음, end-task labeled data를 사용하여 fine-tuning됩니다.

#### 2.2 Architecture

**[원문]**

"BERT uses the now ubiquitous transformer architecture (Vaswani et al., 2017), which we will not
review in detail. We use a transformer architecture
with L layers. Each block uses A self-attention
heads and hidden dimension H."

**[해석]**

BERT는 현재 널리 사용되는 transformer architecture를 사용하며, 본 논문에서는 자세히 다루지 않습니다. 저자들은 L개의 layer를 가진 transformer architecture를 사용합니다. 각 block은 $$A$$개의 self-attention head와 $$H$$ 크기의 hidden dimension을 사용합니다.

#### 2.3 Training Objectives

**[원문]**

"During pretraining, BERT uses two objectives:
masked language modeling and next sentence prediction.
Masked Language Model (MLM) A random
sample of the tokens in the input sequence is
selected and replaced with the special token
[MASK]. The MLM objective is a cross-entropy
loss on predicting the masked tokens. BERT uniformly selects 15% of the input tokens for possible replacement. Of the selected tokens, 80% are
replaced with [MASK], 10% are left unchanged,
and 10% are replaced by a randomly selected vocabulary token.
In the original implementation, random masking and replacement is performed once in the beginning and saved for the duration of training, although in practice, data is duplicated so the mask
is not always the same for every training sentence
(see Section 4.1)."

**[해석]**

BERT는 사전학습 과정에서 masked language modeling과 next sentence prediction 두 가지 학습 목표를 사용합니다. Masked Language Model (MLM) 입력 시퀀스에서 일부 token을 무작위로 샘플링하여 특수 토큰인 [MASK]로 대체합니다. MLM 목표는 이렇게 마스킹된 토큰을 예측하는 cross-entropy loss입니다. BERT는 입력 토큰의 15%를 균일하게 선택하여 대체할 대상으로 삼습니다. 선택된 토큰 중 80%는 [MASK]로 대체하고, 10%는 원래 상태로 유지하며, 나머지 10%는 어휘 목록에서 무작위로 선택한 다른 토큰으로 대체합니다. 원래 BERT에서는 random 마스킹과 토크을 대체하는 작업을 학습 시작 시 한 번만 수행하고 이를 전체 학습 기간 동안 유지합니다. 그러나 실제로는 데이터를 복제하기 때문에 모든 학습 문장에 대해 항상 동일한 마스크가 적용되지는 않습니다.

**[원문]**

"Next Sentence Prediction (NSP) NSP is a binary classification loss for predicting whether two
segments follow each other in the original text.
Positive examples are created by taking consecutive sentences from the text corpus. Negative examples are created by pairing segments from different documents. Positive and negative examples
are sampled with equal probability.
The NSP objective was designed to improve
performance on downstream tasks, such as Natural
Language Inference (Bowman et al., 2015), which
require reasoning about the relationships between
pairs of sentences."

**[해석]**

Next Sentence Prediction (NSP) NSP는 두 segment가 원본 텍스트에서 서로 연속하는지 예측하기 위한 이진 분류 손실(binary classification loss)입니다. positive examples는 텍스트 말뭉치에서 연속된 문장을 가져와 생성합니다. negative examples는 서로 다른 문서에서 세그먼트를 짝지어 생성합니다. 긍정 및 부정 예제는 동일한 확률로 샘플링됩니다.
NSP 목표는 Natural Language Inference와 같이 문장 쌍 간의 관계에 대한 추론이 필요한 downstream task의 성능을 향상시키기 위해 설계되었습니다.

#### 2.4 Optimization

**[원문]**

"BERT is optimized with Adam (Kingma and Ba,
2015) using the following parameters: β1 = 0.9,
β2 = 0.999, ǫ = 1e-6 and L2 weight decay of 0.01. The learning rate is warmed up
over the first 10,000 steps to a peak value of
1e-4, and then linearly decayed. BERT trains
with a dropout of 0.1 on all layers and attention weights, and a GELU activation function (Hendrycks and Gimpel, 2016). Models are
pretrained for S = 1,000,000 updates, with minibatches containing B = 256 sequences of maximum length T = 512 tokens."

**[해석]**

BERT는 다음 매개변수를 사용하여 Adam optimizer로 최적화됩니다.

$$β_{1} = 0.9, β_{2} = 0.999, ǫ = 1e-6$$, $$L2\ weight\ decay = 0.01$$ 학습률(learning rate)은 처음 10,000 스텝 동안 $$1e^{-4}$$ 의 최대값까지 warm up된 다음 선형적으로 감소합니다. BERT는 모든 레이어와 어텐션 가중치에 $$0.1$$의 dropout을 적용하고, 활성화 함수로써 $$GELU$$를 사용합니다. 모델은 $$S = 1,000,000$$ 업데이트 동안 사전학습되며, 각 미니배치는 최대 길이 $$T = 512$$ 토큰의 $$B = 256$$ 시퀀스를 포함합니다.

#### 2.5 Data

**[원문]**

"BERT is trained on a combination of BOOKCORPUS (Zhu et al., 2015) plus English WIKIPEDIA,
which totals 16GB of uncompressed text.3"

**[해석]**

BERT는 BOOKCORPUS와 영어 WIKIPEDIA를 결합한 데이터셋으로 학습되며, 이는 압축되지 않은 텍스트로 총 16GB에 해당합니다.

### 3. Experimental Setup

**[원문]**

"In this section, we describe the experimental setup
for our replication study of BERT."

**[해석]**

이 섹션에서는 BERT replication 연구를 위한 실험 구성에 대해 설명합니다.

#### 3.1 Implementation

**[원문]**

"We reimplement BERT in FAIRSEQ (Ott et al.,
2019). We primarily follow the original BERT
3Yang et al. (2019) use the same dataset but report having
only 13GB of text after data cleaning. This is most likely due
to subtle differences in cleaning of the Wikipedia data.
optimization hyperparameters, given in Section 2,
except for the peak learning rate and number of
warmup steps, which are tuned separately for each
setting. We additionally found training to be very
sensitive to the Adam epsilon term, and in some
cases we obtained better performance or improved
stability after tuning it. Similarly, we found setting
β2 = 0.98 to improve stability when training with
large batch sizes."

**[해석]**

저자들은 FAIRSEQ를 통해 BERT를 재구현했습니다. 각 설정에 대해 별도로 조정된 최대 학습률(peak learning rate)과 warm-up 스텝 수를 제외하고는 주로 섹션 2에 제시된 원래 BERT 최적화 하이퍼파라미터를 따랐습니다. 또한 학습이 Adam optimizer의 $$\epsilon$$ 항에 매우 민감하다는 것을 발견했으며, 일부 경우에서는 이를 조정한 후 더 나은 성능이나 향상된 안정성을 얻었습니다. 마찬가지로, 큰 배치 크기로 학습할 때 $$β_{2} = 0.98$$로 설정하면 안정성이 향상된다는 것을 발견했습니다.

**[원문]**

"We pretrain with sequences of at most T = 512
tokens. Unlike Devlin et al. (2019), we do not randomly inject short sequences, and we do not train
with a reduced sequence length for the first 90% of
updates. We train only with full-length sequences.
We train with mixed precision floating point
arithmetic on DGX-1 machines, each with 8 ×
32GB Nvidia V100 GPUs interconnected by Infiniband (Micikevicius et al., 2018)."

**[해석]**

저자들은 최대 $$T = 512$$ 토큰의 시퀀스로 사전학습을 진행합니다. Devlin et al. (2019) 연구과 달리, 짧은 시퀀스를 무작위로 주입하지 않고, 업데이트의 처음 90%에 대해 감소된 시퀀스 길이로 학습하지 않습니다. 오직 전체 길이 시퀀스로만 학습합니다. 저자들은 각각 8 × 32GB Nvidia V100 GPU가 Infiniband로 상호 연결된 DGX-1 머신에서 mixed precision floating point 연산으로 학습을 수행했습니다.

#### 3.2 Data

**[원문]**

"BERT-style pretraining crucially relies on large
quantities of text. Baevski et al. (2019) demonstrate that increasing data size can result in improved end-task performance. Several efforts
have trained on datasets larger and more diverse
than the original BERT (Radford et al., 2019;
Yang et al., 2019; Zellers et al., 2019). Unfortunately, not all of the additional datasets can be
publicly released. For our study, we focus on gathering as much data as possible for experimentation, allowing us to match the overall quality and
quantity of data as appropriate for each comparison."

**[해석]**

BERT 스타일의 사전학습은 대량의 텍스트에 크게 의존합니다. Baevski et al. (2019) 연구에서는 데이터 양을 늘리면 최종 태스크 성능이 향상될 수 있음을 보여줍니다. 여러 연구들이 원래 BERT보다 더 크고 다양한 데이터셋으로 학습했습니다. 안타깝게도 모든 추가 데이터셋을 공개되지는 않았습니다. 이 연구에서 저자들은 각 비교에 적합한 데이터의 전반적인 품질과 양을 일치시키기 위해 실험을 위해 가능한 한 많은 데이터를 수집하는 데 중점을 두었습니다.

**[원문]**

"We consider five English-language corpora of
varying sizes and domains, totaling over 160GB
of uncompressed text. We use the following text
corpora:
• BOOKCORPUS (Zhu et al., 2015) plus English
WIKIPEDIA. This is the original data used to
train BERT. (16GB).
• CC-NEWS, which we collected from the English portion of the CommonCrawl News
dataset (Nagel, 2016). The data contains 63
million English news articles crawled between
September 2016 and February 2019. (76GB after filtering).4
• OPENWEBTEXT (Gokaslan and Cohen, 2019),
an open-source recreation of the WebText corpus described in Radford et al. (2019). The text
is web content extracted from URLs shared on
Reddit with at least three upvotes. (38GB).5
• STORIES, a dataset introduced in Trinh and Le
(2018) containing a subset of CommonCrawl
data filtered to match the story-like style of
Winograd schemas. (31GB)."

**[해석]**

저자들은 다양한 크기와 도메인의 영어 말뭉치 5개를 고려했으며, 압축되지 않은 텍스트로 총 160GB 이상입니다. 다음 텍스트 코퍼스를 사용했습니다.

- BOOKCORPUS, 영어 WIKIPEDIA : BERT를 학습하는 데 사용된 원래 데이터입니다. (16GB)
- CC-NEWS, CommonCrawl News 데이터셋의 영어 부분에서 수집했습니다. 이 데이터에는 2016년 9월과 2019년 2월 사이에 크롤링된 6300만 개의 영어 뉴스 기사가 포함되어 있습니다. (필터링 후 76GB)
- OPENWEBTEXT에서 설명한 WebText Corpus의 오픈 소스 재현입니다. 이 텍스트는 Reddit에서 최소 3개의 추천를 받은 URL에서 추출한 웹 콘텐츠입니다. (38GB)
- STORIES, Trinh and Le (2018)에서 소개한 데이터셋으로, Winograd schemas의 이야기 같은 스타일과 일치하도록 필터링된 CommonCrawl 데이터들을 포함합니다. (31GB)

#### 3.3 Evaluation

**[원문]**

"Following previous work, we evaluate our pretrained models on downstream tasks using the following three benchmarks."

**[해석]**

이전 연구에 이어, 저자들은 다음 세 가지 벤치마크를 사용하여 downstream task에서 사전학습된 모델을 평가합니다.

**[원문]**

"GLUE The General Language Understanding Evaluation (GLUE) benchmark (Wang et al.,
2019b) is a collection of 9 datasets for evaluating
natural language understanding systems.6 Tasks
are framed as either single-sentence classification
or sentence-pair classification tasks. The GLUE
organizers provide training and development data
splits as well as a submission server and leaderboard that allows participants to evaluate and compare their systems on private held-out test data.
For the replication study in Section 4, we report
results on the development sets after finetuning
the pretrained models on the corresponding singletask training data (i.e., without multi-task training
or ensembling). Our finetuning procedure follows
the original BERT paper (Devlin et al., 2019).
In Section 5 we additionally report test set results obtained from the public leaderboard. These
results depend on a several task-specific modifications, which we describe in Section 5.1."

**[해석]**

GLUE General Language Understanding Evaluation 벤치마크는 자연어 이해 시스템을 평가하기 위한 9개 데이터셋의 모음입니다. task는 단일 문장 분류 또는 문장 쌍 분류 태스크로 구성됩니다. GLUE 단체에서는 학습 및 개발 데이터 분할뿐만 아니라 참가자가 비공개 테스트 데이터에서 시스템을 평가하고 비교할 수 있는 제출 서버와 leaderboard를 제공합니다.

섹션 4의 replication 연구에서는 해당 단일 태스크 학습 데이터로 사전학습된 모델을 파인튜닝한 후 개발 세트에 대한 결과를 정리합니다. (즉, 다중 태스크 학습이나 앙상블 없이). 파인튜닝 절차는 기존 BERT 논문을 따릅니다.

섹션 5에서는 추가로 공개 리더보드에서 얻은 테스트 세트 결과를 보고합니다. 이러한 결과는 섹션 5.1에서 설명하는 여러 태스크별 수정 사항에 따라 달라집니다.

**[원문]**

"SQuAD The Stanford Question Answering Dataset (SQuAD) provides a paragraph of context and a question. The task is to answer the question by extracting the relevant span from the context. We evaluate on two versions of SQuAD: V1.1 and V2.0 (Rajpurkar et al., 2016, 2018). In V1.1 the context always contains an answer, whereas in V2.0 some questions are not answered in the provided context, making the task more challenging. For SQuAD V1.1 we adopt the same span prediction method as BERT (Devlin et al., 2019). For SQuAD V2.0, we add an additional binary classifier to predict whether the question is answerable, which we train jointly by summing the classification and span loss terms. During evaluation, we only predict span indices on pairs that are classified as answerable."

**[해석]**

SQuAD Stanford Question Answering Dataset(SQuAD)는 context paragraph과 question을 제공합니다. task는 문맥에서 관련 스팬을 추출하여 질문에 답하는 것입니다.

저자들은 SQuAD의 두 버전, V1.1과 V2.0에 대해 평가합니다. V1.1에서는 문맥이 항상 답변을 포함하는 반면, V2.0에서는 일부 질문이 제공된 문맥에서 답변되지 않아 태스크가 더 어렵습니다.
SQuAD V1.1의 경우 기존 BERT와 동일한 span loss 방법을 채택했습니다. SQuAD V2.0의 경우, 질문에 답할 수 있는지 예측하는 추가 이진 분류기를 추가했으며, 분류 및 span loss 항을 합산하여 공동으로 학습합니다. 평가 중에는 답변 가능으로 분류된 쌍에 대해서만 span index를 예측합니다.

**Span loss란?**

$$L_span = -log(P(start_{true})) - log(P(end_{true}))$$

**[원문]**

"RACE The ReAding Comprehension from Examinations (RACE) (Lai et al., 2017) task is a
large-scale reading comprehension dataset with
more than 28,000 passages and nearly 100,000
questions. The dataset is collected from English
examinations in China, which are designed for
middle and high school students. In RACE, each
passage is associated with multiple questions. For
every question, the task is to select one correct answer from four options. RACE has significantly
longer context than other popular reading comprehension datasets and the proportion of questions
that requires reasoning is very large."

**[해석]**

RACE ReAding Comprehension from Examinations(RACE) task는 28,000개 이상의 지문과 거의 100,000개의 질문이 있는 대규모 독해 데이터셋입니다. 이 데이터셋은 중/고등학생을 위해 설계된 중국의 영어 시험에서 수집되었습니다. RACE에서 각 지문은 여러 질문과 연결되어 있습니다. 모든 질문에 대한 task는 4개의 보기 중에서 하나의 정답을 선택하는 것입니다. RACE는 다른 독해 데이터셋보다 상당히 긴 context을 가지고 있으며 추론이 필요한 질문의 비율이 높습니다.

### 4. Training Procedure Analysis

(작성중...)
