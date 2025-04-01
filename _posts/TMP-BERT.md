---
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-03-27 19:00:00 +0900
# categories: [AI | 딥러닝, Architecture]
# categories: [AI | 딥러닝, Concept]
categories: [AI | 딥러닝, Paper]
# categories: [MLOps | 인프라 개발, Kserve]
# categories: [Life | 일상 이야기, 와플먹으면서 공부하기]
# categories: [STEM | 수학/통계, Statistics]
tags: [DeepLearning, BERT, Bidirectional Transformers]
description: "BERT 논문을 살펴보고 주요 특징들을 정리해봅시다."
image: assets/img/posts/resize/output/BERT_input_embeddings.png # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://commons.wikimedia.org/wiki/File:BERT_input_embeddings.png">https://commons.wikimedia.org/wiki/File:BERT_input_embeddings.png</a></small>
</div>

## BERT

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) 내용을 토대로 정리합니다.

### Abstract

**[원문]**

"We introduce a new language representa-
tion model called BERT, which stands for
Bidirectional Encoder Representations from
Transformers. Unlike recent language repre-
sentation models (Peters et al., 2018a; Rad-
ford et al., 2018), BERT is designed to pre-
train deep bidirectional representations from
unlabeled text by jointly conditioning on both
left and right context in all layers. As a re-
sult, the pre-trained BERT model can be fine-
tuned with just one additional output layer
to create state-of-the-art models for a wide
range of tasks, such as question answering and
language inference, without substantial task-
specific architecture modifications."

<div align="center">
  <small>Source: <a href="https://arxiv.org/abs/1810.04805">https://arxiv.org/abs/1810.04805</a></small>
</div>

**[해석]**

BERT는 Bidirectional Encoder Representations from Transformers의 약자로, 기존 언어 모델들과 달리 모든 layer에서 양방향(좌우 context)을 동시에 고려하도록 설계된 pre-training 모델입니다. 이 모델의 가장 큰 강점은 pre-training 후에 단지 하나의 output layer만 추가하여 fine-tuning함으로써 question answering이나 language inference와 같은 다양한 task에서 state-of-the-art 성능을 달성할 수 있다는 점입니다. 즉, task별로 아키텍처를 크게 수정할 필요가 없습니다.

"Language model pre-training has been shown to
be effective for improving many natural language
processing tasks (Dai and Le, 2015; Peters et al.,
2018a; Radford et al., 2018; Howard and Ruder,
2018). These include sentence-level tasks such as
natural language inference (Bowman et al., 2015;
Williams et al., 2018) and paraphrasing (Dolan
and Brockett, 2005), which aim to predict the re-
lationships between sentences by analyzing them
holistically, as well as token-level tasks such as
named entity recognition and question answering,
where models are required to produce fine-grained
output at the token level (Tjong Kim Sang and
De Meulder, 2003; Rajpurkar et al., 2016).
"

<div align="center">
  <small>Source: <a href="https://arxiv.org/abs/1810.04805">https://arxiv.org/abs/1810.04805</a></small>
</div>

<img src="https://github.com/user-attachments/assets/04d13b00-08f0-4151-8f89-f69b2d2f0f4a" width="90%" height="90%" title="제목" alt="아무거나"/> 

![](https://github.com/user-attachments/assets/04d13b00-08f0-4151-8f89-f69b2d2f0f4a)

![](assets/img/posts/resize/output/torchserve.jpeg){: width="600px"}