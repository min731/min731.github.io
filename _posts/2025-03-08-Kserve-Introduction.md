---
title: "Kserve 소개"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-03-08 18:00:00 +0900
# categories: [AI | 딥러닝, Architecture]
# categories: [AI | 딥러닝, Concept]
# categories: [AI | 논문 리뷰, Attention is all you need]
categories: [MLOps | 인프라 개발, Model Serving]
# categories: [Life | 일상 이야기, 와플먹으면서 공부하기]
# categories: [STEM | 수학/통계, Statistics]
tags: [DeepLearning, MLOps, Kubeflow, Kserve]
description: "쿠버네티스 상에서 AI 모델을 서빙할 수 있는 Kserve에 대해 알아봅시다."
image: assets/img/posts/resize/output/kserve.png # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://kserve.github.io/website/latest/">https://kserve.github.io/website/latest/</a></small>
</div>

### KServe란?

KServe는 쿠버네티스(Kubernetes) 환경에서 AI 모델을 쉽고 효율적으로 배포하고 관리할 수 있게 해주는 오픈소스 플랫폼입니다. Python으로 개발한 머신러닝 또는 딥러닝 모델을 실제 서비스로 제공할 때 필요한 모든 복잡한 과정을 자동화해주는 도구입니다.

![](assets/img/posts/resize/output/kserve-architecture.png){: width="600px"}

<div align="center">
  <small>Source: <a href="https://kserve.github.io/website/latest/">https://kserve.github.io/website/latest/</a></small>
</div>

원래는 'KFServing'이라는 이름으로 시작했지만, 현재는 'KServe'로 이름이 바뀌었습니다. Kserve는 구글, IBM, 블룸버그 등 여러 기업이 참여하는 Kubeflow 프로젝트의 일부로 개발되었고, 현재 GitHub에서 4,000개 이상의 스타와 1,100개 이상의 포크를 보유한 활발한 프로젝트입니다.

### KServe의 주요 기능과 특징

**(1) 다양한 ML 프레임워크 지원**

- TensorFlow: 구글의 인기 딥러닝 프레임워크
- PyTorch: 페이스북(메타)이 개발한 유연한 딥러닝 도구
- scikit-learn: 데이터 사이언스에서 널리 사용되는 간단한 ML 라이브러리
- XGBoost: 그래디언트 부스팅에 최적화된 라이브러리
- ONNX: 다양한 프레임워크 간 모델 교환 형식

**(2) 서버리스 아키텍처**

"서버리스"란 서버 관리에 신경 쓰지 않아도 된다는 의미입니다.

- Scale to Zero: 모델에 요청이 없을 때는 컴퓨팅 자원을 0으로 줄여 비용 절감
- 자동 확장: 트래픽이 증가하면 자동으로 더 많은 자원 할당
- GPU 지원: 딥러닝 모델을 위한 GPU 가속 및 효율적인 관리

예를 들어, 웹사이트에 방문자가 많을 때는 서버가 자동으로 늘어나고, 한밤중에 방문자가 없을 때는 서버를 전부 종료하듯이 유연하게 대처하는 기능이라고 보시면 됩니다.

**(3) 모델 배포의 유연성**

새로운 모델 버전을 안전하게 배포하는 방법을 제공합니다.

- 카나리 배포: 새 버전을 일부 트래픽(예: 10%)에만 적용해 테스트
- A/B 테스트: 여러 모델 버전을 동시에 실행하고 성능 비교
- 롤백: 문제 발생 시 이전 버전으로 빠르게 되돌리기

Kserve는 Kubernetes에서 사용되는 기술이다보니 K8S 장점을 그대로 가지고 있다고 보시면 됩니다.

**(4) 전처리 및 후처리 파이프라인**

모델에 데이터를 보내기 전이나 결과를 반환하기 전에 처리 단계를 추가할 수 있습니다.

- input 데이터 변환: 원시 입력을 모델에 맞게 변환
- output 결과 포맷팅: 모델 출력을 사용자 친화적인 형태로 변환
- 비즈니스 로직 적용: 예측 결과에 비즈니스 규칙 커스텀

예를 들어, 이미지 분류 모델이라면 입력 이미지 크기 조정(전처리)과 예측 결과를 사람이 읽기 쉬운 레이블로 변환(후처리)을 자동화할 수 있습니다.

**(5) ModelMesh로 대규모 모델 관리**

많은 모델을 효율적으로 관리해야 하는 기업을 위한 고급 기능

- 메모리 최적화: 필요한 모델만 메모리에 로드하여 자원 효율성 극대화
- 지능형 라우팅: 요청을 적절한 모델 인스턴스로 자동 전달
- 고밀도 패킹: 하나의 서버에 여러 모델을 효율적으로 배치

금융 기관이 고객별로 다른 신용 평가 모델을 수백 개 운영해야 한다면, ModelMesh는 이를 효율적으로 관리할 수 있게 해줍니다.

**(6) 모델 모니터링 및 설명 가능성**

AI 시스템의 투명성과 신뢰성을 높이는 기능

- 성능 모니터링: 응답 시간, 처리량 등 기술적 지표 추적
- 데이터 드리프트 감지: 입력 데이터 패턴 변화 감지
- 설명 가능성: 모델이 왜 특정 예측을 했는지 이해하는 도구 제공
- 공정성 모니터링: AI 시스템의 편향 감지 및 모니터링

의료 진단 모델이라면, 단순히 결과만 제공하는 것이 아니라 "이 진단을 내린 주요 특징은 X, Y, Z입니다"라고 설명할 수 있습니다.

### KServe 아키텍처 구성 요소

KServe는 다음과 같은 핵심 구성 요소로 이루어져 있습니다.

- Controller: 모델 배포를 관리하고 쿠버네티스 리소스를 조정
- InferenceService: 모델 서빙을 위한 핵심 쿠버네티스 커스텀 리소스(CRD)
- Model Server(모델 서버): 실제 예측을 수행하는 컨테이너 (TensorFlow Serving, PyTorch Server 등)
- Transformer(변환기): 입력/출력 데이터 변환을 위한 컴포넌트
- Explainer(설명기): 모델 예측에 대한 설명을 제공하는 컴포넌트

## Kserve 정리

KServe는 ML 모델을 연구 환경에서 실제 비즈니스 환경으로 이전하는 과정에서 발생하는 많은 문제를 해결해주는 플랫폼입니다. 표준화된 방식으로 모델을 배포하고, 자동 확장 기능으로 리소스를 효율적으로 관리하며, 모니터링 및 설명 도구로 AI 시스템의 신뢰성을 높여줄 수 있습니다.

현재(작성일 기준/2025년 03월 08일) v0.14.1 버전까지 출시되었으며, 계속해서 활발히 개발되고 있는 오픈소스 프로젝트입며 클라우드 네이티브 ML 인프라를 구축하고자 할 때 괜찮은 선택지입니다.