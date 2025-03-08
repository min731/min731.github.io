---
title: "Kserve 소개 및 설치"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-03-06 00:00:00 +0900
# categories: [AI | 딥러닝, Architecture]
# categories: [AI | 딥러닝, Concept]
# categories: [AI | 논문 리뷰, Attention is all you need]
categories: [MLOps | 인프라 개발, Model Serving]
# categories: [Life | 일상 이야기, 와플먹으면서 공부하기]
# categories: [STEM | 수학/통계, Statistics]
tags: [DeepLearning, MLOps, Kubeflow, Kserve]
description: "쿠버네티스 상에서 인공지능 모델을 서빙할 수 있는 Kserve를 설치해보자."
image: assets/img/posts/resize/output/kserve-architecture.png # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://kserve.github.io/website/latest/">https://kserve.github.io/website/latest/</a></small>
</div>

1. KServe 소개 / 구축 / 설치
KServe란 무엇인가?
KServe의 주요 기능과 특징
KServe 설치 준비: 클러스터 요구 사항, 필요한 툴 (예: kubectl, helm)
KServe 설치 방법 (Helm chart 사용, manifests 적용 등)
KServe 설치 후 확인 작업
1. AI 모델 개발 및 Image 빌드
AI 모델 개발: 예시 (예: Scikit-learn, TensorFlow, PyTorch)
모델을 서빙하기 위한 준비 (모델 파일 포맷, inference API 설계 등)
Docker 이미지 빌드:
Dockerfile 작성법 (모델을 포함한 이미지 생성)
모델을 Docker 이미지에 포함시키는 방법
CI/CD 파이프라인 설정 (옵션)
1. 배포 및 테스트
KServe로 모델 배포:
PersistentVolume/PersistentVolumeClaim 설정
InferenceService 설정 (배포 방법)
KServe와 함께 사용할 모델 포맷 설정 (예: Scikit-learn, TensorFlow)
모델 테스트:
REST API 호출로 모델 테스트
KServe의 예시 테스트 스크립트
다양한 부하 테스트 방법 (성능 테스트)
1. 모델 관리 및 모니터링
모델 업데이트 및 롤링 배포
모델 버전 관리 (예: 모델을 여러 버전으로 서빙)
모델 성능 모니터링 및 로깅
1. 고급 기능 및 최적화
다중 모델 서빙: 여러 모델을 동시에 서빙하는 방법
모델 서버 최적화 (메모리 사용량, 응답 시간 최적화 등)
KServe의 autoscaling 설정 (모델에 맞게 스케일링 조정)
GPU 모델 서빙: GPU 자원 할당 및 최적화
1. 실전 예제
실제 사용 사례 예시 (예: 이미지 분류, 텍스트 분석 모델 등)
단계별 프로젝트 예제: 모델 학습부터 배포까지
문제 해결 사례: 배포 중 발생할 수 있는 문제 및 해결 방법