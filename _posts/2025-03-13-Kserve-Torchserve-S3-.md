---
title: "[Kserve] 3. TorchServe 런타임 InferenceService 배포 (with S3)"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-03-13 19:00:00 +0900
# categories: [AI | 딥러닝, Architecture]
# categories: [AI | 딥러닝, Concept]
# categories: [AI | 논문 리뷰, Attention is all you need]
categories: [MLOps | 인프라 개발, Model Serving]
# categories: [Life | 일상 이야기, 와플먹으면서 공부하기]
# categories: [STEM | 수학/통계, Statistics]
tags: [DeepLearning, MLOps, Kubeflow, Kserve, TorchServe]
description: "Kserve를 활용하여 쿠버네티스 상에서 TorchServe를 배포해봅시다."
image: assets/img/posts/resize/output/kserve.png # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://kserve.github.io/website/latest/">https://kserve.github.io/website/latest/</a></small>
</div>

## Kserve TorchServe 런타임 배포

[Deploy a PyTorch Model with TorchServe InferenceService](https://kserve.github.io/website/latest/modelserving/v1beta1/torchserve/) 내용을 토대로 진행합니다.


### TorchServe란?

![](assets/img/posts/resize/output/torchserve.jpeg){: width="600px"}

<div align="center">
  <small>Source: <a href="https://github.com/pytorch/serve">https://github.com/pytorch/serve</a></small>
</div>

TorchServe는 PyTorch 모델을 서빙하기 위한 유연하고 확장 가능한 서비스입니다. PyTorch 팀에서 개발했으며, 다음과 같은 특징을 가집니다.

- 모델 관리: 모델 저장, 버전 관리, A/B 테스트 지원
- REST API: HTTP 기반의 API로 쉽게 접근 가능
- 성능 최적화: 배치 처리, TorchScript 지원
- 지표 모니터링: Prometheus 통합으로 성능 측정
- 모델 설명: Captum 라이브러리 통합으로 모델 해석 지원

KServe는 TorchServe를 기본 런타임으로 통합하여, PyTorch 모델을 쿠버네티스 환경에서 쉽게 서빙할 수 있습니다.

### 모델 학습 및 패키징

**(1) Mnist 이미지 분류 모델 학습**

- 데이터셋
  - Torchvision 내장 라이브러리 활용 Download
  - [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- 학습 코드
  - [https://github.com/pytorch/examples/blob/main/mnist/main.py](https://github.com/pytorch/examples/blob/main/mnist/main.py)

첨부한 데이터셋과 학습 코드를 다운받고 아래와 같은 디렉토리를 구성합니다.

```bash
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/model-example/mnist/pytorch$ tree
.
├── README.md
├── data
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
├── main.py
└── requirements.txt
```

Pytorch 가상환경을 구축하고 의존성을 설치합니다.

```bash
# 가상환경 설치
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/model-example/mnist/pytorch$ conda create -n pytorch python==3.10

# 가상환경 활성화
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/model-example/mnist/pytorch$ conda activate pytorch

# 의존성 설치
(pytorch) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/model-example/mnist/pytorch$ pip install -r requirements.txt 

# 모델 학습 및 저장
(pytorch) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/model-example/mnist/pytorch$ python main.py --save-model
(pytorch) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/model-example/mnist/pytorch$ ls | grep .pt
-rw-rw-r-- 1 jmlim jmlim 4.6M  3월 13 20:04 mnist_cnn.pt
```

**(2) 모델 패키징**

[TorchServe 공식 문서](https://github.com/pytorch/serve/blob/master/model-archiver/README.md)를 참고하여 MAR 파일로 패키징합니다.

```bash
# 모델 패키징
(pytorch) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/model-example/mnist/pytorch$ pip install torch-model-archiver
(pytorch) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/model-example/mnist/pytorch$ torch-model-archiver --model-name mnist --version 1.0 --serialized-file mnist_cnn.pt --handler image_classifier
(pytorch) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/model-example/mnist/pytorch$ ls | grep mar
-rw-rw-r-- 1 jmlim jmlim 4.3M  3월 13 20:18 mnist.mar
```

TorchServe에서 사용하는 모델 저장소 디렉토리로 구성하고 config.properties 파일을 설정합니다.

- **중요**
  - KServe v1/v2 REST 프로토콜을 지원하기 위해 enable_envvars_config=true 로 설정합니다. 이렇게 하면 KServe 요청 형식을 TorchServe 형식으로 변환할 수 있습니다.

```bash
# 디렉토리 구성
(pytorch) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/model-example/mnist/pytorch$ find config model-store -type d | sort | uniq | xargs tree
config
└── config.properties
model-store
└── mnist.mar

0 directories, 1 file

# config.properties 파일 설정
(pytorch) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/model-example/mnist/pytorch$ cat config/config.properties
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8085
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_metrics_api=true
metrics_format=prometheus
number_of_netty_threads=4
job_queue_size=10
enable_envvars_config=true
install_py_dep_per_model=true
model_store=/mnt/models/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"mnist":{"1.0":{"defaultVersion":true,"marName":"mnist.mar","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":10,"responseTimeout":120}}}}
```

### S3 업로드 및 ServiceAccount 설정

(작성중)