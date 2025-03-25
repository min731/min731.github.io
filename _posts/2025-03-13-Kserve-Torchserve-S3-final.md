---
title: "[Kserve] 3. TorchServe 런타임 InferenceService 배포 (with S3)"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-03-13 20:00:00 +0900
# categories: [AI | 딥러닝, Architecture]
# categories: [AI | 딥러닝, Concept]
# categories: [AI | 논문 리뷰, Attention is all you need]
categories: [MLOps | 인프라 개발, Model Serving]
# categories: [Life | 일상 이야기, 와플먹으면서 공부하기]
# categories: [STEM | 수학/통계, Statistics]
tags: [DeepLearning, MLOps, Kubeflow, Kserve, TorchServe]
description: "Kserve를 활용하여 쿠버네티스 상에 TorchServe 런타임을 배포해봅시다."
image: assets/img/posts/resize/output/kserve.png # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://kserve.github.io/website/latest/">https://kserve.github.io/website/latest/</a></small>
</div>

> 앞선 글
- [[Kserve] 1.소개](https://min731.github.io/posts/Kserve-Introduction/)
- [[Kserve] 2. 설치 및 구축](https://min731.github.io/posts/Kserve-Installation/)

## Kserve TorchServe 런타임 배포

[Kserve Docs 'Deploy a PyTorch Model with TorchServe InferenceService'](https://kserve.github.io/website/latest/modelserving/v1beta1/torchserve/) 내용을 토대로 진행합니다.

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

**(1) 개요**

Kserve 공식 문서에 링크된 [TorchServe 공식 Repository](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist)를 활용하여 Mnist 모델을 패키징합니다.

**(1) 가상환경 구축**

Pytorch 가상환경을 구축하고 의존성을 설치합니다.

```bash
# 가상환경 설치
(base) jmlim@Legion-5:~/model-example/mnist/pytorch$ conda create -n pytorch python==3.10

# 가상환경 활성화
(base) jmlim@Legion-5:~/model-example/mnist/pytorch$ conda activate pytorch

# requirements.txt 파일 작성
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ cat requirements.txt
torch
torchvision==0.20.0

# 의존성 설치
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ pip install -r requirements.txt 
```

**(2) 모델 패키징**

[TorchServe 공식 문서](https://github.com/pytorch/serve/blob/master/model-archiver/README.md)를 참고하여 MAR 파일로 패키징합니다.

```bash
# TorchServe 레포지토리 Clone
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ git clone https://github.com/pytorch/serve.git

# mnist 모델 패키징
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch/serve$ torch-model-archiver --model-name mnist --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/mnist_handler.py

# mnist.mar 파일 생성
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch/serve$ find ./ -name "mnist.mar"
./frontend/archive/src/test/resources/models/mnist.mar
./test/resources/models/mnist.mar
./mnist.mar
```

TorchServe에서 사용하는 모델 저장소 디렉토리로 구성하고 config.properties 파일을 설정합니다.

- **중요**
  - KServe v1/v2 REST 프로토콜을 지원하기 위해 enable_envvars_config=true 로 설정합니다. KServe 요청 형식을 TorchServe 형식으로 변환할 수 있습니다.

```bash
# config.properties 파일 설정
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ cat config/config.properties
inference_address=http://0.0.0.0:8085 # 추론 요청을 처리하는 HTTP 서버 주소 및 포트
management_address=http://0.0.0.0:8085 # 모델 관리 API(모델 등록/삭제 등)를 처리하는 서버 주소 및 포트
metrics_address=http://0.0.0.0:8082 # 메트릭 수집 및 노출을 위한 서버 주소 및 포트
grpc_inference_port=7070 # gRPC 추론 API를 제공하는 포트
grpc_management_port=7071 # gRPC 관리 API를 제공하는 포트
enable_metrics_api=true # 메트릭 API 활성화 여부
metrics_format=prometheus # 메트릭 출력 형식 (Prometheus 형식으로 노출)
number_of_netty_threads=4 # 요청 처리를 위한 Netty 스레드 수
job_queue_size=10 # 대기열에 넣을 수 있는 작업 수
enable_envvars_config=true # 환경 변수를 통한 설정 활성화 (KServe와 통합 시 중요)
install_py_dep_per_model=true # 모델별 Python 종속성 설치 허용
model_store=/mnt/models/model-store # 모델 저장소 경로 지정
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"mnist":{"1.0":{"defaultVersion":true,"marName":"mnist.mar","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":10,"responseTimeout":120}}}} # mnist 모델의 버전, 작업자 수, 배치 크기 등 정의
```

```bash
# 디렉토리 구성
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ find config model-store -type d | sort | uniq | xargs tree
config
└── config.properties # config.properties 파일 복사
model-store
└── mnist.mar # mnist.mar 파일 복사

0 directories, 1 file
```

### S3 업로드 및 ServiceAccount 설정

**(1) S3 버킷에 모델 업로드**

```bash
# awscli 설치
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ pip install awscli

# awscli 인증 정보 설정 (Access Key, Secret Access Key 등)
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ aws configure

# TorchServe 디렉토리 업로드
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ aws s3 cp --recursive config s3://jmbucket731/kserve/torchserve/mnist/config/
upload: config/config.properties to s3://jmbucket731/kserve/torchserve/mnist/config/config.properties
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ aws s3 cp --recursive model-store s3://jmbucket731/kserve/torchserve/mnist/model-store/
upload: model-store/mnist.mar to s3://jmbucket731/kserve/torchserve/mnist/model-store/mnist.mar
```

**(2) S3 인증 정보 Secret 생성**

S3에 저장된 모델에 접근하기 위해 인증 정보가 포함된 Secret을 생성합니다.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: s3creds
  namespace: torchserve
  annotations:
     serving.kserve.io/s3-endpoint: s3.amazonaws.com  # S3 엔드포인트
     serving.kserve.io/s3-usehttps: "1"  # HTTPS 사용 여부 (1: 사용, 0: 미사용)
     serving.kserve.io/s3-region: "ap-northeast-2"  # 리전 설정
     serving.kserve.io/s3-useanoncredential: "false"  # 익명 인증 미사용
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: XXXXXXXXXXXX  # 실제 액세스 키로 교체
  AWS_SECRET_ACCESS_KEY: XXXXXXXXXXXXXXXX  # 실제 시크릿 키로 교체
```

```bash
# torchserve 네임스페이스 생성
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ k create ns torchserve
namespace/torchserve created

# secret 배포 및 확인
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ k apply -f s3-secret.yaml.yaml
secret/s3creds created
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ k get secret -n torchserve
NAME      TYPE     DATA   AGE
s3creds   Opaque   2      19s
```

**(3) ServiceAccount 생성 및 Secret 연결**

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: torchserve-sa
  namespace: torchserve
secrets:
- name: s3creds
```

```bash
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ k apply -f torchserve-sa.yaml
serviceaccount/torchserve-sa created
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ k get sa -n torchserve
NAME            SECRETS   AGE
default         0         3m3s
torchserve-sa   1         4s
```

### TorchServe InferenceService 배포

S3에 저장된 TorchServe 모델을 사용하여 InferenceService를 배포합니다.

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "mnist-torchserve"
  namespacea: torchserve
spec:
  predictor:
    serviceAccountName: torchserve-sa  # 위에서 생성한 ServiceAccount
    model:
      modelFormat:
        name: pytorch  # PyTorch 모델 지정
      storageUri: "s3://jmbucket731/kserve/torchserve/mnist"  # S3 경로
```

```bash
# Inferenceservice 배포 
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ k apply -f mnist-torchserve.yaml
inferenceservice.serving.kserve.io/mnist-torchserve created

# Predictor Pod 확인
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ k get pods -n torchserve
NAME                                                          READY   STATUS    RESTARTS   AGE
mnist-torchserve-predictor-00001-deployment-df8cd9c66-z7rdj   2/2     Running   0          3m20s

# InferenceServive 확인
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ k get isvc -n torchserve
NAME               URL                                                    READY   PREV   LATEST   PREVROLLEDOUTREVISION   LATESTREADYREVISION                AGE
mnist-torchserve   http://mnist-torchserve.torchserve.svc.cluster.local   True           100                              mnist-torchserve-predictor-00001   5m48s
```

### TorchServe InferenceService 추론

**(1) 테스트 데이터 준비**

- TorchServe 공식 Repo Mnist 데이터
  - [https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist/test_data](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist/test_data)

```bash
# TorchServe 예제에서 제공하는 테스트 이미지 사용
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ wget https://github.com/pytorch/serve/raw/master/examples/image_classifier/mnist/test_data/0.png

# 테스트 데이터 확인
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ ls | grep 0.png
0.png
```

[Kserve 공식 Repository](https://github.com/kserve/kserve/tree/master/docs/samples/v1beta1/torchserve/v1/imgconv)를 참고하여 Mnist png 파일을 bytearray 데이터로 변환하고 이를 json 파일로 저장해줍니다.

```bash
jmlim@Legion-5:~/model-example/mnist/pytorch$ ls |grep .json
-rw-rw-r--  1 jmlim jmlim  418  3월 22 11:56 mnist.json
jmlim@Legion-5:~/model-example/mnist/pytorch$ cat mnist.json
{
  "instances": [
    {
      "data": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAA10lEQVR4nGNgGFhgy6xVdrCszBaLFN/mr28+/QOCr69DMCSnA8WvHti0acu/fx/10OS0X/975CDDw8DA1PDn/1pBVEmLf3+zocy2X/+8USXt/82Ds+/+m4sqeehfOpw97d9VFDmlO++t4JwQNMm6f6sZcEpee2+DR/I4A05J7tt4JJP+IUsu+ncRp6TxO9RAQJY0XvrvMAuypNNHuCTz8n+PzVEcy3DtqgiY1ptx6t8/ewY0yX9ntoDA63//Xs3hQpMMPPsPAv68qmDAAFKXwHIzMzCl6AoAxXp0QujtP+8AAAAASUVORK5CYII="
    }
  ]
}

```


**(2) 클러스터 및 Istio 정보 확인**

```bash
# Ingress Host, Port 및 Service Hostname 확인
(pytorch) jmlim@Legion-5:~/model-example/mnist/pytorch$ INGRESS_HOST=$(minikube -p mlops ip)
INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}')
SERVICE_HOSTNAME=$(kubectl get inferenceservice mnist-torchserve -n torchserve -o jsonpath='{.status.url}' | cut -d "/" -f 3)
echo "$INGRESS_HOST" "$INGRESS_PORT" "$SERVICE_HOSTNAME"
192.168.49.2 30439 mnist-torchserve.torchserve.svc.cluster.local
```

**(3) Knative Gateway 확인 및 VirtualService 생성**

[[Kserve] 2. 설치 및 구축](https://min731.github.io/posts/Kserve-Installation/) 단계에서 외부에서 접속하기 위해 'net-istio'를 배포했습니다. 'knative-ingress-gateway' Gateway를 확인해봅니다.

```bash
jmlim@Legion-5:~/model-example/mnist/pytorch$ k get gw knative-ingress-gateway -n knative-serving -o yaml
apiVersion: networking.istio.io/v1
kind: Gateway
metadata:
  annotations:
    kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"networking.istio.io/v1beta1","kind":"Gateway","metadata":{"annotations":{},"labels":{"app.kubernetes.io/component":"net-istio","app.kubernetes.io/name":"knative-serving","app.kubernetes.io/version":"1.17.0","networking.knative.dev/ingress-provider":"istio"},"name":"knative-ingress-gateway","namespace":"knative-serving"},"spec":{"selector":{"istio":"ingressgateway"},"servers":[{"hosts":["*"],"port":{"name":"http","number":80,"protocol":"HTTP"}}]}}
  creationTimestamp: "2025-03-08T11:44:30Z"
  generation: 1
  labels:
    app.kubernetes.io/component: net-istio
    app.kubernetes.io/name: knative-serving
    app.kubernetes.io/version: 1.17.0
    networking.knative.dev/ingress-provider: istio
  name: knative-ingress-gateway
  namespace: knative-serving
  resourceVersion: "162127"
  uid: 99ea6518-3320-4621-bf8e-12377e006046
spec:
  selector:
    istio: ingressgateway
  servers:
  - hosts:
    - '*'
    port:
      name: http
      number: 80
      protocol: HTTP
```

외부에서 모델 서비스에 접근하기 위한 Istio VirtualService를 생성합니다.
- '/v1/models/mnist' 경로로 들어오는 요청을 'mnist-torchserve-predictor.torchserve.svc.cluster.local' 서비스로 라우팅합니다.
- TorchServe는 REST API 표준에 따라 /v1/models/{모델명}:predict 형식의 엔드포인트를 제공합니다.

```yaml
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: mnist-torchserve-external
  namespace: torchserve
spec:
  hosts:
  - "*"
  gateways:
  - knative-serving/knative-ingress-gateway  # knative-ingress-gateway Gateway 참조
  http:
  - match:
    - uri:
        prefix: /v1/models/mnist
    route:
    - destination:
        host: mnist-torchserve-predictor.torchserve.svc.cluster.local
        port:
          number: 80
```

해당 VirtualService를 배포합니다.

```bash
# VirtualService 배포
jmlim@Legion-5:~/model-example/mnist/pytorch$ k apply -f vs-torchserve.yaml
virtualservice.networking.istio.io/mnist-torchserve-external created
```

Host: mnist-torchserve-predictor.torchserve.svc.cluster.local 헤더를 포함하여 외부에서 TorchServe 추론합니다.

```bash
jmlim@Legion-5:~/model-example/mnist/pytorch$ curl -v -H "Host: mnist-torchserve-predictor.torchserve.svc.cluster.local" \
  -H "Content-Type: application/json" \
  http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/mnist:predict \
  -d @./mnist.json
*   Trying 192.168.49.2:30439...
* Connected to 192.168.49.2 (192.168.49.2) port 30439 (#0)
> POST /v1/models/mnist:predict HTTP/1.1
> Host: mnist-torchserve-predictor.torchserve.svc.cluster.local
> User-Agent: curl/7.81.0
> Accept: */*
> Content-Type: application/json
> Content-Length: 411
> 
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< content-length: 19
< content-type: application/json
< date: Tue, 25 Mar 2025 11:18:37 GMT
< server: istio-envoy
< x-envoy-upstream-service-time: 376
< 
* Connection #0 to host 192.168.49.2 left intact
{"predictions":[0]}
```
최종적으로 {"predictions":[0]} 응답을 받았습니다. Mnist 테스트 이미지(0.png)가 숫자 '0'으로 정확하게 분류되었습니다.

## 마치며

이번 Kserve 시리즈를 통해 기술 소개부터 설치, 그리고 TorchServe 모델 배포까지 ML 모델 서빙 파이프라인의 전체 과정을 다뤘습니다. 작성된 내용과 같이 KServe는 Knative의 서버리스 기능과 다양한 ML 프레임워크 지원을 통해 AI 모델 배포를 간소화하는 기술입니다.

Kserve는 최근 실무에서 쿠버네티스 환경에서 AI 모델 서빙을 구현하는 방법을 찾던 중 알게된 기술인데요. 국내에서는 참고할 만한 자료가 거의 없어, 구현 당시 상당히 어려움을 겪었습니다.

그리하여 이에 대해 관심있는 분들에게 조금이나마 도움이 될까 싶어 실무에서 직접 부딪히고 해결한 경험을 토대로 정리해보았습니다. 이글 읽고 많은 분들에게 도움이 되었으면 좋겠습니다. 

읽어주셔서 감사합니다!