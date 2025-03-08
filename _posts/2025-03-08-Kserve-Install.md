---
title: "Kserve 설치 및 구축"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-03-08 19:00:00 +0900
# categories: [AI | 딥러닝, Architecture]
# categories: [AI | 딥러닝, Concept]
# categories: [AI | 논문 리뷰, Attention is all you need]
categories: [MLOps | 인프라 개발, Model Serving]
# categories: [Life | 일상 이야기, 와플먹으면서 공부하기]
# categories: [STEM | 수학/통계, Statistics]
tags: [DeepLearning, MLOps, Kubeflow, Kserve]
description: "쿠버네티스 상에서 AI 모델을 서빙할 수 있는 Kserve 플랫폼을 구축해봅시다."
image: assets/img/posts/resize/output/kserve-architecture.png # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://kserve.github.io/website/latest/">https://kserve.github.io/website/latest/</a></small>
</div>

## KServe 설치

![](assets/img/posts/resize/output/kserve.png){: width="400px"}

<div align="center">
  <small>Source: <a href="https://kserve.github.io/website/latest/">https://kserve.github.io/website/latest/</a></small>
</div>

### 시작하기 전에 필요한 환경

[Kserve 공식 문서](https://kserve.github.io/website/latest/admin/serverless/serverless/)를 참고하여 Serverless 형태로 구축합니다.이에 따라 다음과 같은 환경이 필요합니다.

**(1) 쿠버네티스 클러스터**

- Kubernetes 1.28 이상 버전 (최소 요구사항), Minikube 또는 Kind로 구성된 개발용 클러스터
  - [Minikube Installation](https://minikube.sigs.k8s.io/docs/start/?arch=%2Flinux%2Fx86-64%2Fstable%2Fbinary+download)
  - [Kind Installation](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
- kubectl 최신 버전 (v1.28 이상 권장)
  - [Kubectl Installation](https://kubernetes.io/ko/docs/tasks/tools/install-kubectl-linux/)

필자는 Minikube를 활용하여 **v1.31.0** K8S, **v1.31.1** 버전의 Kubectl을 설치하였습니다.

```bash
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k get nodes -o wide
NAME        STATUS   ROLES           AGE   VERSION   INTERNAL-IP    EXTERNAL-IP   OS-IMAGE             KERNEL-VERSION     CONTAINER-RUNTIME
mlops       Ready    control-plane   50d   v1.31.0   192.168.49.2   <none>        Ubuntu 22.04.4 LTS   6.8.0-52-generic   docker://27.2.0
mlops-m02   Ready    worker          50d   v1.31.0   192.168.49.3   <none>        Ubuntu 22.04.4 LTS   6.8.0-52-generic   docker://27.2.0
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ kubectl version
Client Version: v1.31.1
Kustomize Version: v5.4.2
Server Version: v1.31.0
```

**(2) (선택 사항) GPU 지원**

GPU를 활용하고자 할 경우, gpu-operator가 이미 설치되어 있어야 합니다.

```bash
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k get pods -n gpu-operator
NAME                                                          READY   STATUS      RESTARTS       AGE
gpu-feature-discovery-9bwdm                                   1/1     Running     22 (47h ago)   45d
gpu-feature-discovery-ddjxp                                   1/1     Running     22 (47h ago)   45d
gpu-operator-55566cdcc9-8fd8s                                 1/1     Running     30 (24m ago)   40d
gpu-operator-node-feature-discovery-gc-7f546fd4bc-54pb7       1/1     Running     22 (47h ago)   45d
gpu-operator-node-feature-discovery-master-8448c8896c-89sdj   1/1     Running     41 (24m ago)   45d
gpu-operator-node-feature-discovery-worker-sntpd              1/1     Running     71 (47h ago)   45d
gpu-operator-node-feature-discovery-worker-spsqd              1/1     Running     58 (24m ago)   45d
nvidia-container-toolkit-daemonset-fk66k                      1/1     Running     22 (47h ago)   45d
nvidia-container-toolkit-daemonset-tgqh2                      1/1     Running     22 (47h ago)   45d
nvidia-cuda-validator-dqwpj                                   0/1     Completed   0              25m
nvidia-cuda-validator-lfvd6                                   0/1     Completed   0              24m
nvidia-dcgm-exporter-p2w4w                                    1/1     Running     22 (47h ago)   45d
nvidia-dcgm-exporter-s6r5z                                    1/1     Running     22 (47h ago)   45d
nvidia-device-plugin-daemonset-g58dc                          1/1     Running     50 (47h ago)   45d
nvidia-device-plugin-daemonset-x4w54                          1/1     Running     26 (47h ago)   45d
nvidia-operator-validator-h5r9r                               1/1     Running     22 (47h ago)   45d
nvidia-operator-validator-kkjcn                               1/1     Running     15 (47h ago)   45d
```

### Knative, Istio 설치

**(1) 버전 호환성 확인**

![](assets/img/posts/resize/output/kserve-metrics.png){: width="600px"}

<div align="center">
  <small>Source: <a href="https://kserve.github.io/website/latest/admin/serverless/serverless/">https://kserve.github.io/website/latest/admin/serverless/serverless/</a></small>
</div>

**(2) Knative 설치**

[Knative v1.17 Install Serving with YAML](https://knative.dev/docs/install/yaml-install/serving/install-serving-with-yaml/#install-the-knative-serving-component) 내용을 토대로 진행합니다.

- Knative Serving 구성 요소 중 사용자 정의 리소스(serving-crds),핵심 구성요소 (serving-core) manifests가 필요합니다.

- [Knative v1.17 Knative Serving installation files](https://knative.dev/docs/install/yaml-install/serving/serving-installation-files/)에서 아래 두 파일을 다운받고 설치합니다.

- serving-crds.yaml
- serving-core.yaml
  
```bash
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/kserve 설치/knative$  kubectl apply -f serving-crds.yaml
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/kserve 설치/knative$  kubectl apply -f serving-core.yaml
```

- knative component들을 확인합니다.

```bash
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/kserve 설치/knative$ k get pods -n knative-serving
NAME                          READY   STATUS    RESTARTS        AGE
activator-7c48c6944d-t55t2    1/1     Running   0               21m
autoscaler-775c659bc6-26vbl   1/1     Running   1 (9m45s ago)   21m
controller-7cf4fbd94-4vcnz    1/1     Running   5 (9m36s ago)   21m
webhook-57ccdb4884-25sqp      1/1     Running   3 (9m1s ago)    21m
```

**(3) Istio 설치**

[Knative v1.17 Install Istio for Knative](https://knative.dev/docs/install/installing-istio/#supported-istio-versions)을 토대로 진행합니다.

- 먼저 Istioctl를 설치합니다. (참고 [Install with Istioctl](https://istio.io/latest/docs/setup/install/istioctl/))

```bash
# istioctl 바이너리 파일 다운로드
# https://istio.io/latest/docs/setup/additional-setup/download-istio-release/
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:$ curl -L https://istio.io/downloadIstio | sh -
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05/istio-1.25.0:$ cd istio-1.25.0
# 환경변수 설정
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:$ export PATH=$PWD/bin:$PATH
```

- Istio를 설치하고 확인합니다.

```bash
# istio 설치
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:$ istioctl install
# istio 설치 확인
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:$ k get pods -n istio-system
NAME                                    READY   STATUS    RESTARTS   AGE
istio-ingressgateway-7f56c6746b-j5xbl   1/1     Running   0          73s
istiod-5dc686f4cf-h2rfj                 1/1     Running   0          87s
```
- Knative Istio controller를 설치합니다.

```bash
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ kubectl apply -f https://github.com/knative/net-istio/releases/download/knative-v1.17.0/net-istio.yaml
```

