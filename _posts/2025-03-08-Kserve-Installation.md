---
title: "[Kserve] 2. 설치 및 구축"
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
image: assets/img/posts/resize/output/kserve.png # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://kserve.github.io/website/latest/">https://kserve.github.io/website/latest/</a></small>
</div>

## KServe 설치

- KServe를 설치하기 위해서는 Knative, Istio, Cert Manager 등 여러 third-party 구성 요소를 먼저 구축해야 합니다.
- **용기를 가지고** 아래 가이드를 차근차근 참고하여 설치해봅시다!

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
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/kserve 설치/knative$  k apply -f serving-crds.yaml
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~/kserve 설치/knative$  k apply -f serving-core.yaml
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
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k apply -f https://github.com/knative/net-istio/releases/download/knative-v1.17.0/net-istio.yaml
```

- Istio 외부 접속을 위해 LoadBalancer 타입의 Service를 **Nodeport**로 변경합니다. (혹은 MetaLB를 통해 LoadBalancer로 활용해도 됩니다.)
- 본 글에서 활용하고 있는 Minikube 클러스터에서 Nodeport 사용 시, Minikbe의 IP를 통해 요청할 수 있습니다. 

```bash
# Istio Service 타입 변경
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k patch svc istio-ingressgateway -n istio-system -p '{"spec": {"type": "NodePort"}}'
service/istio-ingressgateway patched
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k get svc -n istio-system
NAME                    TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                                      AGE
istio-ingressgateway    NodePort    10.101.70.126   <none>        15021:31483/TCP,80:30439/TCP,443:30839/TCP   2d22h
istiod                  ClusterIP   10.102.2.212    <none>        15010/TCP,15012/TCP,443/TCP,15014/TCP        2d22h
knative-local-gateway   ClusterIP   10.98.109.221   <none>        80/TCP,443/TCP                               2d22h

# Minikube 클러스터 IP 확인
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ mk -p mlops ip
192.168.49.2
```

**(4) (선택) Knative HPA 설치**

- Kubernetes의 기본 HPA 대신 Knative의 HPA를 사용하여 자동 스케일링하기 위해 설치합니다.

```bash
# knative-serving hpa 설치
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k apply -f https://github.com/knative/serving/releases/download/knative-v1.17.0/serving-hpa.yaml
deployment.apps/autoscaler-hpa created
service/autoscaler-hpa created

# 확인
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k get pods -n knative-serving
NAME                                    READY   STATUS    RESTARTS      AGE
activator-7c48c6944d-t55t2              1/1     Running   2 (62m ago)   2d23h
autoscaler-775c659bc6-26vbl             1/1     Running   3 (62m ago)   2d23h
autoscaler-hpa-998cd99bb-bjxkn          1/1     Running   0             11s
controller-7cf4fbd94-4vcnz              1/1     Running   9 (34m ago)   2d23h
net-istio-controller-5d7c696f76-sdg2p   1/1     Running   3 (62m ago)   2d22h
net-istio-webhook-58bb884dc-jlbvz       1/1     Running   3 (62m ago)   2d22h
webhook-57ccdb4884-25sqp                1/1     Running   6 (62m ago)   2d23h
```

### Cert-Manager 설치

[Cert-Manager Installation](https://cert-manager.io/docs/installation/)을 참고하여 진행합니다.

- Kubernetes 환경에서 TLS 인증서를 자동으로 관리하기 위해 Cert Manager 설치합니다.

```bash
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.17.0/cert-manager.yaml
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k get pods -n cert-manager
NAME                                       READY   STATUS    RESTARTS   AGE
cert-manager-5b85cc56c4-grp7d              1/1     Running   0          43s
cert-manager-cainjector-547db48bc7-2kmvk   1/1     Running   0          43s
cert-manager-webhook-58f7b445c4-jvd27      1/1     Running   0          43s
```
### Kserve v0.14.1 CRD / Controller / ClusterServingRuntime 설치

- 드디어 최종적으로 Kserve를 배포합니다!
- [Install KServe - Install using YAML](https://kserve.github.io/website/latest/admin/serverless/serverless/#1-install-knative-serving)을 참고합니다.

```bash
# CRD, Controller 설치
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.14.1/kserve.yaml

# ClusterServingRuntime 설치
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.14.1/kserve-cluster-resources.yaml
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k get all -n kserve
NAME                                                        READY   STATUS    RESTARTS   AGE
pod/kserve-controller-manager-5c6ff8c6d4-m778w              2/2     Running   0          3m26s
pod/kserve-localmodel-controller-manager-7df647ffbf-dk82j   1/1     Running   0          3m26s

NAME                                                TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
service/kserve-controller-manager-metrics-service   ClusterIP   10.100.25.208   <none>        8443/TCP   3m26s
service/kserve-controller-manager-service           ClusterIP   10.103.244.57   <none>        8443/TCP   3m26s
service/kserve-webhook-server-service               ClusterIP   10.109.48.66    <none>        443/TCP    3m26s

NAME                                         DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR              AGE
daemonset.apps/kserve-localmodelnode-agent   0         0         0       0            0           kserve/localmodel=worker   3m26s

NAME                                                   READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/kserve-controller-manager              1/1     1            1           3m26s
deployment.apps/kserve-localmodel-controller-manager   1/1     1            1           3m26s

NAME                                                              DESIRED   CURRENT   READY   AGE
replicaset.apps/kserve-controller-manager-5c6ff8c6d4              1         1         1       3m26s
replicaset.apps/kserve-localmodel-controller-manager-7df647ffbf   1         1         1       3m26s
```
- Kserve 설치를 확인합니다.

```bash
# 컴포넌트 확인
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k get all -n kserve
NAME                                                        READY   STATUS    RESTARTS   AGE
pod/kserve-controller-manager-5c6ff8c6d4-m778w              2/2     Running   0          3m26s
pod/kserve-localmodel-controller-manager-7df647ffbf-dk82j   1/1     Running   0          3m26s

NAME                                                TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
service/kserve-controller-manager-metrics-service   ClusterIP   10.100.25.208   <none>        8443/TCP   3m26s
service/kserve-controller-manager-service           ClusterIP   10.103.244.57   <none>        8443/TCP   3m26s
service/kserve-webhook-server-service               ClusterIP   10.109.48.66    <none>        443/TCP    3m26s

NAME                                         DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR              AGE
daemonset.apps/kserve-localmodelnode-agent   0         0         0       0            0           kserve/localmodel=worker   3m26s

NAME                                                   READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/kserve-controller-manager              1/1     1            1           3m26s
deployment.apps/kserve-localmodel-controller-manager   1/1     1            1           3m26s

NAME                                                              DESIRED   CURRENT   READY   AGE
replicaset.apps/kserve-controller-manager-5c6ff8c6d4              1         1         1       3m26s
replicaset.apps/kserve-localmodel-controller-manager-7df647ffbf   1         1         1       3m26s

# CRD(Custom Resource Definitions) 확인
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k get crd | grep serving.kserve.io
clusterservingruntimes.serving.kserve.io              2025-01-22T10:59:42Z
clusterstoragecontainers.serving.kserve.io            2025-01-22T10:59:42Z
inferencegraphs.serving.kserve.io                     2025-01-22T10:59:42Z
inferenceservices.serving.kserve.io                   2025-01-22T10:59:42Z
localmodelcaches.serving.kserve.io                    2025-01-22T11:13:06Z
localmodelnodegroups.serving.kserve.io                2025-01-22T11:13:06Z
localmodelnodes.serving.kserve.io                     2025-01-22T11:13:06Z
servingruntimes.serving.kserve.io                     2025-01-22T10:59:44Z
trainedmodels.serving.kserve.io                       2025-01-22T10:59:44Z

# ClusterServingRuntimes 확인
(base) jmlim@jmlim-Lenovo-Legion-5-15ARH05:~$ k get clusterservingruntimes
NAME                        DISABLED   MODELTYPE     CONTAINERS         AGE
kserve-huggingfaceserver               huggingface   kserve-container   4m
kserve-lgbserver                       lightgbm      kserve-container   4m
kserve-mlserver                        sklearn       kserve-container   4m
kserve-paddleserver                    paddle        kserve-container   4m
kserve-pmmlserver                      pmml          kserve-container   4m
kserve-sklearnserver                   sklearn       kserve-container   4m
kserve-tensorflow-serving              tensorflow    kserve-container   4m
kserve-torchserve                      pytorch       kserve-container   4m
kserve-tritonserver                    tensorrt      kserve-container   4m
kserve-xgbserver                       xgboost       kserve-container   4m
```

## 정리

이로써 KServe 설치 과정을 성공적으로 완료했습니다. KServe의 핵심 컴포넌트들을 통해 이제 Kubernetes 환경에서 ML 모델을 효율적으로 배포하고 관리할 수 있는 기반이 마련되었습니다. 다음 글에서는 실제 ML 모델을 KServe에 배포하고 추론 요청을 처리하는 방법에 대해 자세히 알아보겠습니다.