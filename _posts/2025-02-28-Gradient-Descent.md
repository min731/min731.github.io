---
title: "Neural Network / Cost Function / Gradient Descent"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-02-28 00:00:00 +0900
categories: [AI | 딥러닝, Concept]
# categories: [AI | 논문 리뷰, Attention is all you need]
# categories: [MLOps | 인프라 개발, Kserve]
# categories: [Life | 일상 이야기, 와플먹으면서 공부하기]
tags: [DeepLearning, Neural Network, Cost Function, Gradient Descent]
description: "뉴럴 네트워크(Neural Network), 비용 함수(Cost Function), 경사하강법(Gradient Descent)에 대해 자세히 알아봅시다."
image: /assets/img/posts/resize/output/images.jpeg # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

>  *본 게시글은 유튜브 ['김성범[ 교수 / 산업경영공학부 ]' [ 핵심 머신러닝 ]뉴럴네트워크모델 1 (구조, 비용함수, 경사하강법)](https://www.youtube.com/watch?v=YIgLpsJ-1J4) 자료를 참고한 점임을 알립니다.

## Neural Network

### 1. 선형 회귀 모델
- 입력변수($$x$$)의 선형 결합을 통해 출력변수($$y$$)를 표현
- 출력변수($$y$$)는 실수의 범위 내에서 연속적인 값 => '연속형'
![](https://velog.velcdn.com/images/min0731/post/e78ae022-d21b-445e-a39c-6cee2df8d1eb/image.png)

### 2. 로지스틱 회귀모델
- 입력변수($$x$$)의 선형 결합값을 로지스틱 함수에 입력하여 비선형 결합($$\sigma$$)을 통해 출력변수($$y$$)를 표현
- 출력변수($$y$$)는 특정 범주 및 카테고리 중 하나의 값 => '범주형'
- 이진 범주형 : 0 또는 1 (True or False)
- 다중 범주형 : 맑음 또는 흐림 또는 비 또는 눈
![](https://velog.velcdn.com/images/min0731/post/fe101f95-c964-4048-82e2-781d39a340c0/image.png)

- 그림으로 표현
![](https://velog.velcdn.com/images/min0731/post/5f2f0930-da10-4209-9701-6289e321c610/image.png)

### 3. 다중 퍼셉트론(Multi-Layer Perceptron)
- 입력층 : 입력변수의 수 = 입력노드의 수
- 은닉층 
- 출력층 : 출력노드의 수 = 출력변수의 범주 개수(범주형), 출력 변수의 갯수(연속형)
![](https://velog.velcdn.com/images/min0731/post/173f8433-bdc5-427e-9278-6afd219b6df1/image.png)

- MLP(Multi-Layer Perceptron) == ANN(Artifical Nerural Networks)

### 4. 선형 회귀 / 로지스틱 회귀 / 뉴럴 네트워크 비교
- 선형 회귀 모델
$$
f(x) = w_{0} + w_{1}X_{1} + w_{2}X_{2}
$$

- 로지스틱 회귀 모델
$$
f(x) = \frac{1}{1 + e^{-(w_0 + w_1X_1 + w_2X_2)}}
$$

- 뉴럴 네트워크
$$
f(x) = \frac{1}{1 + e^ {-\left( z_{01} + z_{11} \left( \frac{1}{1 + e^{-(w_{01} + w_{11}X_{1} + w_{21}X_{2})}} \right) + z_{21} \left( \frac{1}{1 + e^{-(w_{02} + w_{12}X_{1} + w_{22}X_{2})}} \right) \right) }}
$$

### 5. 활성화 함수(Activation Function)
![](https://velog.velcdn.com/images/min0731/post/3f2daeed-fc52-4ff6-a871-b90c80b28e55/image.png)

###  6. 비용 함수(Cost Function)
- MSE
$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
- CrossEntropy
$$
L = - \sum_{i} t_i \log p_i
$$
- ex) 3 Classes에 대해 계산

![](https://velog.velcdn.com/images/min0731/post/cf74ff48-752d-4bbe-8005-736868f68c5a/image.png)


$$
-\sum_{i=1}^{4} t_i \log p_i = -[\ln(0.3) + \ln(0.8) + \ln(0.6) + \ln(0.4)] = 2.85
$$

$$
*\,CrossEntropy \,값을\,낮추는\,wegiht,\,bias을 탐색하는\,과정이\,Training \,과정
$$

### 7. 경사하강법 (Gradient Descent)
- Gradient Descent Method: First-Order Optimization Algorithm
- Optimization : 함수의 최솟값 혹은 최댓값을 찾는 과정
- Turning Points의 개수는 함수의 차수에 의해 결정
- 모든 Turning Point가 최솟값 혹은 최댓값은 아님
- 전역 최솟값(Global Minimum) : 최솟값들 중 가장 작은 최솟값
- 지역 최솟값(Local Minimum) : 지역적인 최솟값

![](https://velog.velcdn.com/images/min0731/post/1d4d1c8b-a42f-472a-9301-a5e1a39adf8e/image.png)

- 경사하강법(Gradient Descent Method)
![](https://velog.velcdn.com/images/min0731/post/544ca902-67e4-4b26-86cd-17dbb88c9020/image.png)

- 비용함수를 최소화하는 weight들을 찾고자할 때 활용하는 방법론
- gradient가 줄어드는 방향으로 weight들을 찾다보면 최솟값을 찾을 수 있음
![](https://velog.velcdn.com/images/min0731/post/d53907c6-f50b-4604-a542-cf7cf2a30508/image.png)
$$
w_{\tau+1} = w_{\tau} - \alpha \cdot L'(w_{\tau})
$$
$$
(w_{\tau+1} : 업데이트될\,weight,\,w_{\tau} : 현재\,weight\,,\, learning \,rate : 학습률(0 < \alpha < 1))
$$

- $$w_{\tau}$$에 따라 $$w_{\tau+1}$$가 증가 혹은 감소

> 참고 자료
  
- ['김성범[ 교수 / 산업경영공학부 ]' [ 핵심 머신러닝 ]뉴럴네트워크모델 1 (구조, 비용함수, 경사하강법)](https://www.youtube.com/watch?v=YIgLpsJ-1J4)