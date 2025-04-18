---
title: "SGD / AdaGrad / RMSProp / Momentum / Adam"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-03-04 00:00:00 +0900
categories: [AI | 딥러닝 , Concept]
# categories: [AI | 논문 리뷰, Attention is all you need]
# categories: [MLOps | 인프라 개발, Kserve]
# categories: [Life | 일상 이야기, 와플먹으면서 공부하기]
tags: [DeepLearning, Optimizer, SGD, AdaGrad, RMSProp, Momentum, Adam]
description: "SGD(Stochastic Gradient Descent), AdaGrad(Adaptive Gradient Algorithm), RMSProp(Root Mean Square Propagation), Momentum, Adam(Adaptive Moment Estimation) Optimizer 에 대해 알아봅시다."
image: assets/img/posts/resize/output/optimizer-1.png # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://arxiv.org/abs/1609.04747">https://arxiv.org/abs/1609.04747</a></small>
</div>

> ## SGD(Stochastic Gradient Descent)

### 1. SGD Optimizer(확률적 경사 하강법)

SGD(확률적 경사 하강법, Stochastic Gradient Descent)는 경사 하강법의 일종으로, 각 단계에서 전체 데이터셋을 사용하는 대신 랜덤으로 선택된 하나의 샘플 또는 미니배치(batch)만을 사용하여 손실 함수의 경사를 계산하고, 이를 바탕으로 파라미터를 업데이트하는 방법입니다. SGD는 특히 데이터셋이 매우 클 때 계산 비용을 크게 줄일 수 있는 장점이 있습니다.

### 2. 동작 원리

SGD의 핵심 개념은 매번 하나의 샘플 또는 작은 배치(batch)에 대해 손실 함수의 기울기(경사)를 계산하고 그에 따라 모델의 파라미터를 업데이트하는 것입니다. 이를 통해 전체 데이터셋을 처리하는 Batch Gradient Descent(기존 Gradient Descent)보다 빠르게 학습할 수 있지만, 경사 계산이 불안정해져서 수렴 과정에서 진동이 발생할 수 있습니다.


$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

$$
(\nabla_\theta L(\theta_t) : 현재\ 파라미터\ \theta_t\ 에서의\ 손실\ 함수에\ 대한 \ 경사)
$$

$$
(\eta : 학습률)
$$

### 3.미니배치 SGD

실제 적용 시에는 보통 하나의 샘플이 아닌 미니배치(mini batch)를 사용하여 경사를 계산합니다. 미니배치의 크기가 $$m$$인 경우, SGD의 수식은 다음과 같이 확장됩니다.

$$
\theta_{t+1} = \theta_t - \eta \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta L(\theta_t; x_i, y_i)
$$

$$
(m:미니\ 배치\ 크기)
$$

$$
(\theta_t (x_i,y_i) :미니\ 배치\ 내의\ 데이터 샘플\ x_i와\ 레이블\ y_i)
$$

### 4. 장점과 단점

**장점**

- 계산 비용이 적음: 하나의 샘플 또는 작은 미니배치에 대해서만 경사를 계산하므로 대규모 데이터셋을 사용할 때도 빠르게 학습할 수 있습니다.

- 빠른 업데이트: 각 스텝마다 업데이트가 이루어지므로 빠르게 손실 함수를 감소시킬 수 있습니다.

**단점**

- 진동 현상: 경사 계산이 불안정해 최적화 과정에서 진동이 발생할 수 있으며, 최적의 해로 바로 수렴하지 않고 주변을 계속 맴돌 수 있습니다.

- 수렴 속도가 느림: 진동 때문에 최종적으로 수렴하는 데 시간이 걸릴 수 있습니다.

> ## AdaGrad

### 1. 동작 원리

AdaGrad는 파라미터마다 개별적인 학습률을 사용한다는 점에서 기존의 SGD와 차별화됩니다. 기본 아이디어는 자주 등장하는 파라미터에 대해서는 학습률을 낮추고, 드물게 등장하는 파라미터에 대해서는 학습률을 높이는 방식입니다. 이로 인해, 희소한 특성을 가진 데이터셋(예: 자연어 처리에서 자주 등장하지 않는 단어)에서 매우 효과적입니다.

AdaGrad의 핵심 개념은 각 파라미터의 과거 경사값들의 제곱을 누적하여 학습률을 조정하는 것입니다.

$$G$$: 각 파라미터의 과거 경사값들의 제곱 합을 기록합니다. 학습이 진행될수록 이 값이 누적되며, 이는 학습률을 조정하는 데 사용됩니다.

$$
G_{t+1} = G_t + g_t^2
$$

$$
(G_t: 각\ 파라미터에\ 대한\ 과거\ 경사값의\ 제곱\ 합)
$$

$$
(g_t: 시간\ t에서의\ 경사값)
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t
$$

$$
(G_t:\ 경사값의\ 제곱\ 합)
$$

$$
(\epsilon: 0으로\ 나누는\ 것을\ 방지하기\ 위한\ 작은\ 값\ (보통\ 10^{-8})
$$


$$G_t$$(경사값의 제곱 합): AdaGrad의 가장 큰 특징은 각 파라미터마다 경사값의 제곱 합을 누적하여 학습률을 계산한다는 점입니다.**학습이 진행되면서 자주 업데이트되는 파라미터의 학습률은 점차 감소하게 되고, 드물게 업데이트되는 파라미터의 학습률은 상대적으로 유지되거나 증가하게 됩니다.**

### 2. 장점과 단점

**장점**

- 희소한 데이터셋에 유리: AdaGrad는 자주 나타나는 특징에 대해서는 학습률을 낮추고, 드물게 나타나는 특징에 대해서는 학습률을 높이기 때문에, **희소한 특성(sparse feature)**이 많은 데이터셋에서 특히 효과적입니다. 예를 들어, 자연어 처리에서 자주 등장하지 않는 단어에 대해 학습률을 높여 더 빠르게 학습할 수 있습니다.

- 적응적인 학습률: 학습률이 파라미터별로 개별적으로 조정되기 때문에, 각 파라미터가 최적화되는 방식이 다를 수 있는 복잡한 문제에 유용합니다.

**단점**

- 학습률 감소 문제: 경사값의 제곱 합이 계속 누적되면서 학습률이 점점 더 작아지기 때문에, 학습이 진행될수록 학습률이 지나치게 감소하는 문제가 있습니다. 이는 학습이 초기에 빠르게 진행되다가, 나중에 너무 느려져서 최적해에 도달하지 못할 수 있습니다.

- 장기 학습의 어려움: 학습률이 너무 빠르게 감소하면, 장기적인 학습에서 모델이 수렴하지 못할 수 있습니다. 특히, 경사가 평탄해지거나 학습이 깊어질 때 최적화가 제대로 이루어지지 않는 문제가 발생할 수 있습니다.

> ## RMSProp

### 1. RMSProp Optimizer

RMSProp(Root Mean Square Propagation)은 AdaGrad의 단점을 보완하기 위해 제안된 옵티마이저입니다. AdaGrad는 희소한 특성을 가진 데이터셋에서 효율적이지만, 학습이 진행될수록 학습률이 지나치게 감소하는 문제가 있습니다. RMSProp은 학습률을 적절하게 유지하면서, 경사 하강법을 더 효과적으로 수행할 수 있도록 경사값의 제곱에 대한 이동 평균을 유지하여 학습률을 조정합니다.

### 2. 동작 원리

RMSProp은 매번 경사 하강을 할 때마다 모든 파라미터에 대해 적응적인 학습률을 적용합니다. 이를 위해, 경사값의 제곱에 대한 이동 평균을 사용하여 파라미터별로 학습률을 조정합니다. 이를 통해 경사가 크거나 작은 경우에 따라 학습률을 동적으로 조정할 수 있습니다.

이 방식은 매번 경사값의 크기에 따라 학습률을 다르게 적용하기 때문에, 학습 속도가 너무 느려지지 않으면서도 안정적으로 수렴할 수 있습니다.

$$E[g²]$$: 경사값의 제곱에 대한 지수 이동 평균입니다. 이 값은 경사가 큰 곳에서는 학습률을 줄이고, 경사가 작은 곳에서는 학습률을 늘리는 역할을 합니다.

$$
E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) g_t^2
$$

$$
(E[g 2]_t: 시간\ t에서의\ 경사값의\ 제곱에\ 대한\ 지수\ 이동\ 평균)
$$

$$
(ρ: 지수\ 이동\ 평균의\ 감쇠율\ (일반적으로 0.9))
$$

$$
(g_t: 시간\ t에서의\ 경사값)
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$

$$
(ϵ:\ 수치적\ 안정성을\ 위해\ 매우\ 작은\ 값\ (일반적으로\ 10^{−8})
$$

$$E[g²]$$ (경사값의 제곱에 대한 지수 이동 평균): **RMSProp의 핵심 아이디어는 경사값의 제곱에 대한 지수 이동 평균을 사용하여 학습률을 조정하는 것입니다.** 이를 통해 이전의 경사값 정보를 축적하여 학습률이 지나치게 감소하지 않도록 조절합니다.

$$ρ$$(감쇠율): 경사값의 제곱에 대한 지수 이동 평균을 얼마나 빠르게 감소시킬지를 결정하는 하이퍼파라미터입니다. **$$ρ$$ 값이 1에 가까울수록 이전 경사값들의 영향을 더 크게 반영하게 됩니다. 일반적으로 0.9로 설정됩니다.**

### 3. 장점과 단점

**장점**

- 적응적인 학습률: RMSProp은 경사값의 크기에 따라 파라미터별로 학습률을 조정합니다. 이는 경사 하강법에서 학습률이 지나치게 감소하거나 증가하는 것을 방지하고, 더 안정적인 수렴을 유도합니다.

- 빠르고 안정적인 수렴: RMSProp은 경사값이 크게 변동하는 문제를 완화하여 빠르고 안정적인 수렴을 보장합니다. 특히 경사가 작은 곳에서는 학습률을 늘리고, 경사가 큰 곳에서는 학습률을 줄여서 학습 과정을 최적화합니다.

- AdaGrad의 문제 해결: RMSProp은 AdaGrad의 단점인 학습률이 점점 작아져 학습이 정체되는 문제를 해결합니다. 경사값의 제곱에 대한 **지수 이동 평균**을 사용하여 적절한 학습률을 유지할 수 있습니다.

**단점**

- 복잡한 하이퍼파라미터 설정: RMSProp은 감쇠율 $$ρ$$, 학습률 $$η$$ , 안정성을 위한 $$ϵ$$ 등의 하이퍼파라미터를 설정해야 하며, 적절한 값을 찾는 것이 중요합니다.

- GPU 메모리 요구량 증가: 경사값의 제곱에 대한 지수 이동 평균을 계속 계산하고 저장해야 하기 때문에 메모리 사용량이 다소 증가할 수 있습니다.

> ## Momentum

### 1. Momentum Optimizer

Momentum Optimizer는 **SGD(확률적 경사 하강법)**의 단점을 보완하기 위해 제안된 방법 중 하나입니다. 특히 진동 현상을 줄이고, 최적해로 더 빠르게 수렴하도록 돕는 역할을 합니다. 기본적으로, 이전 경사의 방향을 기억해 이를 이용해 학습 속도를 가속화하는 방식으로 작동합니다.

SGD는 학습 중에 파라미터 업데이트가 단순히 현재 기울기(gradient)에만 의존하기 때문에 수렴 과정에서 진동하거나, 경사면이 평탄한 구간에서는 학습 속도가 느려질 수 있습니다. Momentum은 이전 단계의 이동 방향(속도, velocity)을 고려하여 이러한 문제를 해결하려 합니다.

### 2. 동작 원리

Momentum은 마치 물리학에서 물체가 일정한 질량을 가지고 가속하는 것과 비슷하게 작동합니다. 학습 과정에서 관성을 추가하여, 파라미터가 최적의 해로 더 빠르게 수렴하도록 돕습니다. 즉, 이전 단계의 업데이트를 참고하여 현재 경사를 조절하는 방식입니다.

Momentum의 파라미터 업데이트 방식은 아래 수식에 의해 결정됩니다.

Velocity (속도) $$v_t$$: 학습 과정에서 파라미터가 얼마나 빠르게 변화하는지를 나타내는 값입니다. 이는 이전 경사의 영향을 받습니다.

Momentum $$γ_t$$: 이전 속도(velocity)에 얼마나 가중치를 부여할지 결정하는 하이퍼파라미터입니다. 일반적으로 0.9로 설정되며, 0과 1 사이의 값을 가집니다.

$$
v_{t+1} = \gamma v_t + \eta \nabla_\theta L(\theta_t)
$$

$$
(v_t : 현재 시간\ t에서의\ 속도(velocity)
$$

$$
(\gamma : 모멘텀\ 계수(주로\ 0.9)
$$

$$
\theta_{t+1} = \theta_t - v_{t+1}
$$

$$
(v_{t+1} : 업데이트된 속도)
$$

### 3. 장점과 단점

**장점**

진동 감소: SGD의 주요 단점인 진동 현상을 줄여 줍니다. 특히 경사가 급격히 변하거나 최적해로 향하는 과정에서 발생하는 작은 변동을 줄여 보다 안정적인 학습이 가능합니다.

더 빠른 수렴: 이전 경사의 방향을 유지해 최적화 속도를 높일 수 있습니다. 평탄한 영역에서는 경사 하강 속도를 가속화하여 학습을 빠르게 진행시킬 수 있습니다.

지역 최적해(Local Minima) 탈출 가능성 증가: 모멘텀은 수렴 과정에서 지역 최적해에 갇히는 문제를 어느 정도 해결할 수 있습니다. 경사 방향을 누적하여 최적화 경로를 부드럽게 만들기 때문에, 지역 최적해에서 빠져나올 가능성을 높여 줍니다.

**단점**

복잡성 증가: SGD에 비해 모멘텀은 하이퍼파라미터인 $$γ$$ 값을 추가로 조정해야 합니다. 이 값을 적절하게 설정하지 않으면 학습이 잘 이루어지지 않을 수 있습니다.

과도한 속도: 너무 큰 모멘텀 값은 학습이 지나치게 가속화되어 최적해를 지나쳐버리거나, 안정적인 수렴이 어려워질 수 있습니다.

> ## Adam

### 1. Adam Optimizer

Adam(Adaptive Moment Estimation)은 딥러닝 모델을 훈련시키는 데 가장 널리 사용되는 옵티마이저 중 하나입니다. Adam은 Momentum과 RMSProp의 아이디어를 결합하여, 경사 하강법을 빠르고 안정적으로 수행할 수 있도록 합니다. Momentum의 개념을 사용해 경사값의 방향을 고려하고, RMSProp의 개념을 사용해 경사값의 크기에 따라 학습률을 조정합니다. 이를 통해, Adam은 학습 속도와 안정성을 동시에 확보할 수 있습니다.

### 2. 동작 원리

Adam은 두 가지 이동 평균을 추적합니다.

(1) 1차 모멘트($$m_t$$): 경사의 평균값 (경사의 방향)

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
(m_t :\ 시간\ t에서의\ 1차 모멘트\ (경사값의 지수 이동 평균))
$$

$$
(β_1:\ 1차\ 모멘트에\ 대한\ 감쇠율\ (일반적으로\ 0.9)
$$

$$
(g_t: 시간\ t에서의\ 경사값)
$$

(2) 2차 모멘트($$v_t$$): 경사 제곱의 평균값 (경사의 크기)

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
(v_t:\ 시간\ t에서의\ 2차\ 모멘트(경사\ 제곱값의\ 지수\ 이동\ 평균))
$$

$$
(β_2:\ 2차\ 모멘트에\ 대한\ 감쇠율\ (일반적으로\ 0.999))
$$

**편향 보정(bias correction): 초기 단계에서 $$m_t$$와 $$v_t$$는 0에 가까워져서 학습이 느려지는 경향이 있습니다. 이를 방지하기 위해 Adam은 편향 보정을 적용합니다.**

지수 이동 평균은 초기에 데이터가 충분히 누적되지 않았을 때(특히 학습 초기 단계), 계산된 $$m_t$$와 $$v_t$$값이 실제 경사 정보에 비해 과소평가될 수 있습니다. 이것이 편향(bias) 문제입니다.

**즉, 초기 단계에서는 경사 하강에서 파라미터 업데이트가 제대로 이루어지지 않는, 즉 학습이 매우 느리게 진행되는 결과를 초래합니다.**

Adam에서는 이 초기 편향 문제를 해결하기 위해 편향 보정을 적용합니다. 이를 통해 초기 단계에서 $$m_t$$와 $$v_t$$가 0에 가까워지는 문제를 완화하고, 학습 속도를 정상적으로 유지할 수 있습니다.

편향 보정은 단순히 $$m_t$$와 $$v_t$$를 편향 보정 계수로 나누는 방식입니다. 아래와 같은 수식을 사용합니다.

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
(\hat{m}_t:\ 1차\ 모멘트의\ 편향\ 보정\ 값)
$$

$$
(\hat{v}_t:\ 2차\ 모멘트의\ 편향\ 보정\ 값)
$$


이후, 파라미터 업데이트의 수식은

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

1차 모멘트와 2차 모멘트를 모두 고려하여 파라미터를 업데이트합니다. 1차 모멘트는 경사값의 방향을 나타내며, 2차 모멘트는 경사값의 크기를 고려하여 학습률을 조정합니다.

### 3. 장점과 단점

**장점**

- 빠른 수렴: Adam은 1차 모멘트(Momentum)와 2차 모멘트(RMSProp)를 결합하여 빠르게 수렴하는 특성을 가지고 있습니다. 경사의 방향과 크기를 모두 고려하기 때문에 효율적입니다.

- 적응적 학습률: 경사 크기에 따라 각 파라미터마다 학습률이 조정되므로, 모델이 각 파라미터에 맞게 적응적인 학습을 수행합니다. 이는 학습률을 수동으로 조정할 필요성을 줄여줍니다.

- 편향 보정: 초기 학습 단계에서 경사값의 편향을 보정해 줌으로써, 학습이 초반에 너무 느리게 진행되는 문제를 방지합니다.

- 복잡한 모델에 적합: 많은 파라미터를 가진 복잡한 모델에서도 효과적으로 동작하며, 특히 딥러닝에서 매우 널리 사용됩니다.

**단점**

- 과적합 가능성: Adam은 매우 빠르게 수렴하는 경향이 있으므로, 모델이 너무 빠르게 과적합(overfitting)될 수 있습니다. 이를 방지하기 위해서는 early stopping 또는 정규화 기법을 적용하는 것이 좋습니다.

- 학습률 조정 문제: 기본적으로는 적응적 학습률을 제공하지만, 일부 경우에서는 적절한 학습률을 찾기 위해 수동으로 설정해야 할 수도 있습니다.

> 참고 자료

- [https://en.wikipedia.org/wiki/Stochastic_gradient_descent#/media/File:Optimizer_Animations_Birds-Eye.gif](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#/media/File:Optimizer_Animations_Birds-Eye.gif)