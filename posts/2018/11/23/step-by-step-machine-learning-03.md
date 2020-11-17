---
title: Step by Step machine learning - 03
date: 2018-11-24 04:44:01
published: true
tags:
  - machine-learning
  - scikit-learn
mathjax: true
description:
  'Classification 2 ## ROC 커브  ROC 커브는 Binary Classification에서 가장 잘
  사용하는 검증기다. 왼쪽 위로 그래프가 상승해 있을 수록 (우하단 면적이 넓을 수록) 좋은 분류기라고 할 수 있다.  ```py # ROC
  커브는 binary classifiers에서 잘 사용하는 검증기다. # 왼쪽 위로 그래프로 올라갈 수...'
category: machine-learning
slug: /2018/11/23/step-by-step-machine-learning-03/
template: post
---

Classification 2

## ROC 커브

ROC 커브는 Binary Classification에서 가장 잘 사용하는 검증기다. 왼쪽 위로 그래프가 상승해 있을 수록 (우하단 면적이 넓을 수록) 좋은 분류기라고 할 수 있다.

```py
# ROC 커브는 binary classifiers에서 잘 사용하는 검증기다.
# 왼쪽 위로 그래프로 올라갈 수록 좋은 분류기다.

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()
```

![ml-2-3.png](../images/ml-2-3.png)

이런 그래프가 아니고, 점수를 보고 싶다면

```py
# 점수로 보고 싶다면,
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
```

```
0.963161488618153
```

이 ROC 커브를 활용하여 `RandomForestClassifier`와 비교해보자.

```py
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, 'b:', label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()
```

![ml-2-4.png](../images/ml-2-4.png)

```py
roc_auc_score(y_train_5, y_scores_forest)
```

```
0.9922948905208268
```

RandomForestClassifier가 더 낫다는 것을 알 수 있다.

## Multiple Class Classifier

이런 다양한 클래스를 구분하는 Classifier에는 `RandomForestClassifier`와 `BayesClassifier`가 있다. 그리고 Binary Clasifier에는 `SupportVectorMachine`과 `LinearClassifier`가 있다.앞서서는 5인지 아닌지만 확인했다. 만약 1~10까지를
구별해야 한다면 어떻게 해야할까?

1. MultipleClassifier 를 사용하는 법
2. One Versus All: OVA) 0~9를 구별하는 binary 분류기 10개를 만들기. 즉 이것이 무슨 숫자인지 10개의 분류기를 모두 거치는 방법이다.
3. One Versus One: OVO) 이름 그대로, 각각 맞짱(?) 뜨는 방법이다. 0과 1분류기를 거치고, 0과 2분류기를 거치고, 0과 3분류기를 거치고... 이런식으로 가지수가 나올 수 있는 모든 분류기를 만드는 방법이다. 이 방법은 $$N\times(N-1)$$ 개의 분류기가 필요할 것이다. OVO는 시간은 물론 오래걸리지만, (45개) 트레이닝세트가 적게 필요하므로 더 이점이 있다.

SVM과 같은 일부 알고리즘은 트레이닝 세트의 크기에 민감해서, 크기가 커질 경우 성능이 뚝뚝 떨어지는데, OVO를 사용하면 더 빠르게 구별할 수 있다. 그러나, 이 외의 경우에는 OVA를 보통 선호 한다고 한다.

scikit-learn의 경우, 다중 클래스 분류에 binary-classification를 사용할 경우, 알아서 자동으로 OvA (SVM은 OVO)를 사용한다. 아래의 예제를 보자.

```py
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
```

```
array([5.])
```

이 코드에서는, 5만 구별한 `y_train_5`대신 원래 타겟을 사용하여 `SGDClassifier`를 훈련시켰다. 그런다음 예측을 하나 만들어 냈는데, 내부에서는 scikit-learn이 실제로 10개의 이진 분류기를 훈련시키고 각각의 결정점수를 얻어서 가장 높은 클래스 (5) 를 호출해 낸것이다.

```py
some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores
```

```
array([[-210257.90046371, -417036.80313781, -405317.41607626,
         -81007.86225196, -357657.92769475,   32836.21879306,
        -743346.95995739, -210459.08988114, -826317.66173769,
        -523922.97081691]])
```

```py
np.argmax(some_digit_scores)
```

```
5
```

이번에는 OvO 와 OvA를 수동으로 사용하게 해보자.

```py
# OVO
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42, max_iter=5, tol=None))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
```

```
array([5.])
```

5를 잘 예측하였다. `RaondomForestClassifier`의 경우는 더 쉽다.

```py
forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
```

```
array([5.])
```

각각에 대해서 확률을 보고 싶다면

```py
forest_clf.predict_proba([some_digit])
```

```
array([[0. , 0. , 0. , 0.1, 0. , 0.9, 0. , 0. , 0. , 0. ]])
```

90%의 확률로 5임을 예측했다는 것을 알 수 있다. 이제 한번 테스트를 해보자.

```py
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')
```

```
array([0.84953009, 0.84259213, 0.82712407])
```

대략 85%의 확률로 맞추고 있는 것을 볼 수 있다. 여기서 성능을 더 향상 시키기 위해서는, input값을 조절해보면 될 것이다.

```py
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')
```

```
array([0.90586883, 0.91309565, 0.91003651])
```

정확도가 약 90%까지 향상한 것을 알 수 있다.
