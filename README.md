# Implementation of SVM Classifier

## Goals and Overview

***Goals and Overview:*** You will be assigned to implement SVM classifier, a type of linear classifier, to understand the basic concept. You should use CIFAR-10 dataset for classification. <br/><br/>


## Summary

### SVM Classifier Structure ###

![img1](/img1.png)
<br/><br/>

***Forward pass***

- Input

  Bias trick 을 수행한 파라미터 W 와 CIFAR-10 이미지 데이터를 가져온다.

- Score function
  
  dot(x, W) 로 각 클래스에 대한 score 가 계산된다.

- Loss function

  SVM 을 loss function 으로 이용하여 loss 를 계산한다. 여기서 사용될 multiclass SVM function 은 Structured SVM 으로 자세한 공식은 [Weston and Watkins 1999 (pdf)](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es1999-461.pdf) 에서 확인 할 수 있다. Hyperparameter 인 delta 값은 1.0 으로 설정한다. <br/><br/>

- Regularization

  L2 norm 을 이용한 regularization term 을 더한다. 이 때 hyperparameter 인 regularization strength 가 곱해진다.

- Loss

  최종 loss 계산식은 다음과 같다.
  
![img2](/img2.png)
<br/><br/>

***Backward pass***

- Local gradient of margins

  Local gradient of scores 를 바로 구하기 위해 skip 하였다.

- Local gradient of scores

  Score function W*xi 를 통해 dscores / dW = xi 임을 알 수 있다.

- Calculate dW

![img3](/img3.png)
<br/><br/>

  위의 첫번째 식을 통해 원래 해당하는 클래스에 대한 dW 를 구할 수 있다.

![img4](/img4.png)
<br/><br/>

  위의 두번째 식을 통해 해당하지 않는 클래스에 대한 dW 를 구할 수 있다.
  
- Output

  Output 으로 loss 와 dW 를 return 한다.
  
<br/><br/>

***Optimization***

- Gradient Descent

  Hyperparameter 인 iteration 과 step size(learning rate) 값을 설정한 후 optimization 을 위한 gradient descent 를 수행한다. While 문에서 매 iteration 마다 loss 와 dW 를 return 받고, 현재 W 에 의한 test data 의 prediction 을 수행한 후 accuracy 를 계산한다. 매 100회의 iteration 마다 결과를 프린트한 후, W 를 update 시킨다. 추가적으로, graph visualization 을 위해 결과값을 추가적인 리스트에 저장한다.

- Mini-batch Gradient Descent(Stochastic Gradient Descent)

  Full data 에 대해 optimization 을 계속 한다면 비효율적이고 overfitting 이 되어 generalization 이 잘 안되는 문제점이 발생 할 수 있다. 따라서 전체 train data 에서 256 개를 random sampling 하여 optimization 을 수행하는 mini-batch gradient descent 과정을 추가하였다. Sampling number 는 학습시키는 장비에 따라 달라질 수 있는데 보통 2의 지수로 하여 자신의 GPU의 메모리 성능에 맞게 설정해주면 된다(이번 프로젝트에서는 GPU acceleration 을 적용하지 않았기 때문에 CPU에 맞게 설정해주어야 하고 사용 CPU 가 쿼드코어이기 때문에 2^4=16 개의 sampling number 을 설정해야 하는 것으로 보인다. 하지만 수행 결과 2^8 개의 sampling number 을 설정하여도 optimization 속도에 크게 무리가 없었으므로 256 이라는 값을 사용하였다).

  **(Stochastic Gradient Descent 개념을 혼동하여 위 과정에는 오류가 있다. 추후 수정 예정.)**
<br/><br/>

***Visualization graph***

결과값 loss 와 accuracy 를 iteration 에 따른 그래프로 visualization 한다. <br/><br/>

***Visualization W1***

Parameter W 를 visualization 하기 위해,

- W 를 visualization의 의미

  SVM classifier 에서 W 와 X 의 내적의 의미는 X 들을 고차원 feature space 로 보냈을 때 decision boundary 인 hyperplane 의 support vector 와 X 의 거리(정확하게는 W의 크기로 나누어진다면)를 의미한다. 즉, SVM 에서 optimization 은 hyperplane 의 normal vector 방향으로의 X 의 margin 을 최대화시키는 과정이다. 따라서 W 의 하나의 class 에 해당하는 vector 는 해당 클래스의 이미지들의 score 를 높이는 방향으로 업데이트된 normal vector 이고 이 vector 를 reshape 하여 visualization 을 하는 것이다.
 
- Negative value 에 대한 처리

  W 의 element 에는 negative value 도 있는데 이 프로젝트에서는 W 의 element 중 최소값을 0으로, 최대값을 1로 한 후 255를 곱하여 linear 하게 확장시키는 방법을 선택하였다. 물론 vector 를 이루는 모든 단위벡터들이 서로 linear 하지 않을 것이기 때문에 한계점이 존재하므로 이 부분에 대해서는 추가적인 개선이 필요할 것이다.
 
  마지막으로 W 의 data type 은 float 이기 때문에 image 로 출력하기 위해 integer type 으로 바꾸어 imshow 를 하였다. <br/><br/>


### Code description ###

- SVM_Classifier.py

  이 함수에서는 반복문을 줄임으로써 더 간결하고 빠른 실행을 위해 half-vectorized 방식을 선택하였다. SVM loss function and regularization 의 margins 를 계산하는 과정에서 이를 위한 코드를 볼 수 있다. Scores 로 주어진 행렬에 대해 매 행과 열을 도는 대신 scores 의 각 클래스에 해당하는 점수들만 append(newaxis) 시켜서 scores 에 빼준것을 확인 할 수 있다.

  바로 그 다음 코드는 자기 클래스를 뺀 경우에 해당하는 element 들을 0 으로 만드는 과정이다. 다음으로 Calculate dW 에서 dSVM / dscores 으로 볼 수 있는 mask 변수를 기용한 후 dscore / dW 로 곱하여(여기서는 dscores 와 같다.) dW를 계산하는 과정을 확인 할 수 있다. mask 변수를 자세히 보면 맞는 클래스에 대해서는 큰 음수의 값을, 나머지 0이 아닌 element 들에 대해서는 모두 1의 값을 가지고 있고, calculate dW 의 공식 두개와 완벽하게 일치함을 확인 할 수 있다.

  마지막으로 regularization term 의 gradient 를 추가하기 위해 lamda*W*W 을 미분한 식인 2 * lamda * W 를 더해주었다. <br/><br/>

### Result ###

- *Hyperparameter* <br/>
Delta : 1.0 <br/>
Regularization strength : 0.000005 <br/>
Iteration : 10000 <br/>
Step size(Learning rate) : 0.00000001 <br/><br/>

hyperparameter 에 대한 cross-validation 과정과 gradient check 과정은 생략하였다. 하지만 각 hyperparameter 에 대해 다음과 같은 현상을 관찰 할 수 있다.

Regularization strength : lamda 값이 크면 계산되는 loss 가 너무 커서 매 update 시에 제대로된 optimization 이 불가능하였다. 따라서 충분히 작은 값 (lamda < 0.00001) 을 통해 이 문제를 해결하였다.

Iteration : 아래의 graph 에서도 확인 할 수도 있지만 iteration 은 학습하기 충분할 정도로 커야 하지만 일정 수준에서 saturation 되기 때문에 적당하게 설정하였다.

Step size(Learning rate) : 이 hyperparameter 또한 lamda 값과 마찬가지로 loss 에 굉장히 민감한 영향을 준다는 사실을 확인하였다. Step size 가 너무 크면 overshooting 현상이 일어나 제대로 loss function 의 convex 안으로 들어가지 못해 loss 가 증가하고, 조금만 커도 loss function 의 최저점을 잘 찾지 못하고 위아래로 진동하게 된다. 또, 너무 작은 값을 설정한다면 update 가 매우 느리고 local minimum 으로 빠질 위험이 있다. 따라서 여러 번의 시도로 적절한 값을 설정해 주어야 한다. 여기서 step size 값과, W 의 scale 을 낮추기 위해 곱해준 값(0.0001) 과 regularization strength 값이 서로 상충되어 최종 loss 값이 산출된다는 사실을 느낄 수 있었는데, 해당 3개의 값이 균형을 이루지 않고 어느 하나가 너무 크다면 loss 가 예상보다 훨씬 큰 값으로 scaling 되기 때문에 주의하여야 한다. <br/><br/>

- *Loss and Accuracy* <br/>
Gradient descent <br/>
Loss : 4.351582 <br/>
Accuracy : 0.379900 <br/>
Mini-batch gradient descent <br/>
Loss : 2.926527 <br/>
Accuracy : 0.335500 <br/><br/>

- Visualization graph

![graph2](/graph2.PNG)
<br/><br/>

- Visualization W

![vis_W1](/vis_W1.PNG)
<br/><br/>
