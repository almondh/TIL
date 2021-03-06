## DACON
### 반도체 박막 두께 분석 경진대회
    최근 고사양 반도체 수요가 많아지면서 반도체를 수직으로 적층하는 3차원 공정이 많이 연구되고 있습니다. 
    반도체 박막을 수십 ~ 수백 층 쌓아 올리는 공정에서는 박막의 결함으로 인한 두께와 균일도가 저하되는 문제가 있습니다. 
    이는 소자 구조의 변형을 야기하며 성능 하락의 주요 요인이 됩니다.
    이를 사전에 방지하기 위해서는 박막의 두께를 빠르면서도 정확히 측정하는 것이 중요합니다.
    박막의 두께를 측정하기 위해 광스펙트럼 분석이 널리 사용되고 있습니다. 반도체 소자의 두께 분석 알고리즘 경진대회를 개최합니다. 

- 평가함수 : MAE

##### - File
     1. keras_4.py            
          (1) 출력층 softplus보다 linear 함수 성능 우세
          (2) 은닉층 relu보다 softplus 함수 성능 우세
          (3) 은닉층 4개 사용하는것이 속도, 효율측면에서 우세
          (4) dropout 추가 -> 성능 우세
          (5) lr = 0.01 사용 -> 성능 우세
     2. keras_4.h5
          : keras_4.py를 활용하여 학습시킨 모델
