## DACON
### 에너지 빅데이터 활용 데이터 사이언스 콘테스트 - 전력수요예측
      본 대회에서는 2016년 7월 20일부터 2018년 6월 30일까지의 국내 특정 지역의 아파트들과 상가의 전력에너지 사용량이 주어집니다. 
     * 주의: 제공되는 데이터에는 결측치나 이상치(NA, 0인 값)가 포함되어 있습니다.
       참고로, NA가 발생한 경우, 직전 시간의 전력사용량 값이 상당히 큰 경향이 있습니다. 
       이는 미터링 데이터 수집 시스템의 특징으로 보입니다. 그러나 반드시 그런 것은 아닙니다.

     *jupyter notebook이 열리지 않는다면, https://nbviewer.jupyter.org/ 에 접속해서 해당 주소를 입력하거나, 다운받아주세요.
##### - File
      1. EP demand prediction_1.ipynb         :  NA 중 중앙값보다 큰 값들 평균으로 처리, 나머지 최빈값으로 처리 도중 중단
      2. Ep demand prediction_2.ipynb         :  보간법 2가지로 기상데이터 NA 처리
      3. Ep demand prediction_3.ipynb         :  ARIMA, LSTM으로 예측
      4. KNN imputation.r                     :  knn으로 기상데이터 NA 처리
      5. MICE imputation.r                    :  MICE로 원데이터 NA 처리
      6. houly data to daily,month data.ipynb :  시간별 데이터를  일별, 월별로 변경
     7. interpolation_1 imputation.py        :  test 데이터에 보간법 method 1으로 첫 NA들 이후 NA 
     9. LSTM_test_x7.py                      :  test['X7']에 대한 LSTM
     10. DACON_ARIMA.ipynb                   :  colab으로 ARIMA
##### - 참조 사이트
      1. https://otexts.com/fppkr/complexseasonality.html
      2. https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-9-time-series-analysis-in-python-a270cb05e0b3
      3. https://rstudio-pubs-static.s3.amazonaws.com/192402_012091b9adac42dbbd22c4d07cb00d36.html
      4. https://datascienceplus.com/imputing-missing-data-with-r-mice-package/
##### 문제
      본 대회에서는, 기존 전력 사용 기록과 기상 데이터 등 공공 데이터를 이용하여, 
      각 가정 및 회사의 시간별, 일별, 월별 전력 사용량을 예측합니다. 
      2018년 7월1일부터 2018년 11월 30일까지의 에너지 사용량을 예측합니다. 
      보다 정확히는 다음을 예측합니다.
          O 2018년 7월 1일 00시부터 24시까지, 24시간, ‘시간당 전력사용량’ (24게)

          O 2018년 7월1일부터 7월10일까지, 10일간, ‘일간 전력사용량’ (10개)

          O 2018년 7월부터 11월까지, 5개월간, ‘월간 전력사용량’ (5개)

          즉, 각 세대(또는 상가)당 39개(24개,10개,5개)의 값을 예측해야 합니다.

##### 평가
     평가 지표(Metric)은 SMAPE입니다. SMAPE는 Symmetric Mean Absolute Percentage Error입니다. MAPE의 경우 Actual 값이 0이거나 작은 경우 MAPE 값이 지나치게 커지는 문제가 있습니다. SMAPE는 이를 개선한 Metric입니다.
     SMAPE.PNG

     n은 모든 예측 개수(세대수*39개)입니다.
