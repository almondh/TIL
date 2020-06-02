# Deep Neural Networks for Youtube Recommendations
2016

### Abstract
- 1. deep candidate generation model
- 2. deep ranking model


### 1. Introduction
- Youtube 동영상 추천은 크게 세가지 측면에서 어려움
  1. scale   : 방대한 사용자, corpus로 인해 고도로 전문화된 분산 학습 알고리즘 필요
  2. freshness : 초당 여러 영상 업로드 -> 추천 시스템이 새로운 컨텐츠뿐만아니라 최근 유저 액션에 즉각 대응할 수 있어야 함. (새 컨텐츠와 이전 영상의 균형)
  3. noise : f








개인화 추천은 풍부한 정보 환경에서의 **key**
- **Pure Search(Querying)** 와 **Browsing(directed or non-directed)** 의 결합으로 사용자가 효과적으로 정보를 탐색할 수 있도록 함
- 본 논문은 이전 활동을 기반으로 서명된 사용자에게 개인화된 비디오 세트를 제공하는 권장사항을 제시
- 권장사항은 http://www.youtube.com 와 http://www.youtube.com/videos 의 찾아보기 페이지에서 처리

#### 1.1 Goals
- 유저의 관심사에 부합하는 고품
