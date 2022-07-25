# k-최근접 이웃 회귀 알고리즘 [7/24]

- 주변의 가장 가까운 K개의 샘플을 통해 값을 예측하는 방식 (예 : 평균)
- KNeighborsRegressor 클래스
    
    ```java
    class sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, *, weights='uniform', algorithm='auto',
    leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
    ```
    
    - n_neighbors → 이웃의 수 k (default는 5)
    - weights → 가중 방법 (default는 `‘uniform’`)
        - `‘uniform’` → 각각의 이웃이 동일한 가중치를 가짐
        - `‘distance’`→ 거리가 가까울수록 더 높은 가중치를 가짐
        - `‘callable’` → 사용자가 직접 정의한 함수 사용 (입력 : 거리가 저장된 배열, 출력 : 가중치가 저장된 배열)
    - algorithm → 가장 가까운 이웃 계산하는 알고리즘 (default는 auto)
        - `‘auto’` → 훈련 데이터에 기반하여 가장 적절한 것 선택
        - `‘ball_tree’` → Ball-Tree 구조 [https://nobilitycat.tistory.com/entry/ball-tree](https://nobilitycat.tistory.com/entry/ball-tree)
        - `‘kd_tree’` → KD-Tree 구조
        - `‘brute’` → Brute-Force 탐색
    - leaf_size → KD-Tree의 leaf size 결정 (default는 30)
    - p → 민코프스키 미터법(Minkowski)의 차수 결정
        - p=1 → 맨해튼 거리
        - p=2 → 유클리드 거리
