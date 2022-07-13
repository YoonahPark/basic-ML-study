# 주피터 노트북 활용 (7/11)

## 주피터 노트북 환경설정

1. Anaconda prompt 실행

    1. 가상환경 생성
        
        `conda create -n 가상환경이름 python=3.7`
        
    2. 가상환경 리스트 확인
        
        `conda env list`
        
    3. 가상환경 활성화
        
        `conda activate 가상환경이름`
        
    4. 가상환경 비활성화
        
        `conda deactivate`
        
2. 주피터 노트북 실행

    1. 가상환경 내에 주피터 노트북 설치
        
        `pip install jupyter notebook`
        
    2. 가상환경에 kernel 연결
        
        `python -m ipykernel install —user —name 가상환경이름`
        
    3. 주피터 노트북 실행
        
        `jupyter notebook`
        
    4. 내가 만든 가상환경 지정
        
        
## .ipynb to .py
[.ipynb to .py](https://beyonddata.tistory.com/entry/ipynb-to-py-%EB%B3%80%ED%99%98-%EB%B0%A9%EB%B2%95%EC%A3%BC%ED%94%BC%ED%84%B0-%EB%85%B8%ED%8A%B8%EB%B6%81-to-%ED%8C%8C%EC%9D%B4%EC%8D%AC)



# k-최근접 이웃 알고리즘 (k-NN)

[[머신러닝] K-최근접 이웃(K-NN) 알고리즘 및 실습](https://rebro.kr/183)

☝️ 혼자 공부하는 머신러닝+딥러닝  기반 블로그 정리

### k-NN 알고리즘이란?

- 주변의 가장 가까운 K개의 데이터를 보고 데이터가 속할 그룹을 판단하는 알고리즘이 K-NN 알고리즘이다.
- 데이터 간의 거리는 **‘유클리드 거리’**로 한다.
- 최선의 K를 선택하는 것은 데이터마다 다르게 접근해야 하는데, 일반적으로는 총 데이터 수의 제곱근 값을 사용하며, 홀수로 설정하는 것이 좋다.
- K-NN과 같은 거리 기반 알고리즘을 사용할 땐 데이터를 표현하는 기준이 다르면 알고리즘이 올바르게 예측할 수 없다. 따라서 일정한 기준으로 맞춰주어야 하는데, 이런 작업을 데이터 전처리(Data preprocessing)라고 한다.
    - 흔히 사용하는 방식으로 최소-최대 정규화(min-max normalization), z-점수 표준화(z-score standardization)가 있다.
    - 두 방식 중에서는 주로 z-점수 표준화를 많이 사용한다. 최소-최대 정규화는 테스트 셋의 최소/최대가 범위를 벗어나는 경우가 생길 수 있기 때문이다.

