# 인공지능(Artificial Intelligence)

## 동영상 강의 (슬랙)

> * 슬렉 초청 링크 : https://join.slack.com/t/jnuai/shared_invite/zt-gy20hb2e-3kWox3sffiylvyhV_nVwEA </br>

> * 9월 8일 (수) 동영상 강좌: Class_2020_1
> <b>중요! 동영상을 보고 9월 8일 밤 11시까지 댓글을 달아야 출석이 인정됩니다. 기간 내에 댓글을 달지 않을 경우 예외없이 결석으로 처리됩니다.:</b></br>
> https://jnuai.slack.com/archives/C019MU09CBG </br>

> * 9월 9일 (화) 동영상 강좌: Class_2020_2:
> <b>중요! 동영상을 보고 9월 9일 밤 11시까지 댓글을 달아야 출석이 인정됩니다. 기간 내에 댓글을 달지 않을 경우 예외없이 결석으로 처리됩니다.:</b></br>
> https://app.slack.com/client/TNFK94R46/C01A86Z1L5Q/details/top

## 식물 성장 예측하기
> 몇일 후 잎의 길이와 너비가 얼마나 자랄 것인지를 예측함. <br/>
> https://www.kaggle.com/yungbyun/plant-diary-original/

## 식물 성장 예측하기 (작업중)
> https://www.kaggle.com/yungbyun/plant-diary-del

## 성별 알아맞히기
> 키/몸무게/발크기로 성별 알아맞추기 <br/>
> https://www.kaggle.com/yungbyun/female-male-classification-original (original)

## 성별 알아맞히기 (작업중)
> 작업중인 노트북: https://www.kaggle.com/yungbyun/female-male-classification-work

## 집값 예측하는 코드
> https://www.kaggle.com/yungbyun/fork-of-house-price-prediction-for-tutorial

> **각 모듈에 대한 간단한 설명입니다.**
> * **Pandas 판다스 : 데이터를 읽어들이고 유지하고 관리할 수 있는 멋진 모듈 (데이터베이스)**
> * **NumPy 넘파이 : 다양한 수치연산, 변환 기능 등을 갖는 멋진 모듈 (계산기)** 
> * **Seaborn 시본 : 데이터를 멋지게 표시하는 모듈 (엑셀)**
> * **sklearn 싸이킷런 : 머신러닝 모델을 만들 수 있는 멋진 모듈 (인공지능)**


## 추상화한 함수들

'''
import pandas as pd # 데이터 처리 모듈
import matplotlib.pyplot as plt # 데이터 시각화 모듈 
import seaborn as sns # 데이터 시각화 모듈
from sklearn.model_selection import train_test_split # 데이터 분할 모듈

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# data_f = load_csv('../input/plant-diary-new/plant_diary_new.csv')
def load_csv(file):
    import pandas as pd # 데이터 처리 모듈
    # CSV 파일 읽어오기
    imsi = pd.read_csv(file)
    return imsi;

def show_files(f):
    import os
    for dirname, _, filenames in os.walk(f):
        for filename in filenames:
            return os.path.join(dirname, filename)

# plot(data_f, 'day', 'leaf_length', 'owner')
def plot(df, _x, _y, _color_filed):
    # 읽어온 데이터 표시하기
    cl = df[_color_filed].unique()

    col = ['orange', 'blue', 'red', 'yellow', 'black', 'brown']

    fig = df[df[_color_filed] == cl[0]].plot(kind='scatter', x=_x, y=_y, color=col[0], label=cl[0])

    for i in range(len(cl)-1):
        df[df[_color_filed] == cl[i+1]].plot(kind='scatter', x=_x, y=_y, color=col[i+1], label=cl[i+1], ax=fig)

    fig.set_xlabel(_x)
    fig.set_ylabel(_y)
    fig.set_title(_x + " vs. " + _y)
    fig=plt.gcf()
    fig.set_size_inches(12, 7)
    plt.show()

# violin_plot(data_f, 'owner', 'leaf_length')
def violin_plot(df, _x, _y):
    plt.figure(figsize=(5,4))
    plt.subplot(1,1,1)
    sns.violinplot(x=_x,y=_y,data=df)
    
# heatmap(df, ['day', 'height', 'leaf_width', 'leaf_length', 'owner'])
def heatmap(dataf, cols):
    plt.figure(figsize=(12,8))
    sns.heatmap(data_f[cols].corr(),annot=True)

# show_cols(data_f)    
def show_cols(df):
    for col in df.columns: 
        print(col) 
        
# boxplot('owner', 'height')
def boxplot(a, b):
    f, sub = plt.subplots(1, 1,figsize=(7,5))
    sns.boxplot(x=data_f[a],y=data_f[b], ax=sub)
    sub.set(xlabel=a, ylabel=b);
    
# plot_3d('day', 'leaf_length', 'leaf_width')
def plot_3d(a, b, c):
    from mpl_toolkits.mplot3d import Axes3D

    fig=plt.figure(figsize=(12,8))

    ax=fig.add_subplot(1,1,1, projection="3d")
    ax.scatter(data_f[a],data_f[b],data_f[c],c="blue",alpha=.5)
    ax.set(xlabel=a,ylabel=b,zlabel=c)
    

def hist(df):
    data_f.hist(edgecolor='black', linewidth=1.2)
    fig = plt.gcf()
    fig.set_size_inches(14,12)
    plt.show()

    
def get_score_4_algo(a, b, c, d):
    # [1] 결정트리 예측기 머신러닝 알고리즘 학습
    gildong = DecisionTreeRegressor(random_state = 0)
    gildong.fit(train_X, train_y) #학습용 문제, 학습용 정답  
    
    # 점수 계산
    score1 = gildong.score(test_X, test_y) # 시험 문제, 시험 정답
    #print('Score:', format(score,'.3f'))
    # score의 의미: 정확하게 예측하면 1, 평균으로 예측하면 0, 더 못 예측하면 음수  

    # [2] 랜덤 포레스트 예측기 머신러닝 알고리즘
    youngja = RandomForestRegressor(n_estimators=28,random_state=0)
    youngja.fit(train_X, train_y)

    score2 = youngja.score(test_X, test_y)
    #print('Score:', format(score,'.3f'))
    
    # [3] K근접이웃 예측기 머신러닝 알고리즘
    cheolsu = KNeighborsRegressor(n_neighbors=2)
    cheolsu.fit(train_X, train_y)

    score3 = cheolsu.score(test_X, test_y)
    #print('Score:', format(score,'.3f'))
    
    # [4] 선형회귀 머신러닝 알고리즘
    minsu = LinearRegression()
    minsu.fit(train_X, train_y)

    score4 = minsu.score(test_X, test_y)
    #print('Score:', format(score,'.3f')) 
    
    plt.plot(['DT','RF','K-NN','LR'], [score1, score2, score3, score4])
    

def split_4_parts(df, li, dap_col):
    # 학습용(문제, 정답), 테스트용(문제, 정답)으로 데이터 나누기
    train, test = train_test_split(df, train_size = 0.8)

    # 학습용 문제와 정답
    a = train[li]
    b = train[[dap_col]]

    # 시험 문제와 정답
    c = test[li]
    d = test[[dap_col]]
    
    return a, b, c, d


# 다양한 분류 알고리즘 패키지를 임포트함.
from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘
#from sklearn.cross_validation import train_test_split # 데이타 쪼개주는 모듈 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

def get_rate_4_algo(train_X, train_y, test_X, test_y):
    gildong = svm.SVC() # 애기 
    gildong.fit(train_X,train_y) # 가르친 후
    prediction = gildong.predict(test_X) # 테스트
    print('인식률:',metrics.accuracy_score(prediction,test_y) * 100) 

    cheolsu = LogisticRegression()
    cheolsu.fit(train_X,train_y)
    prediction = cheolsu.predict(test_X)
    print('인식률:', metrics.accuracy_score(prediction,test_y) * 100)

    youngja = DecisionTreeClassifier()
    youngja.fit(train_X,train_y)
    prediction = youngja.predict(test_X) # 테스트
    print('인식률:',metrics.accuracy_score(prediction,test_y) * 100)

    minsu = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
    minsu.fit(train_X,train_y)
    prediction = minsu.predict(test_X) # 테스트
    print('인식률:',metrics.accuracy_score(prediction,test_y) * 100)
'''


## 과제 GitHub 링크 제출하는 곳
> 제출마감: </br>
> 제출링크: 
