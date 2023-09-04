#!/usr/bin/env python
# coding: utf-8

# ## 1. 데이터 불러오기

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# > ** 몇 가지 필요한 모듈들을 import 합니다.**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split  # 함수


# > **각 모듈에 대한 간단한 설명입니다.**
# > * **Pandas 판다스 : 데이터를 읽어들이고 유지하고 관리할 수 있는 멋진 모듈 (데이터베이스)**
# > * **NumPy 넘파이 : 다양한 수치연산, 변환 기능 등을 갖는 멋진 모듈 (계산기) ** 
# > * **Seaborn 시본 : 데이터를 멋지게 표시하는 모듈 (엑셀)**
# > * **sklearn 싸이킷런 : 머신러닝 모델을 만들 수 있는 멋진 모듈 (인공지능)**

# > **판다스 모듈은 CSV(Comma-Separated Values, 콤마로 분류되는 값을 갖는) 파일을 읽어들일 수 있습니다. **
# 
# > **다음은 csv 파일의 예입니다. **
# > ![](https://i.imgur.com/KQGT0La.png)

# > **판다스 모듈로 CSV 파일에 있는 데이터를 읽어들이고 표시합니다. 15,035개 데이터 중 처음 10개만을 표시해봅니다.**

# In[ ]:


def read(aaa):
    imsi = pd.read_csv(aaa) #df = data frame
    print(imsi.head(10))
    return imsi

train_df = read('../input/train.csv')


# > **어떤 컬럼(특징)들이 있는지를 알 수 있습니다.**

# In[ ]:


print(train_df.columns)


# > **각 컬럼별 몇 개가 있는지, 데이터 유형은 어떤지 등에 대해서도 알 수 있습니다.**

# In[ ]:


train_df.info()


# > ** 실행 결과를 보면 다음과 같은 내용을 알 수 있겠네요.**
# > * **컬럼(특징) 수가 모두 21개**
# > * **(ID: 주택ID, Date: 거래일, Price: 주택가격, Bedrooms: 방 수, Bathrooms: 욕실 수, Sqft_Living: 주택 크기, Sqft_Lot: 차고지 면적, Floors: 층 수, Waterfront: 호수(바다)전망, View: 뷰, Condition: 상태, Grade: 평가점수, Sqft_Above: 지상면적, Sqft_Basement: 지하면적, Yr_Built: 건축년도, Yr_Renovated: 리모델링 년도, Zipcode: 우편번호, Lat: 위도, Long: 경도, Sqft_Living15: 2015년도 면적, Sqft_Lot15: 2015년도 차고지 면적**
# > * **가격 정보(데이터) 수가 자그마치 15,035개**
# > * **대부분 숫자 데이터 : float64(5), int64(15)**
# > * **비어있는 데이터가 없음.** 
# 
# > **비어있는 데이터가 없으니 다행입니다. 이제 각 컬럼(특징)별로 유일한 값이 몇개나 있는지를 확인해보겠습니다.**

# In[ ]:


# 컬럼별 유닉한 값의 수
def show_unique_column(df):
    for column in df:
        print(column,':', df[column].nunique())

show_unique_column(train_df)


# > **가령 주택 등급(grade)에는 12가지가 있네요. 주택 층수(floors)는 6가지가 있습니다. 호수(바다)전망을 의미하는 waterfront는 2가지가 있네요. 있다/없다 인듯 합니다. **

# ## 2. 데이터 시각화와 분석

# > **사람의 키와 발 사이즈는 어떤 관계가 있을까요?** <br/>
# 
# > **키가 크면 대체로 발도 큽니다. 즉, 키와 발 크기의 관계는 비례관계에 있습니다.**

# In[ ]:


# 지정한 컬럼들 간의 관계를 그래프로 그림. 이때 h로 지정된 컬럼의 값에 따라 색을 달리 표시함.
def pairplots(df, cols, h): 
    plt.figure(figsize=(10,6))
    sns.plotting_context('notebook',font_scale=1.2)
    g = sns.pairplot(df[cols], hue=h,size=2)
    g.set(xticklabels=[])

pairplots(train_df, ['sqft_lot','sqft_above','sqft_living', 'bedrooms','grade'], 'bedrooms')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# > **위 그림에서 주택 가격(price)과 상관 관계가 비교적 뚜렷이 나타나는 것은 무엇일까요?**
# 
# > **참고로, 카테고리형 데이터가 섞여 있는 경우에는 hue 인수에 카테고리 이름을 지정하면 그 카테고리 값에 따라 색상을 다르게 하여 표시할 수 있습니다. 예를 들어, 키와 발 사이즈의 관계 그래프를 표시할 때 성별을 hue로 설정하면 남/녀가 다른 색으로 표시됩니다.**
# 
# > **다음은 주택 가격과 집 크기의 관계만 따로 그린 것입니다. hue 속성을 A로 지정하면 A에 따라 색을 달리하여 표시해 보았습니다.**

# In[ ]:


def lmplot(df, a, b, c):
    # sqft_living과 price간의 관계를 표시하되 등급(grade)을 다른 색으로 출력함.
    sns.lmplot(x=a, y=b, hue=c, data=df, fit_reg=False)

lmplot(train_df, 'sqft_living', 'price', 'grade')


# In[ ]:


lmplot(train_df, 'sqft_living', 'price', 'condition')


# In[ ]:


lmplot(train_df, 'sqft_living', 'price', 'waterfront')


# > **각 컬럼별 상관관계를 heatmap(열지도)으로 표현해보겠습니다.**

# In[ ]:


def heatmap(df, columns):
    plt.figure(figsize=(15,10))
    sns.heatmap(df[columns].corr(),annot=True)
    #CHECK THE PPT SLIDE

heatmap(train_df, ['sqft_lot','sqft_above','sqft_living', 'bedrooms','grade','price'])


# > **가격과 가장 상관관계가 큰 것은 무엇일까요? **
# 
# >**다음은 방 수(bedrooms)에 따른 주택 가격를 boxplot으로 그려본 것입니다.**

# In[ ]:


def boxplot(df, a, b):
    f, sub = plt.subplots(1, 1,figsize=(12.18,5))
    sns.boxplot(x=df[a],y=df[b], ax=sub)
    sub.set(xlabel=a, ylabel=b);

boxplot(train_df, 'bedrooms', 'price')


# > **boxplot은 가격의 최소값, 최대값, 중간값, 이상치(아웃라이어, outlier)을 보여줍니다.**
# 
# > **두 개의 그림을 함께 표시할 수도 있네요.**

# In[ ]:


f, sub = plt.subplots(1, 2,figsize=(12,4))
sns.boxplot(x=train_df['bedrooms'],y=train_df['price'], ax=sub[0])
sns.boxplot(x=train_df['floors'],y=train_df['price'], ax=sub[1])
sub[0].set(xlabel='Bedrooms', ylabel='Price')
sub[1].yaxis.set_label_position("right")
sub[1].yaxis.tick_right()
sub[1].set(xlabel='Floors', ylabel='Price')


# > **등급(grade)의 경우 13개가 있는데, 등급에 따른 주택 가격(price) 정보를 보면 다음과 같습니다.**

# In[ ]:


boxplot(train_df, 'grade', 'price')


# > **3차원으로 그릴 수도 있네요. 다음은 방 수, 층 수, 집 크기 3개 값에 따라 점을 찍어 본 예입니다.**

# In[ ]:


def plot_3d(df, a, b, c):
    from mpl_toolkits.mplot3d import Axes3D

    fig=plt.figure(figsize=(12,8))

    ax=fig.add_subplot(1,1,1, projection="3d")
    ax.scatter(df[a],df[b],df[c],c="darkred",alpha=.5)
    ax.set(xlabel=a,ylabel=b,zlabel=c)

plot_3d(train_df, 'floors', 'bedrooms', 'sqft_living')


# > **다음은 방 수, 주차 면적, 집 크기 3개 값에 따라 점을 찍어 본 예입니다.**

# In[ ]:


plot_3d(train_df, 'sqft_living', 'sqft_lot', 'bedrooms')


# > **이상으로 데이터 표현하고(visualization) 분석해 보았습니다.**

# ## 3. 집값 예측하기

# > **이제 이러한 데이터를 이용하여 주택 가격을 예측하는 시스템을 만들어보겠습니다** 

# > **주택 가격 예측 시스템을 만들텐데, 우선, 데이터 컬럼 중 id와 date는 주택 가격과는 전혀 무관한 정보입니다. 따라서 이를 제거합니다.**

# In[ ]:


def drop(df, col):
    df = df.drop(['id', 'date'], axis=1) 
    
drop(train_df, ['id', 'date'])


# >**이제까지 살펴 본 traincsv 데이터를 80:20 두 부분으로 나누겠습니다. 80% 데이터는 모델을 학습시키는데 사용하고, 나머지 20%는 학습이 얼마나 잘되어 있는지를 확인하는데 사용합니다. **

# ### 3.1 데이터 쪼개기 (학습 데이터, 테스트 데이터)

# In[ ]:


def split(df):
    i, j = train_test_split(train_df, train_size = 0.8, random_state=3)  # 3=seed
    return i, j
    
a, b = split(train_df)


# In[ ]:


print(train_df.shape, a.shape, b.shape)


# ### 3.2 선형회귀 모형을 이용한 예측

# > **다음은 길동(gildong)이라는 이름의 머신러닝 모델(선형회귀모델)을 만든 후 학습(fit)시키는 코드입니다. **
# 
# > **80% 데이터에서 집 크기(sqft_living)만을 뽑아내어 입력으로 주고, 그에 대한 정답도 같이 주며 fit 함수를 호출합니다. **

# In[ ]:





# > **sklearn, 싸이킷 런(https://scikit-learn.org/stable/), 이건 무엇일까요? **
# > * **scikit-learn은 Machine Learning in Python**
# > * **간단하고 효과적인 머신러닝, 데이터 분석 도구**
# > * **누구나 사용할 수 있고 다양한 곳에 재사용 가능**
# > * **NumPy, SciPy, 그리고 matplotlib를 이용하여 구현**
# > * **오픈소스이면서 상업적으로도 이용가능(BSD 라이센스)**

# > **길동이 학습이 끝났습니다. 이제 남은 20% 데이터에서 집 크기(sqft_living)만을 뽑아 길동이에게 주어 주택 가격을 예측하도록 합니다. 예측한 값을 정답과 비교하면 점수(score)를 계산할 수 있습니다.**

# In[ ]:


# 학습 데이터의 집크기, 가격을 주고 학습시킴(fit).
# 테스트 데이터의 집크기를 주고 알아맞춘 후 정답과 비교하여 스코어를 구함. 

from sklearn import linear_model

gildong = LinearRegression()
    
gildong.fit(train_df_part1[['sqft_living']], train_df_part1[['price']])

score = gildong.score(train_df_part2[['sqft_living']], train_df_part2['price'])
print(format(score,'.3f'))


# > **여러 컬럼(특징)들 중에서 오직 주택 면적(sqft_living) 정보만을 입력으로 주었더니 그럭저럭 0.497 정도 얻었습니다. ** <br/>
# >**예측한 주택 가격을 표시할 수도 있습니다. **

# In[ ]:


predicted = gildong.predict(train_df_part2[['sqft_living']])
print(predicted, '\n', predicted.shape)


# In[ ]:


print('실제 정답')
print(train_df_part2['price'])


# > **선형 모델 직선의 기울기와 상수의 값을 출력해보면 다음과 같습니다.**

# In[ ]:


print('Intercept: {}'.format(gildong.intercept_))
print('Coefficient: {}'.format(gildong.coef_))


# ### 3.3 더 많은 특징 이용한 선형 회귀 모형

# > **주택 크기 외에 방과 화장실 수도 추가해서 학습시켜보겠습니다.**

# In[ ]:


features = ['sqft_living','bedrooms','bathrooms']


# > **80% 데이터로 길동이에게 학습하도록 시킨 후 나머지 20%를 주면서 주택 가격을 예측하도록 해보니 결과가 조금 더 좋게 나왔네요.**

# In[ ]:


gildong = LinearRegression()
gildong.fit(train_df_part1[features], train_df_part1['price'])
score = gildong.score(train_df_part2[features], train_df_part2['price'])
print(format(score,'.3f'))


# > **이번에는 택지 면적(sqft_lot), 층 수(floors), 우편번호(zipcode)를 더 주면서 학습을 시킵니다.**
# 
# > **그랬더니 조금 더 올라갔네요.**

# In[ ]:


features = ['sqft_living','bedrooms','bathrooms','sqft_lot','floors','zipcode']


# In[ ]:


gildong = LinearRegression()
gildong.fit(train_df_part1[features], train_df_part1['price'])
score = gildong.score(train_df_part2[features], train_df_part2['price'])
print(format(score,'.3f'))


# > **몇 가지 정보를 더 줘서 학습을 시켜봅니다. 조금 더 좋아졌습니다.**

# In[ ]:


features = ['sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'view',
             'grade','yr_built','zipcode']


# In[ ]:


gildong = LinearRegression()
gildong.fit(train_df_part1[features], train_df_part1['price'])
score = gildong.score(train_df_part2[features], train_df_part2['price'])
print(format(score,'.3f'))


# ### 3.4 K-NN 알고리즘 이용하여 예측하기

# > **이제까지는 선형회귀 모델을 이용하여 주택 가격을 예측해 보았습니다. 이제부터는 몇 가지 다른 모델을 이용하여 시험해 보겠습니다. 먼저 K-근접 이웃 방법입니다.**

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

babo = KNeighborsRegressor(n_neighbors=10)
babo.fit(train_df_part1[features], train_df_part1['price'])
score = babo.score(train_df_part2[features], train_df_part2['price'])
print(format(score,'.3f'))


# ### 3.5 결정트리 알고리즘 이용하여 예측하기

# > **결정트리 모델입니다. 점수와 예측한 주택 가격을 출력해 보았습니다.**

# In[ ]:


youngja = DecisionTreeRegressor(random_state = 0)
youngja.fit(train_df_part1[features], train_df_part1['price'])
score = youngja.score(train_df_part2[features], train_df_part2['price'])
print(format(score,'.3f'))

predicted = youngja.predict(train_df_part2[features])
print(predicted, '\n', predicted.shape)


# ### 3.6 랜덤 포레스트 알고리즘 이용하여 예측하기

# > **마지막으로 랜덤 포레스트 모델을 이용해봅니다. 가장 좋은 결과를 내고 있습니다.** 

# In[ ]:


cheolsu = RandomForestRegressor(n_estimators=28,random_state=0)
cheolsu.fit(train_df_part1[features], train_df_part1['price'])
score = cheolsu.score(train_df_part2[features], train_df_part2['price'])
print(format(score,'.3f'))

predicted = youngja.predict(train_df_part2[features])
print(predicted, '\n', predicted.shape)


# ## 4. 답안지 제출하기

# > **머신러닝 모형을 만든 후 데이터로 학습시키고 나머지 데이터로 어느 정도 잘 알아맞추는지를 확인해 보았습니다.**
# 
# > **이제 풀 문제를 읽어와서 가격을 예측해 봅니다.**

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
print(test_df.shape)
predicted = cheolsu.predict(test_df[features])
print(predicted.shape)
print(predicted)


# > **모델이 예측한 값 6,468개를 옮겨적을 답안지를 읽어옵니다. 그리고 예측한 값을 옮겨적은 후 my_submission.csv라는 파일로 저장합니다. 이 파일을 Kaggle.com에서 제출하면 모든 것이 끝납니다.**

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['price'] = predicted 
sub.to_csv('my_submission.csv', index=False)
print('Submission file created!')


# > **성능을 높이기 위하여 어떤 것을 더 해야 하는지 생각해봅시다.**

# ## 5. 성능을 높이기 위한 추가 방법들
# 

# > * **기존의 정보를 이용하여 새로운 컬럼을 생성할 수 있을까?**
# > * **값이 너무 튀는 것들(outliers)은 버리는 것이 어떨까? **
# > * **값을 구간 값으로 바꾸는 것을 어떨까?**
# > * **어떤 컬럼의 값은 대체로 작고, 어떤 컬럼의 값은 아주 큰데, 이를 일정한 크기로 바꾸면 어떨까(정규화)? **
# > * **원핫 인코딩을 사용해야 하지 않나?**
# > * **앙상블 모델을 사용하면 어떨까?**
# > * **딥 뉴럴 네트워크, 딥러닝을 이용하면 어떨까?**
# 
# 
# 
# **끝**
# 