#!/usr/bin/env python
# coding: utf-8

# ## 키, 몸무게, 발 크기로 성별 알아맞추기
# 
# > 이 프로그램은 아래 코드를 참고하여 작성하였습니다.. <br/>
# > Source: https://www.kaggle.com/ash316/ml-from-scratch-with-iris <br/>
# ****
# > 위 샘플 코드는 초보자를 위한 Machine Learning 튜토리얼로서 <br/>
# > 복사꽃 데이터로 머신러닝을 이해하기 위한 코드입니다.  <br/>
# > 좀 더 고급의 내용을 알고싶으면 아래를 참고하기 바랍니다. <br/>
# adsfka;sdfkjasldfaskjdfasdfk;ljaskldfkjalsdf
# > https://www.kaggle.com/ash316/ml-from-scratch-part-2/notebook

# ## 1. 데이터 불러오기

# In[1]:


import numpy as np # 수학 연산 수행을 위한 모듈
import pandas as pd # 데이터 처리를 위한 모듈
import seaborn as sns # 데이터 시각화 모듈
import matplotlib.pyplot as plt # 데이터 시각화 모듈 

# 어떤 파일이 있는지 표시하기
from subprocess import check_output

print(check_output(["ls", "../input/my-xxx-data"]).decode("utf8"))


# In[2]:


# CSV 파일 읽어오기
gildong = pd.read_csv("../input/my-xxx-data/dataset_new.csv")


# In[3]:


print(gildong) # 데이터 프레임에 들어있는 10개 데이터 보여줌.


# In[4]:


gildong.info()


# In[5]:


print(gildong)


# In[6]:


gildong.head(5)


# ## 2. 데이터 전처리

# In[7]:


# 불필요한 열(ID) 제거
gildong.drop('Id',axis=1,inplace=True) # ID라는 컬럼(열)을 삭제하라는 의미

# 불필요한 Id 컬럼 삭제
# axis=1 : 컬럼을 의미
# inplace=True : 삭제한 후 데이터 프레임에 반영하라는 의미


# In[8]:


gildong.head(10)


# > 남자, 여자의 키와 몸무게를 표시해보자. <br/>
# > 여자의 키(Height)와 몸무게(Weight)를 오렌지 색으로 표시(plot)하라. <br/>
# > 남자의 키와 몸부게는 파란색으로 표시하라. 

# In[9]:


def myplot(a, b, c):
    fig =     gildong[gildong[c]==0].plot(kind='scatter',x=a,y=b,color='orange', label='Female')
    gildong[gildong[c]==1].plot(kind='scatter',x=a,y=b,color='blue', label='Male', ax=fig)
    fig.set_xlabel(a)
    fig.set_ylabel(b)
    fig.set_title(a + " vs. " + b)
    fig=plt.gcf()
    fig.set_size_inches(10,6)
    plt.show()


myplot("FeetSize", "Height", "Sex")


# > 이번에는 남녀에 따른 키와 발 크기를 표시하면 어떨까? <br/>
# > 여자의 키(Height)와 발크기(FeetSize)를 오렌지 색으로 표시(plot)하라. <br/>
# > 남자의 키와 발크기는 파란색으로 표시하라. 

# In[10]:


myplot("Height", "FeetSize", "Sex")


# > 데이터에는 키/몸무게/발크기/학년/성별 특징이 있다. <br/>
# > 각 특징별로 어떤 분포를 보이는지도 표시할 수 있다.

# In[ ]:





# In[11]:


def myhist(a):
    a.hist(edgecolor='black', linewidth=1.2)
    fig = plt.gcf()
    fig.set_size_inches(12,10)
    plt.show()

myhist(gildong)


# ## 설명
# > 어떻게 분포하는지도 알 수 있다. 바이올린 모양으로 표시할 수 있는데 violinplot 함수를 이용한다. <br/>
# > 성별에 따라 꽃받침 너비, 꽃받침 길이, 꽃잎 너비, 꽃잎 길이 등이 어떻게 분포하는지 알 수 있다. <br/>
# > 하하하하

# In[12]:


def myviolinplot(df, a, b):
    plt.figure(figsize=(5,4))
    plt.subplot(1,1,1)
    sns.violinplot(x=a,y=b,data=df)

myviolinplot(gildong, 'Sex', 'Height')
myviolinplot(gildong, 'Sex', 'Weight')
myviolinplot(gildong, 'Sex', 'FeetSize')
myviolinplot(gildong, 'Sex', 'Year')


# > 얇은 것은 다양하게 분포가 되어있다는 것을 의미하고<br/>
# > 넓은 것은 특정 특정 범위에 몰려있다는 것을 의미

# ## 4. 남자인지 여자인지 알아맞추기 (분류)

# > **특징(features), 속성(attributes)**--> 머신러닝 알고리즘 입력으로 주어지는 값들 <br/>
# > **정답, 목표값(target value)**, 맞춰야 하는 정답. 이 경우는 꽃의 유형

# In[13]:


# 다양한 분류 알고리즘 패키지를 임포트함.
from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘
#from sklearn.cross_validation import train_test_split # 데이타 쪼개주는 모듈 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm


# In[14]:


print(gildong.shape) # 데이터 모양(shape)을 표시
print(gildong.head(5))


# > 남자 여자를 구별하는데 어떤 정보가 중요할까? <br/>
# > 남자와 여자의 발 크기는 다를까? <br/>
# > 남자와 여자의 키는 어떨까? <br/>
# > 남자와 여자의 몸무게는? <br/>
# > 남자와 여자의 학년은? <br/>
# > 상관 관계 <br/>
# 
# > 학습을 시키려면 정보의 상관 관계 정보가 중요함. <br/>
# > 상관 관계 값이 크면 학습을 잘할 것이고 <br/>
# > 상관 관계 값이 작으면 학습을 잘 못할 것이다. <br/>
# > 상관 관계 정보를 이용하여 어떤 내용으로 학습을 시키는지가 중요함.

# In[15]:


def display_heatmap(df):
    plt.figure(figsize=(10,6)) 
    sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r') 
    plt.show()

display_heatmap(gildong)


# > 위의 결과를 보고 설명해보자. 

# > 머신러닝 구현 절차는 다음과 같다. 
# 
# >  1. 데이터 쪼개기 (학습용, 테스트용) <br/>
# >  2. 알고리즘 선택 (학습시킬 아기 객체 만들기) <br/>
# >  3. 아기의 **.fit()** 함수 호출하여 학습 시키키 <br/>
# >  4. 아기의 **.predict()** 함수를 호출하여 테스트하기 <br/>
# >  5. 아기가 예측한 값과 실제 정답 비교하여 인식률 계산하기

# ### 4.1 데이터 쪼개기 (학습 데이터, 테스트 데이터)

# In[16]:


train, test = train_test_split(gildong, test_size = 0.3)
# train=70% and test=30%
print(train.shape)
print(test.shape)


# > 학습 데이터(70%)에서 키와 발크기만 따로 뽑아 보관함. 이는 학습시 입력으로 사용할 것임. <br/>
# > 정답도 뽑아 보관함. 

# In[17]:


print(train)


# In[18]:


train_X = train[['Height','FeetSize']] # 키와 발크기만 선택
train_y = train.Sex # 정답 선택


# In[19]:


print(train_X)
print(train_y)


# > 테스트 데이터(30%)에서도 키와 발크기만 따로 뽑아 보관함. 이는 테스트시 입력으로 사용할 것임. <br/>
# > 정답도 뽑아 보관함. 

# In[20]:


test_X = test[['Height','FeetSize']] # taking test data features
test_y = test.Sex   #output value of test data


# In[21]:


print(test_X)
print(test_y)


# In[22]:


train_X.head(2)


# In[23]:


test_X.head(2)


# In[24]:


train_y.head()  ## 테스트 데이터 정답 표시


# ### 4.2 써포트 벡터 머신(SVM) 알고리즘 이용하여 알아맞추기

# In[25]:


import warnings  
warnings.filterwarnings('ignore')


# In[26]:


baby1 = svm.SVC() # 애기 
baby1.fit(train_X,train_y) # 가르친 후
prediction = baby1.predict(test_X) # 테스트

print('인식률:',metrics.accuracy_score(prediction,test_y) * 100) 


# ### 4.3 논리 회귀(Logistic Regression) 알고리즘 이용하여 알아맞추기

# In[27]:


baby2 = LogisticRegression()
baby2.fit(train_X,train_y)
prediction = baby2.predict(test_X)
print('인식률:', metrics.accuracy_score(prediction,test_y) * 100)


# ### 4.4 결정 트리(Decision Tree) 알고리즘 이용하여 알아맞추기

# In[28]:


baby3 = DecisionTreeClassifier()
baby3.fit(train_X,train_y)
prediction = baby3.predict(test_X)
print('인식률:',metrics.accuracy_score(prediction,test_y) * 100)


# ### 4.5 근접 이웃(K-Nearest Neighbours) 알고리즘 이용하여 알아맞추기 

# In[29]:


baby4 = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
baby4.fit(train_X,train_y)
prediction = baby4.predict(test_X)
print('인식률:',metrics.accuracy_score(prediction,test_y) * 100)


# > 특징을 더 이용하면 어떨까?

# In[30]:


train_X = train[['Height','Weight','FeetSize']] # 키와 발크기뿐만 아니라 몸무게도 추가
train_y = train.Sex # 정답 선택

test_X = test[['Height','Weight','FeetSize']] # taking test data features
test_y = test.Sex   #output value of test data


# In[31]:


baby5 = svm.SVC() # 애기 
baby5.fit(train_X,train_y) # 가르친 후
prediction = baby5.predict(test_X) # 테스트
print('인식률:',metrics.accuracy_score(prediction,test_y) * 100) 

baby6 = LogisticRegression()
baby6.fit(train_X,train_y)
prediction = baby6.predict(test_X)
print('인식률:', metrics.accuracy_score(prediction,test_y) * 100)

baby7 = DecisionTreeClassifier()
baby7.fit(train_X,train_y)
prediction = baby7.predict(test_X)
print('인식률:',metrics.accuracy_score(prediction,test_y) * 100)

baby8 = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
baby8.fit(train_X,train_y)
prediction = baby8.predict(test_X)
print('인식률:',metrics.accuracy_score(prediction,test_y) * 100)


# ## 5. 결론
# 
# > - 데이터를 분석하고, 상관관계를 이해할 수 있다. <br/>
# > - 4가지 머신러닝 알고리즘을 이용하여 학습을 시키고 테스트를 수행할 수 있다. <br/> 
# > - 특징을 2개만 이용할 때보다는 3개를 이용할 때가 성능은 조금 더 향상되었다. <br/>
