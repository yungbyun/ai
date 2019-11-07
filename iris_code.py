#!/usr/bin/env python
# coding: utf-8

# # 머신러닝을 이용한 붓꽃(IRIS) 인식
# 

# > Source: https://www.kaggle.com/ash316/ml-from-scratch-with-iris
# 
# > 초보자를 위한 Machine Learning 튜토리얼 <br/>
# > 복사꽃 데이터로 머신러닝을 구현하는 방법 <br/>
# > 좀 더 고급 내용을 알고싶으면 아래 클릭  <br/>
# > https://www.kaggle.com/ash316/ml-from-scratch-part-2/notebook

# ## 1. 데이터 불러오기

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# 파일을 읽어서 2차원 테이블 형식의 데이터 프레임을 만듬.
iris = pd.read_csv("../input/Iris.csv") 


# In[ ]:


# 데이터에 비어있는(널) 값이 있는지 확인할 수 있음.
iris.info() 


# In[ ]:


# 데이터 프레임에 들어있는 10개 데이터 보여줌.
iris.head(10) 


# ## 2. 데이터 전처리

# > 불필요한 열(ID) 제거

# In[ ]:


iris.drop('Id',axis=1,inplace=True) 
# 불필요한 Id 컬럼 삭제
# axis=1 : 컬럼을 의미
# inplace=1 : 삭제한 후 데이터 프레임에 반영


# In[ ]:


iris.head(5)


# ## 3. 데이터 시각화와 분석

# > 붓꽃의 종류에는 3가지(세토사 Setosa, 버시컬러 Versicolor, 버지니카 Vriginica)가 있음. <br/>
# > 각 붓꽃에 대하여 꽃받침 길이와 꽃받침 너비를 점으로 찍어본다. 이때 붓꽃 종류에 따라 색을 달리하여 찍어본다. <br/> 

# In[ ]:


fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor', ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Sepal Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# > 이번에는 붓꽃의 꽃잎의 길이와 너비의 관계를 출력해보자. 

# In[ ]:


fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor', ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal Length VS Petal Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# > 꽃받침과 꽃잎 중 어느 것을 이용할 경우 꽃을 더 쉽게 분류할 수 있을까?

# > 이제, 꽃잎 너비, 꽃잎 길이, 곷받침 너비, 꽃받침 길이 값이 어떻게 분포하는지 살펴보자. 

# In[ ]:


iris.hist(edgecolor='black', linewidth=1.2)
fig = plt.gcf()
fig.set_size_inches(12,10)
plt.show()


# > 각각의 종류에 따라 꽃잎 길이, 꽃잎 너비, 꽃받침 길이, 꽃받침 너비 값이 어떻게 분포하는지 바이올린 차트로 표시해보자. 

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris)


# > 얇은 것은 다양하게 분포가 되어있다는 것을 의미하고<br/>
# > 넓은 것은 특정 특정 범위에 몰려있다는 것을 의미

# ## 4. 꽃의 종류 알아맞추기 (분류)

# > **특징(features) 혹은 속성(attributes)** : 어떤 사물의 특징. 머신러닝 알고리즘 입력으로 주어지는 값들 <br/>
# > **정답 혹은 목표값(target value)** : 맞춰야 하는 정답. 이 경우는 꽃의 종류 (3가지 중 하나)

# In[ ]:


# 다양한 분류 알고리즘 패키지를 임포트함.
from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘
#from sklearn.cross_validation import train_test_split # 데이타 쪼개주는 모듈 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm


# In[ ]:


iris.shape # 데이터 프레임 모양(shape)


# > 사람의 키와 발의 크기는 어떤 관계가 있을까? 예를 들어, 키가 클수록 발이 클까? 매우 그렇다면 상관관계가 큰 것을 의미한다. <br/>
# > 그렇다면 꽃잎의 길이는 꽃잎의 너비, 꽃받침대의 길이, 곷받침대의 너비와 어떤 관계가 있을까? <br/>

# In[ ]:


iris.head(5)


# In[ ]:


plt.figure(figsize=(14,8)) 
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')
plt.show()


# > 꽃받침대 길이(SepalLengthCm)와 꽃받침대 너비(SepalWidthCm)는 -0.11로서 상관관계가 매우 적음. <br/>
# > 대신 꽃잎의 길이와 너비와 밀접한 상관관계를 가짐. <br/>
# > 꽃잎 길이(PetalLengthCm)와 꽃잎 너비(PetalWidthCm)는 0.96으로 상관관계가 매우 높음.

# > 머신러닝 구현 절차는 다음과 같음. <br/>
# 
# >  1. 데이터 쪼개기 (학습용, 테스트용) <br/>
# >  2. 알고리즘 선택 (아기 객체 만들기) <br/>
# >  3. 아기 **.fit()** 함수 호출하여 학습 시키키 <br/>
# >  4. 아기 **.predict()** 함수를 호출하여 테스트하기 <br/>
# >  5. 아기가 예측한 값과 실제 정답 비교하여 인식률 계산하기

# ### 4.1 데이터 쪼개기 (학습 데이터, 테스트 데이터)

# In[ ]:


train, test = train_test_split(iris, test_size = 0.3)# in this our main data is split into train and test

print(train.shape)
print(test.shape)


# In[ ]:


train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]# taking the training data features
train_y = train.Species# output of our training data

test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features
test_y =test.Species   #output value of test data


# > 학습용 입력 데이터 표시

# In[ ]:


train_X.head(10)


# > 학습용 정답 데이터 표시

# In[ ]:


train_y.head(10)


# > 테스트 문제 표시

# In[ ]:


test_X.head(10)


# > 테스트 문제에 대한 정답

# In[ ]:


train_y.head(10)  ##output of the training data


# ### 4.2 써포트 벡터 머신(SVM) 알고리즘 이용하여 알아맞추기

# In[ ]:


import warnings  
warnings.filterwarnings('ignore')


# In[ ]:


# 머신러닝 알고리즘 객체 만들기

model = svm.SVC() # 애기 
model.fit(train_X,train_y) # 가르친 후
prediction=model.predict(test_X) # 테스트

print('정확도:',metrics.accuracy_score(prediction,test_y) * 100)


# ### 4.3 논리 회귀(Logistic Regression) 알고리즘 이용하여 알아맞추기

# In[ ]:


model = LogisticRegression()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('인식률:',metrics.accuracy_score(prediction,test_y))


# ### 4.4 결정 트리(Decision Tree) 알고리즘 이용하여 알아맞추기

# In[ ]:


model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))


# ### 4.5 근접 이웃(K-Nearest Neighbours) 알고리즘 이용하여 알아맞추기 

# In[ ]:


model=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))


# > 가장 가까운 이웃의 수 K를 1에서 12까지 조절하면서 인식 실험해봄. 

# In[ ]:


print(list(range(1,13)))

a = [0]
for i in list(range(1,13)):
    model = KNeighborsClassifier(n_neighbors = i) 
    model.fit(train_X,train_y)
    prediction = model.predict(test_X)
    score = metrics.accuracy_score(prediction,test_y)*100
    a.append(score)
    print(i, score)

print(a)

plt.figure(figsize=(12, 7))
plt.plot(a)
plt.ylabel('Rate')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
plt.show()


# ### 4.6 꽃잎(길이/너비), 혹은 꽃받침(길이/너비) 정보만 이용하여 분류해보기

# > 학습 데이터 만들기

# In[ ]:


petal=iris[['PetalLengthCm','PetalWidthCm','Species']]
sepal=iris[['SepalLengthCm','SepalWidthCm','Species']]


# In[ ]:


train_p,test_p=train_test_split(petal,test_size=0.3,random_state=0)  #petals
train_x_p=train_p[['PetalWidthCm','PetalLengthCm']]
train_y_p=train_p.Species
test_x_p=test_p[['PetalWidthCm','PetalLengthCm']]
test_y_p=test_p.Species


train_s,test_s=train_test_split(sepal,test_size=0.3,random_state=0)  #Sepal
train_x_s=train_s[['SepalWidthCm','SepalLengthCm']]
train_y_s=train_s.Species
test_x_s=test_s[['SepalWidthCm','SepalLengthCm']]
test_y_s=test_s.Species


# > SVM 이용하여 인식해보면

# In[ ]:


model=svm.SVC()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('꽃잎 길이/너비 정보만 이용할 경우 인식률:',metrics.accuracy_score(prediction,test_y_p)*100)

model=svm.SVC()
model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('꽃받침 길이/너비 정보만 이용할 경우 인식률:',metrics.accuracy_score(prediction,test_y_s)*100)


# > Logistic Regression을 이용하여 인식해보면

# In[ ]:


model = LogisticRegression()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('꽃잎 길이/너비 정보만 이용할 경우 인식률:',metrics.accuracy_score(prediction,test_y_p)*100)

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('꽃받침 길이/너비 정보만 이용할 경우 인식률:',metrics.accuracy_score(prediction,test_y_s)*100)


# > Decision Tree를 이용하여 실험해보면

# In[ ]:


model=DecisionTreeClassifier()
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('꽃잎 길이/너비 정보만 이용할 경우 인식률:',metrics.accuracy_score(prediction,test_y_p)*100)

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('꽃받침 길이/너비 정보만 이용할 경우 인식률:',metrics.accuracy_score(prediction,test_y_s)*100)


# > K-Nearest Neighbours 알고리즘을 이용하여 테스트해보면

# In[ ]:


model=KNeighborsClassifier(n_neighbors=3) 
model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('꽃잎 길이/너비 정보만 이용할 경우 인식률:',metrics.accuracy_score(prediction,test_y_p)*100)

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('꽃받침 길이/너비 정보만 이용할 경우 인식률:',metrics.accuracy_score(prediction,test_y_s)*100)


# ## 5. 결론
# > - 꽃잎과 꽃받침 정보를 이용하여 꽃의 종류를 알아 맞출 수 있는지 구현해 보았음.  
# > - 모든 정보를 이용하여 SVM 알고리즘으로 인식할 경우, 혹은 K의 수가 10, 11, 12일 때 100%로 인식률이 가장 높음. 
# > - 꽃잎(길이/너비), 혹은 꽃받침(길이/너비) 정보 각각만을 이용할 경우 100% 인식률을 얻을 수 없었음. 

# In[ ]:




