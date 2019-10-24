import numpy as np # 수학 연산 수행을 위한 모듈
import pandas as pd # 데이터 처리를 위한 모듈
import seaborn as sns # 데이터 시각화 모듈
import matplotlib.pyplot as plt # 데이터 시각화 모듈

# 다양한 분류 알고리즘 패키지를 임포트함.
from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘
#from sklearn.cross_validation import train_test_split # 데이타 쪼개주는 모듈
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

def show(df):
    print(df.info())
    print(df.head(5))

def drop(df, col):
    # 불필요한 열(ID) 제거
    df.drop(col, axis=1, inplace=True)  # ID라는 컬럼(열)을 삭제하라는 의미

    # 불필요한 Id 컬럼 삭제
    # axis=1 : 컬럼을 의미
    # inplace=True : 삭제한 후 데이터 프레임에 반영하라는 의미

def read(fn):
    imsi = pd.read_csv(fn)
    return imsi

# sample: myplot(gildong, "Height", "Weight", "Sex")
def myplot(df, a, b, c):
    fig = df[df[c]==0].plot(kind='scatter',x=a,y=b,color='orange', label='Female')
    df[df[c]==1].plot(kind='scatter',x=a,y=b,color='blue', label='Male', ax=fig)
    fig.set_xlabel(a)
    fig.set_ylabel(b)
    fig.set_title(a + " vs. " + b)
    fig=plt.gcf()
    fig.set_size_inches(10,6)
    plt.show()

def myhist(a):
    a.hist(edgecolor='black', linewidth=1.2)
    fig = plt.gcf()
    fig.set_size_inches(12,10)
    plt.show()

#a에 따라서 b의 분포를 보여줌.
#예) 성별에 따라 키의 분포를 보여줌.
def myviolinplot(df, a, b):
    plt.figure(figsize=(5,4))
    plt.subplot(1,1,1)
    sns.violinplot(x=a,y=b,data=df)
    plt.show()

def display_heatmap(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r')
    plt.show()

def data_nanum(df):
    a, b = train_test_split(df, test_size=0.3)
    # train=70% and test=30%
    return a, b


