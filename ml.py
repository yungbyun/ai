import pandas as pd # 데이터 처리 모듈
import matplotlib.pyplot as plt # 데이터 시각화 모듈 
import seaborn as sns # 데이터 시각화 모듈
from sklearn.model_selection import train_test_split # 데이터 분할 모듈
              

def disable_warning():
    import warnings
    warnings.filterwarnings('ignore')

    
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

        
# show_cols(data_f)    
def show_cols(df):
    for col in df.columns: 
        print(col)

        
def hist(df):
    data_f.hist(edgecolor='black', linewidth=1.2)
    fig = plt.gcf()
    fig.set_size_inches(12,10)
    plt.show()

    
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
    fig.set_size_inches(10, 6)
    plt.show()

    
# boxplot('owner', 'height')
def boxplot(df, a, b):
    f, sub = plt.subplots(1, 1,figsize=(8,6))
    sns.boxplot(x=df[a],y=df[b], ax=sub)
    sub.set(xlabel=a, ylabel=b);
    
    
# violin_plot(data_f, 'owner', 'leaf_length')
def violin_plot(df, _x, _y):
    plt.figure(figsize=(8,6))
    plt.subplot(1,1,1)
    sns.violinplot(x=_x,y=_y,data=df)

    
# plot_3d('day', 'leaf_length', 'leaf_width')
def plot_3d(a, b, c):
    from mpl_toolkits.mplot3d import Axes3D

    fig=plt.figure(figsize=(12,8))

    ax=fig.add_subplot(1,1,1, projection="3d")
    ax.scatter(data_f[a],data_f[b],data_f[c],c="blue",alpha=.5)
    ax.set(xlabel=a,ylabel=b,zlabel=c)

    
def label2value(col):
    #Labeling the object datas
    from sklearn.preprocessing import LabelEncoder
    labelencoder=LabelEncoder()
    for dataset in [data_f]:
        dataset.loc[:,col]=labelencoder.fit_transform(dataset.loc[:,col].values)
      
    
# heatmap(df, ['day', 'height', 'leaf_width', 'leaf_length', 'owner'])
def heatmap(dataf, cols):
    plt.figure(figsize=(12,8))
    sns.heatmap(data_f[cols].corr(),annot=True)

    
def split_4_parts(df, li, dap_col):
    # 학습용(문제, 정답), 테스트용(문제, 정답)으로 데이터 나누기
    train, test = train_test_split(df, train_size = 0.8)

    # 학습용 문제와 정답
    a = train[li]
    b = train[dap_col]

    # 시험 문제와 정답
    c = test[li]
    d = test[dap_col]

    return a, b, c, d


# 다양한 예측 알고리즘 패키지를 임포트함.              
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def run_4_regressors(a, b, c, d):
    # [1] 결정트리 예측기 머신러닝 알고리즘 학습
    gildong = DecisionTreeRegressor(random_state = 0)
    gildong.fit(train_X, train_y) #학습용 문제, 학습용 정답  
    score1 = gildong.score(test_X, test_y) # 시험 문제, 시험 정답
    # score의 의미: 정확하게 예측하면 1, 평균으로 예측하면 0, 더 못 예측하면 음수  

    # [2] 랜덤 포레스트 예측기 머신러닝 알고리즘
    youngja = RandomForestRegressor(n_estimators=28,random_state=0)
    youngja.fit(train_X, train_y)
    score2 = youngja.score(test_X, test_y)

    # [3] K근접이웃 예측기 머신러닝 알고리즘
    cheolsu = KNeighborsRegressor(n_neighbors=2)
    cheolsu.fit(train_X, train_y)
    score3 = cheolsu.score(test_X, test_y)

    # [4] 선형회귀 머신러닝 알고리즘
    minsu = LinearRegression()
    minsu.fit(train_X, train_y)
    score4 = minsu.score(test_X, test_y)

    plt.plot(['DT','RF','K-NN','LR'], [score1, score2, score3, score4])
    print('스코어: {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}'.format(score1, score2, score3, score4))


# 다양한 분류 알고리즘 패키지를 임포트함.
from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

def run_4_classifiers(train_X, train_y, test_X, test_y):
    gildong = svm.SVC() # 애기 
    gildong.fit(train_X,train_y) # 가르친 후
    prediction = gildong.predict(test_X) # 테스트
    rate1 = metrics.accuracy_score(prediction,test_y) * 100

    cheolsu = LogisticRegression()
    cheolsu.fit(train_X,train_y)
    prediction = cheolsu.predict(test_X)
    rate2 = metrics.accuracy_score(prediction,test_y) * 100

    youngja = DecisionTreeClassifier()
    youngja.fit(train_X,train_y)
    prediction = youngja.predict(test_X) # 테스트
    rate3 = metrics.accuracy_score(prediction,test_y) * 100

    minsu = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
    minsu.fit(train_X,train_y)
    prediction = minsu.predict(test_X) # 테스트
    rate4 = metrics.accuracy_score(prediction,test_y) * 100

    plt.plot(['SVM','Logistic','DTree','K-NN'], [rate1, rate2, rate3, rate4])
    print('인식률: {0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}'.format(rate1, rate2, rate3, rate4))
    
