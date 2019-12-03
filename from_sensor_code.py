import numpy as np # 수학 연산 수행을 위한 모듈
import pandas as pd # 데이터 처리를 위한 모듈
import seaborn as sns # 데이터 시각화 모듈
import matplotlib.pyplot as plt # 데이터 시각화 모듈

# 다양한 분류 알고리즘 패키지를 임포트함.
from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

class MyClassifier:
    df = 0

    rate_svm = 0
    rate_l_l = 0
    rate_n_c = 0
    rate_d_t_c = 0

    # csv 파일을 로드함. 예)df = read("a.csv")
    def read(self, fn):
        self.df = pd.read_csv(fn)

    #
    def show(self):
        print(self.df.info())
        print(self.df.head(5))
        print(self.df.shape)

    def drop(self, col):
        # 불필요한 열(ID) 제거
        self.df.drop(col, axis=1, inplace=True)  # ID라는 컬럼(열)을 삭제하라는 의미

        # 불필요한 Id 컬럼 삭제
        # axis=1 : 컬럼을 의미
        # inplace=True : 삭제한 후 데이터 프레임에 반영하라는 의미

    # plot('Height', 'Weight', 'Sex')
    def plot(self, x_col, y_col, color_field):
        cl = self.df[color_field].unique()
        col = ['orange', 'blue', 'red', 'yellow', 'black', 'brown']

        fig = self.df[self.df[color_field] == cl[0]].plot(kind='scatter', x=x_col, y=y_col, color=col[0], label=cl[0])
        for i in range(len(cl) - 1):
            self.df[self.df[color_field] == cl[i + 1]].plot(kind='scatter', x=x_col, y=y_col, color=col[i + 1], label=cl[i + 1],
                                                  ax=fig)

        fig.set_xlabel(x_col)
        fig.set_ylabel(y_col)
        fig.set_title(x_col + " vs. " + y_col)
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        plt.show()

    # histogram()
    def histogram(self):
        self.df.hist(edgecolor='black', linewidth=1.2)
        fig = plt.gcf()
        fig.set_size_inches(12, 10)
        plt.show()

    # violinplot(df, 'Height', 'Weight')
    # a(성별, 학년)에 따라서 b(키, 몸무게)의 분포를 보여줌.
    # 예) 성별에 따라 키의 분포를 보여줌.
    def violinplot(self, a, b):
        plt.figure(figsize=(5, 4))
        plt.subplot(1, 1, 1)
        sns.violinplot(x=a, y=b, data=self.df)
        plt.show()

    # 예)히트맵으로 성별과 가장 상관관계가 높은 필드(발크기,
    # 몸무게, 키 등)를 알 수 있음.
    def heatmap(self):
        plt.figure(figsize=(14, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='cubehelix_r')
        plt.show()

    # 전체데이터를 학습용(70%), 테스트 용(30%)으로 나눔
    def data_nanum(self):
        a, b = train_test_split(self.df, test_size=0.3)
        # train=70% and test=30%
        return a, b

    # 데이터프레임에서 원하는 컬럼만을 뽑아냄.
    # a = p.extract(df, 'Sex')
    # a = p.extract(df, ['Sex'])
    def extract(self, field_list):
        t = self.df[field_list]
        return t

    def ignore_warning(self):
        import warnings
        warnings.filterwarnings('ignore')

    def run_svm(self, input_cols, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[input_cols]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[input_cols]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = svm.SVC()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        self.rate_svm = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', self.rate_svm)

    def run_logistic_regression(self, input_cols, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[input_cols]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[input_cols]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = LogisticRegression()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        self.rate_l_l = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', self.rate_l_l)

    def run_neighbor_classifier(self, input_cols, target, num):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[input_cols]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[input_cols]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = KNeighborsClassifier(n_neighbors=num)  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        self.rate_n_c = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', self.rate_n_c)

    def run_decision_tree_classifier(self, input_cols, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[input_cols]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[input_cols]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = DecisionTreeClassifier()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        self.rate_d_t_c = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', self.rate_d_t_c)

    def run_all(self, input_cols, target, neighbor_num):
        self.run_logistic_regression(input_cols, target)
        self.run_decision_tree_classifier(input_cols, target)
        self.run_svm(input_cols, target)
        self.run_neighbor_classifier(input_cols, target, neighbor_num)

    def draw_4_accuracy(self):
        plt.figure(figsize=(8, 5))
        plt.title(' ')
        plt.plot(['SVM', 'Logistic_L', 'Neighbor_C', 'Decision_T_C'],
                 [self.rate_svm, self.rate_l_l, self.rate_n_c, self.rate_d_t_c],
                 label='Accuracy')
        plt.legend()
        plt.xlabel('')
        plt.ylabel('Accuracy')
        plt.show()

