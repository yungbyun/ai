import pandas as pd  # 데이터 처리 모듈
import matplotlib.pyplot as plt  # 데이터 시각화 모듈
import seaborn as sns  # 데이터 시각화 모듈
from sklearn.model_selection import train_test_split  # 데이터 분할 모듈

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def show_files():
    import os
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def load_csv(fn):
    # CSV 파일 읽어오기
    imsi = pd.read_csv(fn)
    return imsi

def show_col(df):
    print(df.columns)

class Visualization:  # jini
    def plot(self, df, a, b, c):
        # 읽어온 데이터 표시하기
        cl = df[c].unique()

        col = ['orange', 'blue', 'red', 'yellow', 'black', 'brown', 'teal', 'darkviolet', 'mediumblue', 'dodgerblue', 'crimson', 'darkred', 'lightseagreen', 'darkgoldenrod', 'firebrick']

        fig = df[df[c] == cl[0]].plot(kind='scatter', x=a, y=b, color=col[0], label=cl[0])

        for i in range(len(cl) - 1):
            df[df[c] == cl[i + 1]].plot(kind='scatter', x=a, y=b, color=col[i + 1], label=cl[i + 1], ax=fig)

        fig.set_xlabel(a)
        fig.set_ylabel(b)
        fig.set_title(a + " vs. " + b)
        fig = plt.gcf()
        fig.set_size_inches(12, 7)
        plt.show()

    def violin_plot(self, df, a, b):
        plt.figure(figsize=(5, 4))
        plt.subplot(1, 1, 1)
        sns.violinplot(x=a, y=b, data=df)
        plt.show()

    def show_heatmap(self, df, a):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[a].corr(), annot=True)
        plt.show()

    def box_plot(self, df, a, b):
        f, sub = plt.subplots(1, 1, figsize=(7, 5))
        sns.boxplot(x=df[a], y=df[b], ax=sub)
        sub.set(xlabel=a, ylabel=b);
        plt.show()

    def show_3d(self, df, a, b, c):
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 8))

        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(df[a], df[b], df[c], c="blue", alpha=.5)
        ax.set(xlabel=a, ylabel=b, zlabel=c)
        plt.show()

class Base:
    train_X = None
    train_y = None
    test_X = None
    test_y = None

    gildong = None

    def split_4_parts(self, df, i, answer):
        # 학습용 문제, 학습용 정답, 테스트용 문제, 테스용 정답으로 데이터를 4개로 쪼개기, split_4_parts
        train, test = train_test_split(df, train_size=0.8)

        # 학습용 문제와 정답
        self.train_X = train[i]  # 문제
        self.train_y = train[answer]  # 정답

        # 테스트용 문제와 정답
        self.test_X = test[i]
        self.test_y = test[answer]

    def predict_score(self):
        # 테스트한(기말고사) 후 점수까지 계산하기, test_score
        score = self.gildong.score(self.test_X, self.test_y)  # 테스트 문제, 테스트 정답
        print('Score:', format(score, '.3f'))  # score의 의미: 정확하게 예측하면 1, 평균으로 예측하면 0, 더 못 예측하면 음수

    def show_test_data(self):
        # 테스트 문제지와 정답 출력해보기, show_test_data
        print(self.test_X)  # 입력: 10일 후에는
        print('-----')
        print(self.test_y)  # 정답: 35만큼 자란다.

    def predict(self, d):
        # 테스트 문제 전부 주고 테스트하기, test
        predicted = self.gildong.predict(d)
        print('Predicted:', predicted)


class MyModels(Base):  # DecisionTreeRegressor

    def learn_dtr(self):
        # 시험공부시키, learn
        self.gildong = DecisionTreeRegressor(random_state=0)
        self.gildong.fit(self.train_X, self.train_y)  # 학습용 문제, 학습용 정답

    def learn_knr(self):
        # 시험공부시키, learn
        self.gildong = KNeighborsRegressor(n_neighbors=2)
        self.gildong.fit(self.train_X, self.train_y)  # 학습용 문제, 학습용 정답

    def learn_lr(self):
        # 시험공부시키, learn
        self.gildong = LinearRegression()
        self.gildong.fit(self.train_X, self.train_y)  # 학습용 문제, 학습용 정답

    def learn_rfr(self):
        # 시험공부시키, learn
        self.gildong = RandomForestRegressor(n_estimators=28, random_state=0)
        self.gildong.fit(self.train_X, self.train_y)  # 학습용 문제, 학습용 정답
