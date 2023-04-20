import random
import numpy
import math
import time
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
class solution:
 def __init__(self):
     self.best = 0
     self.bestIndividual = []    # Rabbit_Location,C ,gamma
     self.convergence = []   # fitness
     self.optimizer = ""
     self.objfname = ""
     self.startTime = 0
     self.endTime = 0
     self.executionTime = 0
     self.lb = 0
     self.ub = 0
     self.dim = 0
     self.popnum = 0
     self.maxiers = 0

def HHO(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # dim=2
    # SearchAgents_no=50
    # lb=-100
    # ub=100
    # Max_iter=500

    # dim：搜索空间的维数，即优化问题中决策变量的数量。
    #
    # SearchAgents_no：算法中使用的hawks（搜索代理）数量。此参数决定总体的大小。
    #
    # lb：表示每个决策变量的搜索空间下界的列表或标量。如果lb是标量，则假设所有决策变量都具有相同的下界。如果lb是一个列表，那么它的长度必须等于dim，并且它必须包含每个决策变量的下限。
    #
    # ub：表示每个决策变量的搜索空间上界的列表或标量。如果ub是标量，则假设所有决策变量都具有相同的上界。如果ub是一个列表，那么它的长度必须等于dim，并且它必须包含每个决策变量的上限。
    #
    # 最大迭代次数：算法将运行的最大迭代次数。这个参数决定了算法的停止标准。






    # 初始化兔子的位置和能量
    Rabbit_Location = numpy.zeros(dim)
    Rabbit_Energy = float("inf")  # 对于最大化问题，将其更改为-inf

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = numpy.asarray(lb)
    ub = numpy.asarray(ub)

    # 初始化哈里斯鹰的位置
    X = numpy.asarray(
        [x * (ub - lb) + lb for x in numpy.random.uniform(0, 1, (SearchAgents_no, dim))]
    )
    print(X.shape)

    # 初始化收敛
    convergence_curve = numpy.zeros(Max_iter)#收敛曲线

    ############################
    s = solution()

    print('HHO is now tackling  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    # 主回路
    while t < Max_iter:
        for i in range(0, SearchAgents_no):#SearchAgents_no 鹰个数

            # 检查边界

            X[i, :] = numpy.clip(X[i, :], lb, ub)#clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min


            # fitness of locations地点适宜性
            fitness = objf(X[i, :])

            # Update the location of Rabbit更新Rabbit的位置
            if fitness < Rabbit_Energy:  # Change this to > for maximization problem对于最大化问题，将其更改为>
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()


        E1 = 2 * (1 - (t / Max_iter))# **0.5 #0-2 factor to show the decreaing energy of rabbit显示兔子能量下降的因素

        # Update the location of Harris' hawks更新哈里斯鹰的位置
        for i in range(0, SearchAgents_no):

            E0 = 2 * random.random() - 1  # -1<E0<1
            Escaping_Energy = E1 * (
                E0
            )  # escaping energy of rabbit Eq. (3) in the paper兔子的逃逸能量。（3）在论文中

            # -------- Exploration phase Eq. (1) in paper -------------------勘探阶段方程（1）在论文中

            if abs(Escaping_Energy) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:哈里斯鹰根据两种策略随机栖息：
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index, :]
                if q >= 0.5:
                    # perch based on other family members以其他家庭成员为基础
                    X[i, :] = X_rand - random.random() * abs(
                        X_rand - 2 * random.random() * X[i, :]
                    )

                elif q < 0.5:
                    # perch on a random tall tree (random site inside group's home range)栖息在一棵随机的高树上（在小组的家范围内的随机地点）
                    X[i, :] = (Rabbit_Location - X.mean(0)) - random.random() * (
                        (ub - lb) * random.random() + lb
                    )

            # -------- Exploitation phase -------------------开采阶段
            elif abs(Escaping_Energy) < 1:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                # 使用关于兔子行为的4种策略攻击兔子

                # phase 1: ----- surprise pounce (seven kills) ----------
                # 第一阶段：突然袭击
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks
                # 突袭（七杀）：不同的鹰进行多次短距离快速俯冲

                r = random.random()  # probablity of each event每个事件的概率

                if (
                    r >= 0.5 and abs(Escaping_Energy) < 0.5
                ):  # Hard besiege Eq. (6) in paper论文中的硬围攻方程（6）
                    X[i, :] = (Rabbit_Location) - Escaping_Energy * abs(
                        Rabbit_Location - X[i, :]
                    )

                if (
                    r >= 0.5 and abs(Escaping_Energy) >= 0.5
                ):  # Soft besiege Eq. (4) in paper论文中的软围攻方程（4）
                    Jump_strength = 2 * (
                        1 - random.random()
                    )  # random jump strength of the rabbit兔子的随机跳跃强度
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :]
                    )

                # phase 2: --------performing team rapid dives (leapfrog movements)----------
                # 第二阶段：--------进行团队快速跳水（蛙跳动作）----------

                if (
                    r < 0.5 and abs(Escaping_Energy) >= 0.5
                ):  # Soft besiege Eq. (10) in paper论文中的软围攻方程（10）
                    # rabbit try to escape by many zigzag deceptive motions兔子试图通过许多曲折的欺骗动作逃跑
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :]
                    )
                    X1 = numpy.clip(X1, lb, ub)

                    if objf(X1) < fitness:  # improved move?改进的动作？
                        X[i, :] = X1.copy()
                    else:  # hawks perform levy-based short rapid dives around the rabbit鹰队围绕兔子进行基于征税的短距离快速跳水
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X[i, :])
                            + numpy.multiply(numpy.random.randn(dim), Levy(dim))
                        )
                        X2 = numpy.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i, :] = X2.copy()
                if (
                    r < 0.5 and abs(Escaping_Energy) < 0.5
                ):  # Hard besiege Eq. (11) in paper论文中的硬围攻方程（11）
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X.mean(0)
                    )
                    X1 = numpy.clip(X1, lb, ub)

                    if objf(X1) < fitness:  # improved move?
                        X[i, :] = X1.copy()
                    else:  # Perform levy-based short rapid dives around the rabbit围绕兔子进行基于征税的短距离快速潜水
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X.mean(0))
                            + numpy.multiply(numpy.random.randn(dim), Levy(dim))
                        )
                        X2 = numpy.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i, :] = X2.copy()

        convergence_curve[t] = Rabbit_Energy

        if t % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(t)
                    + " the best fitness is "
                    + str(Rabbit_Energy)
                    + ",bestIndividual C and gamma"
                    + str(Rabbit_Location)
                ]
            )
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "HHO"
    s.objfname = objf.__name__
    s.best = Rabbit_Energy
    s.bestIndividual = Rabbit_Location

    return s


def Levy(dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from solution import solution
# from HHO import HHO

# Load the dataset


def normalize(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)
def dispart_data_train(data_patch):
    all_data_ = pd.read_excel(data_patch)
    data_1 = all_data_.iloc[0:1000]
    data_1 = data_1.apply(normalize, axis=0)
    data_train = data_1
    return data_train

def dispart_data_test(data_patch):
    all_data_ = pd.read_excel(data_patch)
    data_1 = all_data_.iloc[1000:1200]
    data_1 = data_1.apply(normalize, axis=0)
    data_test = data_1
    return data_test

def read(data):
    x = data[['Wind Speed (m/s)', 'LV ActivePower (kW)']]
    y = data[['label']]
    y = np.ravel(y)
    # print("aaaaaaaaaaaa",x)
    # print("bbbbbbbbbbb",y)

    return x, y

train_dataset = dispart_data_train('异常值4.xlsx')
test_dataset = dispart_data_test('异常值4.xlsx')
train_x, train_y = read(train_dataset)
test_x, test_y = read(test_dataset)

# X = np.loadtxt('data.csv', delimiter=',')
# y = np.loadtxt('labels.csv', delimiter=',')
#
# # Split the dataset into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function that evaluates the performance of SVM on the validation set
def objective_function(params):
    C, gamma = params
    clf = SVC(C=C, gamma=gamma, kernel='rbf')
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)

    # 计算F1得分作为评估指标
    f1_scores = f1_score(test_y, y_pred)

    precision = precision_score(test_y, y_pred)

    accuracy = accuracy_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred)
    auc = roc_auc_score(test_y, y_pred)
    print("AUC值：", auc)
    print("准确率：", accuracy)
    print("精确率：", precision)
    print("召回率：", recall)
    print("F1分数：", f1_scores)
    # return 1/np.exp(auc+precision+recall)
    return 1/(auc+precision+recall)


    # return -clf.score(test_x,test_y)  # maximize the negative of classification accuracy

# Define the search space for C and gamma
lb = [0.1, 0.0001]
ub = [100, 10]

# Set the parameters for HHO
dim = 2
SearchAgents_no = 20
Max_iter = 20

# Run the HHO algorithm to optimize C and gamma
s = HHO(objective_function, lb, ub, dim, SearchAgents_no, Max_iter)

# Print the best solution found
print('Best solution:')
print('C =', s.bestIndividual[0])
print('gamma =', s.bestIndividual[1])
print('Classification accuracy =', s.best)

# y_pred = clf.predict(test_x)
#
# # 计算F1得分作为评估指标
# f1_score = f1_score(test_y, y_pred)
#
# precision = precision_score(test_y, y_pred)
#
# accuracy = accuracy_score(test_y, y_pred)
# recall = recall_score(test_y, y_pred)

import matplotlib.pyplot as plt

# Plot the convergence curve of the HHO algorithm
plt.plot(range(Max_iter), s.convergence)
plt.title('Convergence curve')
plt.xlabel('Iteration')
plt.ylabel('Classification accuracy')
plt.show()