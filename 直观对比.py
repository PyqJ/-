import pandas as pd
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
# data = pd.read_excel('异常值4.xlsx')
data = pd.read_excel('真实值3.xlsx')
# wind_t = data.iloc[:, 0]
# b = 3
wind = data.iloc[:, 0]
# print(wind)
t_power = data.iloc[:, 2]
# t2_power =t_power * ((b + wind) / wind)**3

a_power = data.iloc[:, 3]
# a2_power =a_power * ((b + wind) / wind)**3
# print(a2_power)



# a2_power = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 示例数组
# a = 3  # a的值
# new_arr = a2_power.copy()  # 复制数组
# new_arr[:, 0] = new_arr[:, 0] * (new_arr[:, 1] * 0.1 * (a == 3))  # 满足条件时进行乘法运算
# # print(new_arr)  # 输出结果

label = data.iloc[:, 4]
# print(wind.shape)
mean = t_power.mean()
# print(mean)

# t1_power = [num + 0.1 * mean for num in t_power]
# t2_power = [num - 0.1 * mean for num in t_power]
# # print(t2_power)
#
# for i, item in enumerate(my_list):
a = abs(a_power - t_power)
# print(a)
y_label =[]
# for i, n in enumerate(a):
#     if n >= 0.1*mean or n<= 0.1*mean:
#         Y_label = 0
#         y_label.append(Y_label)
#     else:
#         Y_label = 1
#         y_label.append(Y_label)
# print(y_label)
# for i in (a):
#     print(i)

for i in (a):
    if i >= 0.8*mean:
        Y_label = 1
        y_label.append(Y_label)
    else:
        Y_label = 0
        y_label.append(Y_label)
print(y_label)


# 计算混淆矩阵
tn, fp, fn, tp = confusion_matrix(label, y_label).ravel()

# 计算精确率、召回率和F1分数
acc = (tp + tn)/(tp + fn + fp + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

print("tn：", tn)
print("fp：", fp)
print("fn：", fn)
print("tp：", tp)

print("准确率：", acc)
print("精确率：", precision)
print("召回率：", recall)
print("F1分数：", f1_score)




train_colors = ['red' if label == 1 else 'blue' for label in y_label]
# test_colors = ['red' if label == 1 else 'blue' for label in data_pred_test_labels]

from sklearn.metrics import roc_auc_score

plt.scatter(wind, a_power, c=y_label)
# plt.scatter(test_x_n[:, 0], test_x_n[:, 1], c=test_colors)

plt.show()
#
#
#
#
#
#
#
#
#
# def precision(output, target):
#     with torch.no_grad():
#         pred = torch.round(output)
#         return precision_score(target, pred)
#
# def recall(output, target):
#     with torch.no_grad():
#         pred = torch.round(output)
#         return recall_score(target, pred)
#
# def f1(output, target):
#     with torch.no_grad():
#         pred = torch.round(output)
#         return f1_score(target, pred)
#

