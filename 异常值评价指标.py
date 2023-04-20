from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

y_true = [1, 0, 1, 0, 1, 0]
y_pred = [1, 1, 0, 0, 1, 0]

# 计算混淆矩阵
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# 计算精确率、召回率和F1分数
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

print("tn：", tn)
print("fp：", fp)
print("fn：", fn)
print("tp：", tp)


print("精确率：", precision)
print("召回率：", recall)
print("F1分数：", f1_score)