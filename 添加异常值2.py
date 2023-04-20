import matplotlib.pyplot as plt
import pandas as pd
import random
import copy
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel('真实值3.xlsx')
d = data[data.columns[0:3]]
# d = data[data.columns[1:2]]
power = data[data.columns[3:4]]
label = data[data.columns[4:5]]
print(power)
print(label)
power_t = copy.deepcopy(power)
label_t = copy.deepcopy(label)
# print(power.loc[5])
# print(len(power))
sampel = random.sample(range(len(power)), int(0.1*len(power)))
sampel = sorted(sampel)
print(sampel)
mean = power.mean().item()
# a=random.uniform(0.7,1)
outlier = []
# print("qq",type(mean))
for i in sampel:
    # print(label.loc[i])
    # print(type(power.loc[i].item()))
    if power.loc[i].item() >= 1.5*mean:
        # print( power.loc[i])
        power.loc[i] = power.loc[i].item() - 1.5*mean * random.uniform(0.8, 1)
        outlier.append(power.loc[i].item())
    else:
        power.loc[i] = power.loc[i].item() + 1.5*mean * random.uniform(0.8, 1)
        outlier.append(power.loc[i].item())


    # label.loc[i]= 0*label.loc[i].item()
    label.loc[i] = 1

I = list(range(len(power)))
l_I = sampel

plt.plot(I, power, color='r')
plt.plot(I, power_t)
plt.scatter(l_I, outlier, color='r')
plt.xlim(100, 1000)
plt.show()

plt.scatter(I[100:1150], label_t[100:1150])
plt.scatter(I[100:1150], label[100:1150])
plt.show()

print(type(power))
print(type(label))
d.colums = ['Wind Speed (m/s)', 'Wind Direction(°)', 'Theoretical_Power_Curve (KWh)']
d.insert(3, 'LV ActivePower (kW)', power)
d.insert(4, 'label', label)
d.to_excel('D:/Learn Python/异常值处理/Outlier/异常值5.xlsx', index=False)
