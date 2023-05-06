import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
#利用randint随机生成50个1~10之间的整数，用reshape将其由1行50列变成5列10行
df = DataFrame(np.random.randint(1, 10, 50).reshape(
    10, 5), index=[i for i in range(1, 11)], columns=['A', 'B', 'C', 'D', 'E'])
print(DataFrame(np.random.randint(1, 10, 50).reshape(
    10, 5), index=[i for i in range(1, 11)], columns=['A', 'B', 'C', 'D', 'E']))
df.plot()#默认为折线图
#绘制直方图
df.plot(kind='hist')
#绘制柱状图
df.plot(kind='bar')
#绘制散点图
df.plot.scatter(x=1,y=2)
#绘制填充线性图
df.plot(kind='area')
#绘制水平柱状
df.plot(kind='barh')
plt.show()
