import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import datasets
import random
import matplotlib.pyplot as plt

class Kmeans:


    '''
    ===初始化==
    data为数据集
    k为质心数量默认为3
    '''
    def __init__(self,data):
        self.data = self.pca(data)
        

    '''
    ====kmeans主代码====
    cluster表示质心的点
    belong_lst表示每条的所属关系 结构如[[],[],[]...]中共有k个嵌套列表,每个列表中表示对应质心包含的点
    '''
    def kmeans(self,k=3,draw=True):
        # =======基本步骤========
        #随机初始质心
        #循环下面步骤
        #计算距离并归类
        #重新归类
        #当质心不在发生变化停止循环
        #画图
        # ======================
        self.k = k
        self.cluster = self.random_cluster()
        # 返回最终质心和他对应的点(结构见下面注释)
        self.cluster,self.belong_lst = self.update()
        if draw:
            self.draw()



    '''
    ==计算距离==
    vector1,vector2是向量 np.array
    '''
    def caldis(self,vector1,vector2):
        return np.sqrt(sum((vector1 - vector2) ** 2))
    

    '''
    ====降维====
    transfer是示例化PCA类并设置维度
    返沪降维后的数据
    '''
    def pca(self,data):
        # 实例化PCA并设置维度为2
        transfer = PCA(n_components=2)
        #降维
        transed_data = transfer.fit_transform(data)
        return transed_data

    '''
    =======随机初始化质心=======
    机制是随机从数据点中选k个
    '''
    def random_cluster(self):
        choice = []
        for i in range(self.k):
            choice.append(random.choice(self.data))
        return choice
    
    '''
    =====质心更新=====
     - belong_lst
        - 结构 [[],[],[]] belong_lst[i]表示第i个质心所包含的点,点的类型是np.array
     - cluster
        - last_cluster
         - 表示更新前三个质心
         - 结构 [np.array,np.array,...] 存储的第i个质心的点
        - new_cluster
         - 表示更新后的三个质心
         - 结构同上
    '''
    def update(self):
        while True:
            last_cluster = self.cluster
            belong_lst = [[] for i in range(self.k)]
            for i,point in enumerate(self.data):
                dis = [] #临时距离 用于保存第i个点到第j质心的距离
                for center in self.cluster:
                    dis.append(self.caldis(point,center))
                belong_lst[dis.index(min(dis))].append(self.data[i]) #定位最短距离对应的质心并将点添加到该质心 
            
            # 通过质心包含的点来更新新的质心
            new_cluster = []
            for i in range(self.k):
                x = 0
                y = 0
                if len(belong_lst[i]) != 0:
                    for point in belong_lst[i]:
                        # print(point)
                        # print(self.cluster[i])
                        x += point[0]
                        y += point[1]
                    x /= len(belong_lst[i])
                    y /= len(belong_lst[i])
                new_cluster.append(np.array([x,y]))
            self.cluster = new_cluster

            # 判断是否收敛
            if not self.change(last_cluster,new_cluster):
                break
        return new_cluster,belong_lst
    
    '''
    ====看质心是否发生变化===
    发生变化返回true
    没有返回false
    判断条件是每个质心的坐标是否相等
    '''
    def change(self,last_cluster,new_cluster):
        for i in range(self.k):
            # print(last_cluster[i])
            if last_cluster[i].all() != new_cluster[i].all():
                return True
        return False


    '''
    画图
    '''
    def draw(self):
        color = ['red','green','yellow','pink']
        for index,i in enumerate(self.belong_lst):
            xs = [point[0] for point in i]
            ys = [point[1] for point in i]
            plt.scatter(xs,ys,marker='o',color=color[index],s=40,label='数据点')
        xs = [point[0] for point in self.cluster]
        ys = [point[1] for point in self.cluster]
        plt.scatter(xs,ys,marker='x',color='blue',s=50,label='质心')
        plt.show()


    '''
    ======计算每个k的sse值并绘制sse-k图像
    '''
    def SSE(self):
        sse = []
        for k in range(1,11):   
            self.kmeans(k=k,draw=False)
            cluster,belong_lst = self.cluster,self.belong_lst
            sse.append(self.calsse(cluster,belong_lst))
        plt.plot(range(1,11),sse)
        plt.show()


    '''
    =========计算当前状态下的sse
    '''
    def calsse(self,cluster,belong_lst):
        sse = 0
        # i 表示质心的序号,center表示对应质心的坐标 类型为np.array
        for i,center in enumerate(cluster):
            for point in belong_lst[i]:
                sse += self.caldis(point,center)
        return sse
            


iris = datasets.load_iris()
iris = iris['data'] # 读取数据 类型为np.array

#实例化Kmeans类
kmean = Kmeans(iris)

# 调用kmeans方法
# 参数可传 k 即质心的数量
kmean.kmeans()

# 调用SSE方法绘制sse-k图
# 绘制的图可以用于观测k的最优值
kmean.SSE()