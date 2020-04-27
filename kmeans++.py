import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets
from sys import maxsize

#返回两个点距离平方
def distance(a,b):
    return np.sum(np.power(a-b,2))
#随机选取k个类中心
def random_pick(k,n,X):
    clusters=X[np.random.choice(n,k,replace=False)]
    dist=[maxsize/n] * n
    nearst = [None] * n
    for i in range(n):
        for j in range(k):
            if distance(X[i], clusters[j]) < dist[i]:
                dist[i] = distance(X[i], clusters[j])
                nearst[i] = j
    return clusters,nearst

def kmeans(k,X):
    n = X.shape[0]
    d = X.shape[1]
    # clusters记录点坐标，nearst记录每个点与clusters中哪个中心最近，只记录序号，dist记录每个点到最近类中心距离平方和
    # 这里暂时只有nearst在下面有用
    clusters, nearst= random_pick(k, n, X)
    phi = maxsize
    dist = [maxsize / n] * n
    # new_clusters为新的类中心，初始化每个点为np.zeros((1,d))[0]
    new_clusters = [np.zeros((1, d))[0]] * k
    # clusters_num记录每个类有多少个点，用于求平均
    clusters_num = [0] * k
    # 更新类中心
    while True:
        # 计算新类中心
        for i in range(n):
            new_clusters[nearst[i]] = new_clusters[nearst[i]] + X[i]
            clusters_num[nearst[i]] += 1
        for i in range(k):
            new_clusters[i] /= clusters_num[i]
        # 计算每个点属于哪一类
        for i in range(n):
            for j in range(k):
                if distance(X[i], new_clusters[j]) < dist[i]:
                    dist[i] = distance(X[i], new_clusters[j])
                    nearst[i] = j
        # 此时还没有更新phi，如果phi=当前计算出的新phi，则返回
        if (phi == sum(dist)):
            return phi, new_clusters, nearst
        else:
            new_clusters = [np.zeros((1, d))] * k
            clusters_num = [0] * k
            phi = sum(dist)
            dist = [maxsize / n] * n

def keanspp_pick(k,n,X):
    #这里dist就是每个点距离最近类中心距离的平方，初始设为maxsize/n是为了在随机选取时，分母sum_dist不超过整数最大范围
    #maxsize=9223372036854775807
    dist = [maxsize/n] * n
    #初始聚类为空
    clusters = []
    #nearst保存距离每个点最近的类中心编号，也就是说在clusters中编号
    nearst = [None] * n
    #按照D2选取k个类中心
    for j in range(k):
        #概率的分母是dist求和
        sum_dist=sum(dist)
        #c是按照D2选取的一个点，从range(n)中选取1个，按照p的概率分布，这里c[0]只是序号，X[c[0]]才是选取的点
        c=np.random.choice(n, 1, replace=False,p=[a/sum_dist for a in dist])
        #new_cluster是新加入的类中心
        new_cluster=X[c[0]]
        #new_dist是每个点到新加入类中心的距离
        new_dist=[distance(p,new_cluster) for p in X]
        #更新每个点最近的类中心nearst
        for i in range(n):
            if new_dist[i]<dist[i]:
                dist[i]=new_dist[i]
                nearst[i]=j
        clusters.append(new_cluster)
    #clusters是一个数组，数组元素是array
    return clusters,nearst,dist


def kmeanspp(k,X):
    n = X.shape[0]
    d = X.shape[1]
    #clusters记录点坐标，nearst记录每个点与clusters中哪个中心最近，只记录序号，dist记录每个点到最近类中心距离平方和
    #这里暂时只有nearst在下面有用
    clusters,nearst,dist = keanspp_pick(k, n, X)
    phi = sum(dist)
    dist = [maxsize/n] * n
    #new_clusters为新的类中心，初始化每个点为np.zeros((1,d))[0]
    new_clusters=[np.zeros((1,d))[0]]*k
    #clusters_num记录每个类有多少个点，用于求平均
    clusters_num=[0]*k
    #更新类中心
    while True:
        #计算新类中心
        for i in range(n):
            new_clusters[nearst[i]] =new_clusters[nearst[i]]+X[i]
            clusters_num[nearst[i]]+=1
        for i in range(k):
            new_clusters[i]/=clusters_num[i]
        #计算每个点属于哪一类
        for i in range(n):
            for j in range(k):
                if distance(X[i],new_clusters[j])<dist[i]:
                    dist[i]=distance(X[i],new_clusters[j])
                    nearst[i]=j
        #此时还没有更新phi，如果phi=当前计算出的新phi，则返回
        if(phi==sum(dist)):
            return phi,new_clusters,nearst
        else:
            new_clusters = [np.zeros((1, d))] * k
            clusters_num = [0] * k
            phi=sum(dist)
            dist = [maxsize / n] * n


phi,clusters,nearst=kmeanspp(5,np.arange(1000).reshape(100,10))
print(clusters)
print(phi)
print(nearst)

phi2,clusters2,nearst2=kmeans(5,np.arange(1000).reshape(100,10))
print(clusters2)
print(phi2)
print(nearst2)