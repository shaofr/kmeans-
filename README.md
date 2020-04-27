## kmeans++算法

还存在一些问题，在数据量较大时会出现

+ 初始k个类中心时是按照概率选取的，当数据多时选取概率低，不可避免有系统误差，无法准确存储概率，在使用np.random.choice(n,1,p=[d/sum_dist for p d in dist])时，概率求和不为1

    