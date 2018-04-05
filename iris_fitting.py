import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from random import sample
iris = load_iris()

#数据集划分
#index_tr: 训练集样本序号
#index_te: 测试集样本序号
index_tr = sample(range(0,50),40)+\
           sample(range(50,100),40)+\
           sample(range(100,150),40)
index_te = [i for i in range(150) if i not in index_tr]

x_data = np.float32(iris.data[index_tr,:].T) #训练集自变量
y_ = np.float32(iris.target[index_tr])       #训练集目标变量

W = tf.Variable(tf.zeros([1,4]))  #线性模型的自变量系数
b = tf.Variable(tf.zeros([1]))    #线性模型的常量值

#构造一个线性模型: y=w1*x1 + w2*x2 + w3*x3 + w4*x4 +b
y = tf.matmul(W,x_data)+b

#最小化方差
loss = tf.reduce_mean(tf.square(y-y_))   #计算实际值与模型输出值的均方误差
optimize = tf.train.GradientDescentOptimizer(0.005)  #构建一个学习速率为0.005的梯度下降优化器
train = optimize.minimize(loss)          #目标函数:最小化均方误差

#启动图&初始化变量
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

#开始训练
for i in range(1000):
    # print(sess.run(y),y_)
    sess.run(train)
print(sess.run(y),y_)
res = sess.run(y)
W = sess.run(W)   #获得训练后线性模型的自变量系数
b = sess.run(b)   #获得训练后线性模型的常量值
sess.close() 

#测试集数据评测
pre = np.dot(W,iris.data[index_te].T)+b #将测试集数据放入训练好的模型中
pre = np.round(pre)  #四舍五入
pr = sum(sum(abs(np.round(pre))==iris.target[index_te]))/len(index_te)
#pr:预测精度
print(pr)
