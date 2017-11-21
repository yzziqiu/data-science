import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 生成101组二维训练数据
trX = np.linspace(-1, 1, 101) #创建-1到1之间的一维等差数组
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # 目标函数为y=2x，并施加一个高斯分布的噪音

X = tf.placeholder("float") #创建占位符
Y = tf.placeholder("float")


def model(X, w):
    return tf.mul(X, w) # 线性回归模型：x与w的内积


w = tf.Variable(0.0, name="weights") # 为模型权重W创建共享变量
y_model = model(X, w)

cost = tf.square(Y - y_model) # 定义平方损失函数

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) #使用步长为0.01的梯度下降算法，最小化损失函数cost

# 创建会话
with tf.Session() as sess:
    # 初始化变量W
    tf.initialize_all_variables().run()

    for i in range(100):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    weight = sess.run(w)
    print(weight)   #输出训练完成的权重w，结果应接近2
#画图
fig = plt.figure()
ax1 = fig.add_subplot(111)
xx = np.linspace(-1, 1, 101)
yy = xx*weight
ax1.scatter(trX ,trY)
plt.plot(xx,yy,"r")
plt.show() 
