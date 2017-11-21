import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 操作符号变量来描述这些可交互的操作单元
# 第一个维度任何长度
x = tf.placeholder("float", [None, 784])
# 全为零的张量来初始化W和b
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# softmax 函数
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 计算交叉熵，新的占位符
y_ = tf.placeholder("float", [None,10])
# 交叉熵公式
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 梯度下降算法最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 预测是否真实标签匹配
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 布尔值转换成浮点数
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
