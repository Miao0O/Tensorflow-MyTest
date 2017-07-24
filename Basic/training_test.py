import tensorflow as tf
import numpy as np #ke xue ji suan

#creat data
x_data = np.random.rand(100).astype(np.float32) #tensorflos most number is float32
y_data = x_data*0.1+0.3

###create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0)) #sui ji shu lie shengcheng, one-dimensional initial number is random number
bias = tf.Variable(tf.zeros([1])) #initial number is zero

y = Weights*x_data+bias

loss = tf.reduce_mean(tf.square(y-y_data)) #diffrence between the predicted y and the real y
optimizer = tf.train.GradientDescentOptimizer(0.5) #0.5 learning efficiency
train = optimizer.minimize(loss)

init = tf.initialize_all_variables() #chu shi hua jie gou

### create tensorflow structure end###

sess = tf.Session() #dingyi session
sess.run(init) #ji huo chu li difang, activate the process, it's very important

for step in range (201):
    sess.run(train)
    if step%20==0:
        print step, sess.run(Weights), sess.run(bias)
