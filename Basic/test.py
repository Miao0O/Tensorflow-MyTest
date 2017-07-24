import tensorflow as tf
hello = tf.constant("Hello Yan miao")
sess = tf.Session()
print(sess.run(hello))