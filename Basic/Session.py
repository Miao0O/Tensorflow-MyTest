import tensorflow as tf
a = tf.constant(3)
b = tf.constant(6)
with tf.Session() as sess:
    print "a=3, b=6"
    print "addition with constants: %i" %sess.run(a+b)
    print "multiplication with constants: %i" %sess.run(a*b)
    print "division with constants: %i" %sess.run(b/a)
