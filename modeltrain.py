import os
import tensorflow as tf
from PIL import Image

IMG_SIZE = 128
LABEL_CNT = 2
P_KEEP_INPUT = 0.8
P_KEEP_HIDDEN = 0.5


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

img_batch, label_batch = read_and_decode('dog_train.tfrecords')


def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


X = tf.placeholder("float", [None, IMG_SIZE, IMG_SIZE, 3])
Y = tf.placeholder("float", [None, 2])

w = init_weights([3, 3, 3, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([3, 3, 128, 128])
w5 = init_weights([4 * 4 * 128, 625])
w_o = init_weights([625, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")



def simple_model(X, w, w_2, w_3, w_4, w_5, w_o, p_keep_input, p_keep_hidden):
    # batchsize * 128 * 128 * 3
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    # 2x2 max_pooling
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # dropout
    l1 = tf.nn.dropout(l1, p_keep_input)  # 64 * 64 * 32

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w_2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_hidden)  # 32 * 32 * 64

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w_3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_hidden)  # 16 * 16 * 128

    l4a = tf.nn.relu(tf.nn.conv2d(l3, w_4, strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.max_pool(l4a, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')  # 4 * 4 * 128
    l4 = tf.reshape(l4, [-1, w_5.get_shape().as_list()[0]])

    l5 = tf.nn.relu(tf.matmul(l4, w_5))
    l5 = tf.nn.dropout(l5, p_keep_hidden)

    return tf.matmul(l5, w_o)


y_pred = simple_model(X, w, w2, w3, w4, w5, w_o, p_keep_input, p_keep_hidden)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pred))

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)




disp_step = 5
save_step = 20
max_step = 1000
step = 0
saver = tf.train.Saver()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and step < max_step:
            step += 1
            imgs, labels = sess.run([img_batch, label_batch])
            sess.run(train_op, feed_dict={X: imgs, Y: labels, p_keep_hidden: P_KEEP_HIDDEN, p_keep_input: P_KEEP_INPUT})
            if epoch % disp_step == 0:

                acc = sess.run(accuracy, feed_dict={X: imgs, Y: labels, p_keep_hidden: 1.0, p_keep_input: 1.0})
                print('%s accuracy is %.2f' % (step, acc))
            if step % save_step == 0:

                save_path = saver.save(sess, './0_train/graph.ckpt', global_step=step)
                print("save graph to %s" % save_path)
    except tf.errors.OutOfRangeError:
        print("reach epoch limit")
    finally:
        coord.request_stop()
    coord.join(threads)
    save_path = saver.save(sess, './0_train/graph.ckpt', global_step=epoch)

print("training is done")