# encoding=utf-8
import tensorflow as tf
import preprocessing
import inference
import os
import pandas as pd
import random
import test

LEARNING_RATE_BASE=0.05
LEARNING_RATE_DECAY=0.9
BATCH_SIZE=64
REGULARIZATION_RATE=0.0001
TRAINING_STEP=2000
MODEL_SAVE_PATH="/home/administrator/PengXiao/plant/model"
MODEL_NAME="plant.ckpt"

def train(trainX, trainY, testX, testY):
    print("Start to train")
    print("The length of training data is {}".format(len(trainX)))
    x = tf.placeholder(tf.float32, [BATCH_SIZE, preprocessing.IMAGE_SIZE, preprocessing.IMAGE_SIZE,
                                    preprocessing.IMAGE_CHANNELS], "x-input")
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, preprocessing.OUTPUT_NODE], "y-input")
    global_step = tf.Variable(0.0, dtype=tf.float32, trainable=False)

    # regularization
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference.infer(x, True, None, False)
    # cross entropy loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, axis=1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean
    # loss = cross_entropy_mean + tf.add_n(tf.get_collection("loss"))
    # learning rate
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, len(trainX) / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # save the model
    saver = tf.train.Saver()

    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, global_step)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    train_len = len(trainX)

    # The following is accuracy on test data.
    test = tf.placeholder(tf.float32, [1, preprocessing.IMAGE_SIZE, preprocessing.IMAGE_SIZE, preprocessing.IMAGE_CHANNELS], 'test-input')
    test_y_ = tf.placeholder(tf.float32, [1, preprocessing.OUTPUT_NODE], 'test-y-input')
    test_y = inference.infer(test, False, None, True)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(test_y, 1), tf.argmax(test_y_, 1)), tf.float32))
    test_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(test_y_,axis=1),logits=test_y))
    # accuracy end

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEP):
            for _ in range(int(train_len/BATCH_SIZE)+1):
                l = list(range(train_len))
                rdn = random.sample(l, BATCH_SIZE)
                xs = []
                ys = []
                for r in rdn:
                    xs.append(trainX[r])
                    ys.append(trainY[r])
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 10 == 0:
                acc_result = 0
                for j in range(len(testX)):
                    testX_feed = testX[j].reshape(1, preprocessing.IMAGE_SIZE, preprocessing.IMAGE_SIZE, 3)
                    testY_feed = testY[j].reshape(1, 12)
                    acc_result += sess.run(accuracy, feed_dict={test: testX_feed, test_y_: testY_feed})
                acc = acc_result / len(testX)
                print("After %d epoch(s),loss on training batch is %f, accuracy on test is %f" % (i, loss_value, acc))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step)
