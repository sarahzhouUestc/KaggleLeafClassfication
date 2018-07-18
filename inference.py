# encoding=utf-8
import tensorflow as tf

def infer(input_tensor, train, regularizer, reuse):
    with tf.variable_scope("layer1-conv1",reuse=reuse):
        conv1_weights=tf.get_variable("weight",[3,3,3,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("bias",[16],initializer=tf.constant_initializer(0.0))
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    # with tf.variable_scope("layer2-pool1",reuse=reuse):
    #     pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.variable_scope("layer3-conv2",reuse=reuse):
        conv2_weights=tf.get_variable("weight",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable("bias",[16],initializer=tf.constant_initializer(0.0))
        conv2=tf.nn.conv2d(relu1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    with tf.variable_scope("layer4-pool2",reuse=reuse):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.variable_scope("layer5-conv3",reuse=reuse):
        conv3_weights=tf.get_variable("weight",[3,3,16,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases=tf.get_variable("bias",[32],initializer=tf.constant_initializer(0.0))
        conv3=tf.nn.conv2d(pool2,conv3_weights,strides=[1,1,1,1],padding="SAME")
        relu3=tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))
    with tf.variable_scope("layer6-conv4",reuse=reuse):
        conv4_weights = tf.get_variable("weight",[3,3,32,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias",[32],initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3,conv4_weights,strides=[1,1,1,1],padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))
    with tf.variable_scope("layer7-pool3",reuse=reuse):
        pool3=tf.nn.max_pool(relu4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # convert to vector
    pool3_shape=pool3.get_shape().as_list()
    nodes=pool3_shape[1]*pool3_shape[2]*pool3_shape[3]
    reshaped=tf.reshape(pool3,[pool3_shape[0],nodes])

    with tf.variable_scope("layer8-fc1",reuse=reuse):
        fc1_weights=tf.get_variable("weight",[nodes,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("loss",regularizer(fc1_weights))
        fc1_biases=tf.get_variable("bias",[64],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        # if train:
        #     fc1=tf.nn.dropout(fc1,keep_prob=0.9)

    with tf.variable_scope("layer9-fc2",reuse=reuse):
        fc2_weights=tf.get_variable("weight",[64,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("loss",regularizer(fc2_weights))
        fc2_biases=tf.get_variable("bias",[32],initializer=tf.constant_initializer(0.1))
        fc2=tf.nn.relu(tf.matmul(fc1,fc2_weights)+fc2_biases)
        # if train:
        #     fc2=tf.nn.dropout(fc2,keep_prob=0.9)

    with tf.variable_scope("layer10-fc3",reuse=reuse):
        fc3_weight=tf.get_variable("weight",[32,12],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("loss",regularizer(fc3_weight))
        fc3_biases=tf.get_variable("bias",[12],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc2,fc3_weight)+fc3_biases
    return tf.nn.softmax(logit)

