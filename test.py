# encoding=utf-8
import tensorflow as tf
import inference
import train
import preprocessing
import os

def eval(testX, testY):
    x=tf.placeholder(tf.float32,[len(testX),preprocessing.IMAGE_SIZE,preprocessing.IMAGE_SIZE,preprocessing.IMAGE_CHANNELS],\
                     'x-input')
    y_=tf.placeholder(tf.float32,[len(testX),preprocessing.OUTPUT_NODE],'y-input')
    y=inference.infer(x,False,None)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32))
    saver=tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        xs_reshaped=tf.reshape(testX,[len(testX),preprocessing.IMAGE_SIZE,preprocessing.IMAGE_SIZE,preprocessing.IMAGE_CHANNELS])
        test_feed={x:sess.run(xs_reshaped),y_:testY}
        ckpt=tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH,"checkpoint")
        if ckpt and ckpt.model_checkpoint_path:
            save_path=ckpt.model_checkpoint_path
            saver.restore(sess,save_path)
            acc=sess.run(accuracy,test_feed)
            step=save_path.split("/")[-1].split("-")[-1]
            print("After {} steps, the accuracy on test is {}".format(step,acc))

def main(argv=None):
    trainX, testX, trainY, testY = preprocessing.create_dataset(0.1)
    eval(testX,testY)

if __name__ == '__main__':
    tf.app.run()