# encoding=utf-8
import pandas as pd
import os
from PIL import Image,ImageOps
import tensorflow as tf
import preprocessing
import inference
import train
import numpy as np
TEST_PATH="/home/administrator/PengXiao/plant/dataset/test"

def predict():
    pred_imgs = []
    pred_label = []
    result = []
    files = [x[2] for x in os.walk(TEST_PATH)]
    for file in files[0]:
        file_path = os.path.join(TEST_PATH,file)
        new_img = Image.open(file_path)
        new_img = ImageOps.fit(new_img, (96, 96),Image.ANTIALIAS).convert('RGB')
        pred_imgs.append(np.array(new_img))

    x=tf.placeholder(tf.float32,[len(pred_imgs),preprocessing.IMAGE_SIZE,preprocessing.IMAGE_SIZE,preprocessing.IMAGE_CHANNELS],'x-input')
    y=inference.infer(x,False,None)
    saver=tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt=tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH,"checkpoint")
        if ckpt and ckpt.model_checkpoint_path:
            save_path=ckpt.model_checkpoint_path
            saver.restore(sess,save_path)
            pred=sess.run(y,feed_dict={x:pred_imgs})
            pred_idx = np.argmax(pred,axis=1)

            for i in range(len(pred_idx)):
                if pred_idx[i] == 0:
                    pred_label.append("Black-grass")
                elif pred_idx[i] == 1:
                    pred_label.append("Cleavers")
                elif pred_idx[i] == 2:
                    pred_label.append("Common wheat")
                elif pred_idx[i] == 3:
                    pred_label.append("Loose Silky-bent")
                elif pred_idx[i] == 4:
                    pred_label.append("Scentless Mayweed")
                elif pred_idx[i] == 5:
                    pred_label.append("Small-flowered Cranesbill")
                elif pred_idx[i] == 6:
                    pred_label.append("Charlock")
                elif pred_idx[i] == 7:
                    pred_label.append("Common Chickweed")
                elif pred_idx[i] == 8:
                    pred_label.append("Fat Hen")
                elif pred_idx[i] == 9:
                    pred_label.append("Maize")
                elif pred_idx[i] == 10:
                    pred_label.append("Shepherds Purse")
                elif pred_idx[i] == 11:
                    pred_label.append("Sugar beet")

            df = pd.DataFrame(data={'file': files[0], 'species': pred_label})
            print(df)
            df_sort = df.sort_values(by=['file'])
            df_sort.to_csv('/home/administrator/PengXiao/plant/results.csv', index=False, sep=',')

def main(argv=None):
    predict()

if __name__ == '__main__':
    tf.app.run()