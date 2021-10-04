import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file
from PIL import Image
import cv2
import glob
from res_zp_deeplab import res_zp_Deeplabv3
from tensorflow.keras.losses import categorical_crossentropy

# -------------------------------------------------------------#
#   Dice loss
# -------------------------------------------------------------#
def dsc(y_true, y_pred):
  smooth = 1.
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
  return score

def dice_loss(y_true, y_pred):
  loss = 1 - dsc(y_true, y_pred)
  return loss
# -------------------------------------------------------------#
#   Focal Tversky loss
# -------------------------------------------------------------#
def tversky(y_true, y_pred):
    smooth = 1.
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def unit_loss(y_true, y_pred):
    return (2 - tversky(y_true, y_pred)-dsc(y_true, y_pred))/2


# -------------------------------------------------------------#
#   IoU
# -------------------------------------------------------------#
def iou(y_true, y_pred):
  smooth = 1.
  intersection = K.sum(y_true * y_pred)
  sum = K.sum(y_true + y_pred)
  iou = (intersection + smooth) / (sum - intersection + smooth)
  return iou

# -------------------------------------------------------------#
#   A generater to read images and labels
# -------------------------------------------------------------#
def generate_arrays_from_file(lines, batch_size):
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            img = glob.glob("./img/" + name)
            img = cv2.resize(cv2.imread(img[0], 1), (int(WIDTH), int(HEIGHT)))
            img = np.array(img) / 255.0
            X_train.append(img)
            name = lines[i].split(';')[1].split()[0]
            label = glob.glob(r"./label/" + name)
            label = cv2.resize(cv2.imread(label[0], 0), (int(WIDTH), int(HEIGHT)))
            label = np.array(label) / 255.0
            if len(np.shape(label)) == 3:
                label = np.array(label)[:, :, 0]
            label = np.reshape(np.array(label)/255, [-1])
            one_hot_label = np.eye(NCLASSES)[np.array(label, np.int32)]
            Y_train.append(one_hot_label)
        yield (np.array(X_train), np.array(Y_train))

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------#
    #   Define the height and width of the input image, and the number of types
    # ------------------------------------------------------------------------------------------#
    HEIGHT = 512
    WIDTH = 512
    batch_size = 4
    NCLASSES = 2

    log_dir ="./log/"
    lr_set = 1e-7
    Xcep_weight_dir = None#r"C:\wyh\deeplab\deeplabv3-X\text_xception.h5"
    model = res_zp_Deeplabv3(classes=NCLASSES, input_shape=(HEIGHT, WIDTH, 3))#,train_cl=True,Xcep_weight_dir=Xcep_weight_dir)
    weights_path = './log/last1.h5'
    model.load_weights(weights_path, by_name=True)

    
    with open("./train.txt", "r") as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-LOSS{loss:.3f}-val_LOSS{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True,mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1,mode='min')
    early_stopping = EarlyStopping(monitor='val_iou', min_delta=0, patience=10, verbose=1,mode='max')

    if True:
        lr = lr_set
        #batch_size = 2
        model.compile(loss=dice_loss,#focal_tversky,#dice_loss,#tversky_loss,#'categorical_crossentropy',#custom_loss,
                      optimizer=Adam(lr=lr),
                      metrics=[ dice_loss,tversky_loss,focal_tversky, iou,'acc'])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=100,
                            initial_epoch=0,
                            callbacks=[checkpoint, reduce_lr, tensorboard_callback])#, early_stopping])
        model.save_weights(log_dir + 'last1.h5')

