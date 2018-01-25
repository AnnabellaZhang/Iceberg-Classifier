from __future__ import print_function, division
# https://www.kaggle.com/yuhaichina/single-model-vgg16-mobilenet-lb-0-1568-with-tf
import time
# Global parameters
image_size = 299
batch_size=16
modelname = 'ResNet50+InceptionV3' # 'VGG16'
model2weight = 'imagenet'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from os.path import join as opj
import keras
import math
from utils import create_logger

#from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pylab
#plt.rcParams['figure.figsize'] = 10, 10
#%matplotlib inline

# limit GPU usage
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def print_and_log(string, logger):
    print(string)
    if logger:
        logger.info(string)

cur_time = time.strftime('%Y-%m-%d-%H-%M')
if not os.path.exists('output/' + modelname + "-" + cur_time):
    exp_name = modelname + "-" + cur_time
    out_path = 'output/' + modelname + "-" + cur_time
    os.makedirs(out_path)
    logger = create_logger('./{}/logs'.format(out_path), exp_name)
    print_and_log('Creating folder: {}'.format(out_path), logger)


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

train = pd.read_json("./data/train.json")
target_train=train['is_iceberg']
test = pd.read_json("./data/test.json")

target_train=train['is_iceberg']
test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce') # invalid parsing will be set as NaN
train['inc_angle']=pd.to_numeric(train['inc_angle'], errors='coerce')#We have only 133 NAs.
train['inc_angle']=train['inc_angle'].fillna(method='pad')  #propagate last valid observation
X_angle=train['inc_angle']
test['inc_angle']=pd.to_numeric(test['inc_angle'], errors='coerce')
X_test_angle=test['inc_angle']

#Generate the training data
X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_band_3_0 = (X_band_1+X_band_2)/2
X_band_3 = np.fabs(np.subtract(X_band_1,X_band_2))
X_band_4 = np.maximum(X_band_1,X_band_2)
X_band_5 = np.minimum(X_band_1,X_band_2)
X_train = np.concatenate([     
                          X_band_3[:, :, :, np.newaxis],X_band_4[:, :, :, np.newaxis],X_band_5[:, :, :, np.newaxis]], axis=-1)



X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_band_test_3_0 = (X_band_test_1+X_band_test_2)/2
X_band_test_3=np.fabs(np.subtract(X_band_test_1,X_band_test_2))
X_band_test_4=np.maximum(X_band_test_1,X_band_test_2)
X_band_test_5=np.minimum(X_band_test_1,X_band_test_2)
X_test = np.concatenate([
                          X_band_test_3[:, :, :, np.newaxis], X_band_test_4[:, :, :, np.newaxis],X_band_test_5[:, :, :, np.newaxis]],axis=-1)

# creating size data
def iso(arr):
    p = np.reshape(np.array(arr), [75,75]) >(np.mean(np.array(arr))+2*np.std(np.array(arr)))
    return p * np.reshape(np.array(arr), [75,75])
def size(arr):
    return float(np.sum(arr<-5))/(75*75)

train['iso1'] = train.iloc[:, 0].apply(iso)
train['iso2'] = train.iloc[:, 1].apply(iso)
test['iso1'] = test.iloc[:, 0].apply(iso)
test['iso2'] = test.iloc[:, 1].apply(iso)
# Feature engineering s1 s2 and size.
train['s1'] = train.iloc[:,5].apply(size)
train['s2'] = train.iloc[:,6].apply(size)
test['s1'] = test['iso1'].apply(size)
test['s2'] = test['iso2'].apply(size)
X_size_1=train['s1']
X_size_2=train['s2']
X_test_size1=test['s1']
X_test_size2=test['s2']

from tqdm import tqdm
from skimage.transform import resize
import six.moves.cPickle as pickle
if image_size != 75:
    if os.path.exists('./data/X_new_train_{}.pkl'.format(image_size)) and os.path.exists('./data/X_new_test_{}.pkl'.format(image_size)):
        print_and_log('load resized images...', logger)
        X_train = pickle.load(open('./data/X_new_train_{}.pkl'.format(image_size), 'rb'))
        X_test = pickle.load(open('./data/X_new_test_{}.pkl'.format(image_size), 'rb'))
    else:
        print_and_log('resize images...', logger)
        width = image_size
        n = len(X_train)
        X_train_resized = np.zeros((n, width, width, 3), dtype=np.float32)
        for i in tqdm(range(n)):
            x = X_train[i]
            x = (x - x.min()) / (x.max() - x.min())  # normalize for each pseudo pixel value
            X_train_resized[i] = resize(x, (width, width), mode='reflect')
        X_train = X_train_resized
        n = len(X_test)
        X_test_resized = np.zeros((n, width, width, 3), dtype=np.float32)
        for i in tqdm(range(n)):
            x = X_test[i]
            x = (x - x.min()) / (x.max() - x.min())  # normalize for each pseudo pixel value
            X_test_resized[i] = resize(x, (width, width), mode='reflect')
        X_test = X_test_resized
        pickle.dump(X_train, open('./data/X_new_train_{}.pkl'.format(image_size), 'wb'), 4)
        pickle.dump(X_test, open('./data/X_new_test_{}.pkl'.format(image_size), 'wb'), 4)


#from matplotlib import pyplot
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.optimizers import rmsprop
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.datasets import cifar10
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import  InceptionV3
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input    

#Data Aug for multi-input
from keras.preprocessing.image import ImageDataGenerator
# Define the image transformations here
gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0., #0.2,
                         height_shift_range = 0., #0.2,
                         channel_shift_range=0,
                         zoom_range = 0.5,
                         rotation_range = 10 #30
                         )

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_four_inputs(X1, X2,X3,X4, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=55) #creating batch data
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=55)
    genX3 = gen.flow(X1, X3, batch_size=batch_size, seed=55)
    genX4 = gen.flow(X1, X4, batch_size=batch_size, seed=55)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            X4i = genX4.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1],X3i[1],X4i[1]], X1i[1]

# Finally create generator
def get_callbacks(filepath, patience=2):
   #es = EarlyStopping('val_loss', patience=10, mode="min")
   es = EarlyStopping('val_loss', patience=20, mode="min")
   msave = ModelCheckpoint(filepath, save_best_only=True)
   return [es, msave]


def getVggAngleModel():
    print_and_log("Base Model: {}".format(modelname), logger)

    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)

    input_3 = Input(shape=[1], name="s1")
    s1_layer = Dense(1, )(input_3)

    input_4 = Input(shape=[1], name="s2")
    s2_layer = Dense(1, )(input_4)
    with tf.device('/gpu:0'):
        if modelname == "VGG16":
            base_model = VGG16(weights='imagenet', include_top=False,
                        input_shape=X_train.shape[1:], classes=1)
            x = base_model.get_layer('block5_pool').output
        elif modelname == "Xception":
            base_model = Xception(include_top=False, weights='imagenet',
                                        input_shape=X_train.shape[1:],classes=1)
            x = base_model.output
        elif modelname == "VGG19":
            base_model = VGG19(include_top=False,weights='imagenet',
                               input_shape=X_train.shape[1:],classes=1)
            x = base_model.get_layer('block5_pool').output

        elif modelname == "ResNet50":
            base_model = ResNet50(include_top=False, weights='imagenet', input_shape=X_train.shape[1:], classes=1)
            x = base_model.get_layer('avg_pool').output

        elif modelname == "InceptionV3":
            base_model = InceptionV3(include_top=False,weights='imagenet',
                                                    input_shape=X_train.shape[1:],pooling=None,classes=1)
            x = base_model.output
        elif modelname == "ResNet50+InceptionV3":
            base_model = ResNet50(include_top=False, weights='imagenet', input_shape=X_train.shape[1:], classes=1)
            x = base_model.get_layer('avg_pool').output
        else:
            print_and_log("BaseModel error, default VGG16", logger)
            base_model = VGG16(weights='imagenet', include_top=False,
                               input_shape=X_train.shape[1:], classes=1)
            x = base_model.get_layer('block5_pool').output
        x = GlobalMaxPooling2D()(x)

    # original: weights = None , alpha = 0.9
    with tf.device('/gpu:1'):
        if modelname == "ResNet50+InceptionV3":
            base_model2 = InceptionV3(include_top=False,weights='imagenet',
                                                    input_tensor=base_model.input,pooling=None,classes=1)
            x2 = base_model2.output
        else:
            print_and_log("BaseModel2 error, default MobileNet", logger)
            if model2weight == 'imagenet':
                print("Model2 weight:imagenet")
                base_model2 = keras.applications.mobilenet.MobileNet(weights='imagenet', # None,
                                    alpha=1.0, # 0.9,
                                        input_tensor = base_model.input,include_top=False, input_shape=X_train.shape[1:])
            else:
                print("Model2 weight: None")
                base_model2 = keras.applications.mobilenet.MobileNet(weights= None,
                                    alpha=0.9,
                                        input_tensor=base_model.input, include_top=False,input_shape=X_train.shape[1:])
            base_model2.get_layer('conv1').name = 'mb_conv1'
            x2 = base_model2.output
        x2 = GlobalAveragePooling2D()(x2)


    merge_one = concatenate([x, x2, angle_layer,s1_layer,s2_layer])

    merge_one = Dropout(0.6)(merge_one)
    predictions = Dense(1, activation='sigmoid',kernel_initializer='he_normal')(merge_one)
    
    model = Model(input=[base_model.input, input_2,input_3,input_4], output=predictions)
    
    sgd = Adam(lr=1e-4) #SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


#Using K-fold Cross Validation with Data Augmentation.
def myAngleCV(X_train, X_angle,X_size_1,X_size_2, X_test):
    K = 5  # K-fold
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))
    y_test_pred_log = 0
    y_train_pred_log=0
    y_valid_pred_log = 0.0*target_train
    for j, (train_idx, test_idx) in enumerate(folds):
        print_and_log('\n===================FOLD={}'.format(j), logger)
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout= target_train[test_idx]
        
        #Angle
        X_angle_cv=X_angle[train_idx]
        X_angle_hold=X_angle[test_idx]

        #Size
        X_size_1_cv = X_size_1[train_idx]
        X_size_1_hold = X_size_1[test_idx]
        X_size_2_cv = X_size_2[train_idx]
        X_size_2_hold = X_size_2[test_idx]

        #define file path and get callbacks
        file_path = 'output/' + modelname + "-" + cur_time + "/%s_aug_model_weights.hdf5"%j
        callbacks = get_callbacks(filepath=file_path, patience=10)
        gen_flow = gen_flow_for_four_inputs(X_train_cv, X_angle_cv,X_size_1_cv,X_size_2_cv, y_train_cv)
        galaxyModel= getVggAngleModel()
        print_and_log("N_train: {}".format(len(X_train)), logger)
        print_and_log("Batch_size: {}".format(batch_size), logger)
        print_and_log("Steps_per_epoch: {}".format(math.ceil(len(X_train)/batch_size)), logger)
        galaxyModel.fit_generator(
                gen_flow,
                steps_per_epoch=math.ceil(len(X_train)/batch_size),
                epochs=100,
                shuffle=True,
                verbose=1,
                validation_data=([X_holdout,X_angle_hold,X_size_1_hold,X_size_2_hold], Y_holdout),
                callbacks=callbacks)

        #Getting the Best Model
        galaxyModel.load_weights(filepath=file_path)
        #Getting Training Score
        score = galaxyModel.evaluate([X_train_cv,X_angle_cv,X_size_1_cv,X_size_2_cv], y_train_cv, verbose=0)
        print_and_log('fold {}, Train loss: {}'.format(j, score[0]), logger)
        print_and_log('fold {}, Train accuracy: {}'.format(j, score[1]), logger)
        #Getting val Score
        score = galaxyModel.evaluate([X_holdout,X_angle_hold,X_size_1_hold,X_size_2_hold], Y_holdout, verbose=0)
        print_and_log('fold {}, Val loss: {}'.format(j, score[0]), logger)
        print_and_log('fold {}, Val accuracy: {}'.format(j, score[1]), logger)

        #Getting validation Score.
        pred_valid=galaxyModel.predict([X_holdout,X_angle_hold,X_size_1_hold,X_size_2_hold])
        y_valid_pred_log[test_idx] = pred_valid.reshape(pred_valid.shape[0])

        #Getting Test Scores
        temp_test=galaxyModel.predict([X_test, X_test_angle,X_test_size1,X_test_size2])
        y_test_pred_log+=temp_test.reshape(temp_test.shape[0])

        #Getting Train Scores
        temp_train=galaxyModel.predict([X_train, X_angle,X_size_1,X_size_2])
        y_train_pred_log+=temp_train.reshape(temp_train.shape[0])

    y_test_pred_log=y_test_pred_log/K
    y_train_pred_log=y_train_pred_log/K

    print_and_log('\n Train Log Loss Validation= {}'.format(log_loss(target_train, y_train_pred_log)), logger)
    print_and_log(' Val Log Loss Validation= {}'.format(log_loss(target_train, y_valid_pred_log)), logger)
    return y_test_pred_log

preds=myAngleCV(X_train, X_angle,X_size_1,X_size_2, X_test)
#Submission for each day.
submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=preds
submission.to_csv("sub-new-" + modelname + "-" + cur_time + ".csv", index=False)
