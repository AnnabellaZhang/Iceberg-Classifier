from __future__ import print_function, division
# https://www.kaggle.com/yuhaichina/single-model-vgg16-mobilenet-lb-0-1568-with-tf
import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from os.path import join as opj
import keras
import math
from utils import create_logger
filelist = [
    'ResNet50+InceptionV3-2018-01-16-20-34',
    'ResNet50+InceptionV3-2018-01-16-11-05',
    'ResNet50+InceptionV3-2018-01-22-15-46'
]
print(filelist)
rootpath = '/data/zrb/Iceberg-Classifier/output/'
modelname = 'ResNet50+InceptionV3'
image_size = 299
crop_size = 350

def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

def print_and_log(string, logger):
    print(string)
    if logger:
        logger.info(string)

cur_time = time.strftime('%Y-%m-%d-%H-%M')
if not os.path.exists('output/10-crop-' + modelname + "-" + cur_time):
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

# Get data
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
#X_band_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in train["inc_angle"]])

X_train = np.concatenate([
                          X_band_3[:, :, :, np.newaxis],X_band_4[:, :, :, np.newaxis],X_band_5[:, :, :, np.newaxis]], axis=-1)
#print(X_train.shape())



X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])


X_band_test_3_0 = (X_band_test_1+X_band_test_2)/2
X_band_test_3=np.fabs(np.subtract(X_band_test_1,X_band_test_2))
X_band_test_4=np.maximum(X_band_test_1,X_band_test_2)
X_band_test_5=np.minimum(X_band_test_1,X_band_test_2)
#X_band_test_3=np.array([np.full((75, 75), angel).astype(np.float32) for angel in test["inc_angle"]])
X_test = np.concatenate([
                          X_band_test_3[:, :, :, np.newaxis], X_band_test_4[:, :, :, np.newaxis],X_band_test_5[:, :, :, np.newaxis]],axis=-1)


def resizeX(image_size,X_train):
    width = image_size
    n = len(X_train)
    X_train_resized = np.zeros((n, width, width, 3), dtype=np.float32)
    for i in tqdm(range(n)):
        x = X_train[i]
        x = (x - x.min()) / (x.max() - x.min())  # normalize for each pseudo pixel value
        X_train_resized[i] = resize(x, (width, width), mode='reflect')
    return X_train_resized

#resize data
from tqdm import tqdm
from skimage.transform import resize
import six.moves.cPickle as pickle
if os.path.exists('./data/X_train_{}.pkl'.format(image_size)) and os.path.exists('./data/X_test_{}.pkl'.format(image_size)):
    print_and_log('load resized images...', logger)
    X_train = pickle.load(open('./data/X_train_{}.pkl'.format(image_size), 'rb'))
    X_test = pickle.load(open('./data/X_test_{}.pkl'.format(image_size), 'rb'))
else:
    print_and_log('resize images...', logger)
    X_train = resizeX(image_size, X_train)
    X_test = resizeX(image_size, X_test)
    pickle.dump(X_train, open('./data/X_train_{}.pkl'.format(image_size), 'wb'), 4)
    pickle.dump(X_test, open('./data/X_test_{}.pkl'.format(image_size), 'wb'), 4)

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
def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=55) #creating batch data
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=55)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]


def getVggAngleModel():
    print_and_log("Base Model: {}".format(modelname), logger)

    input_2 = Input(shape=[1], name="angle")
    angle_layer = Dense(1, )(input_2)

    with tf.device('/gpu:0'):
        if modelname == "VGG16":
            base_model = VGG16(weights='imagenet', include_top=False,
                               input_shape=X_train.shape[1:], classes=1)
            x = base_model.get_layer('block5_pool').output
        elif modelname == "Xception":
            base_model = Xception(include_top=False, weights='imagenet',
                                  input_shape=X_train.shape[1:], classes=1)
            x = base_model.output
        elif modelname == "VGG19":
            base_model = VGG19(include_top=False, weights='imagenet',
                               input_shape=X_train.shape[1:], classes=1)
            x = base_model.get_layer('block5_pool').output

        elif modelname == "ResNet50":
            base_model = ResNet50(include_top=False, weights='imagenet', input_shape=X_train.shape[1:], classes=1)
            x = base_model.get_layer('avg_pool').output

        elif modelname == "InceptionV3":
            base_model = InceptionV3(include_top=False, weights='imagenet',
                                     input_shape=X_train.shape[1:], pooling=None, classes=1)
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
            base_model2 = InceptionV3(include_top=False, weights='imagenet',
                                      input_tensor=base_model.input, pooling=None, classes=1)
            x2 = base_model2.output
        else:
            print_and_log("BaseModel2 error, default MobileNet", logger)
            base_model2 = keras.applications.mobilenet.MobileNet(weights='imagenet',  # None,
                                                                 alpha=1.0,  # 0.9,
                                                                 input_tensor=base_model.input, include_top=False,
                                                                 input_shape=X_train.shape[1:])
            base_model2.get_layer('conv1').name = 'mb_conv1'
            x2 = base_model2.output
        x2 = GlobalAveragePooling2D()(x2)

    merge_one = concatenate([x, x2, angle_layer])

    merge_one = Dropout(0.6)(merge_one)
    predictions = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(merge_one)

    model = Model(input=[base_model.input, input_2], output=predictions)

    sgd = Adam(lr=1e-4)  # SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw+1,centerh-halfh:centerh+halfh+1, :]


def build_10_crop(img,image_size):
    flipped_X = np.fliplr(img)
    crops = [
        img[:image_size, :image_size, :],  # Upper Left
        img[:image_size, img.shape[1] - image_size:, :],  # Upper Right
        img[img.shape[0] - image_size:, :image_size, :],  # Lower Left
        img[img.shape[0] - image_size:, img.shape[1] - image_size:, :],  # Lower Right
        center_crop(img, (image_size, image_size)),

        flipped_X[:image_size, :image_size, :],
        flipped_X[:image_size, flipped_X.shape[1] - image_size:, :],
        flipped_X[flipped_X.shape[0] - image_size:, :image_size, :],
        flipped_X[flipped_X.shape[0] - image_size:, flipped_X.shape[1] - image_size:, :],
        center_crop(flipped_X, (image_size, image_size))
    ]
    return np.array(crops)

def build_10_crop_set(X_train,crop_size):
    n = len(X_train)
    X_10_crop = [[] for i in range(10)]
    for i in tqdm(range(n)):
        x = X_train[i]
        x = (x - x.min()) / (x.max() - x.min())  # normalize for each pseudo pixel value
        img = resize(x, (crop_size, crop_size), mode='reflect')
        crops = build_10_crop(img,image_size)
        for j in range(10):
            X_10_crop[j].append(crops[j])

    X_10_crop = [np.array(X_10_crop[i]) for i in range(10)]
    '''
        if i == 0:
            crop = crops
        elif i == 1:
            crop = np.concatenate([crop[:, np.newaxis, :, :, :], crops[:, np.newaxis, :, :, :]], axis=1)
        else:
            crop = np.concatenate([crop[:,:,:,:],crops[:,np.newaxis,:,:,:]],axis = 1) #(10,len(X_train),299,299,3)
    '''
    return X_10_crop

def build_10_crop_set_k(X_train, crop_size, k):
    n = len(X_train)
    im_shape = X_train[0].shape
    s = [n]
    s.extend(im_shape)
    X_crop_k = np.zeros(s, dtype='float32')
    for i in tqdm(range(n)):
        x = X_train[i]
        x = (x - x.min()) / (x.max() - x.min())  # normalize for each pseudo pixel value
        img = resize(x, (crop_size, crop_size), mode='reflect')
        crops = build_10_crop(img,image_size)
        X_crop_k[i] = crops[k]
        del crops
    return X_crop_k
    

# Using K-fold Cross Validation with Data Augmentation.
def myAngleCV(X_train, X_angle, X_test):
    K = 5  # K-fold
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X_train, target_train))

    if os.path.exists('./data/X_test_crop_{}_9.pkl'.format(crop_size)):
        print_and_log('load croped test set...', logger)
        # test_crop = pickle.load(open('./data/X_test_crop_{}.pkl'.format(crop_size), 'rb'))
    else:
        print_and_log('Crop test set', logger)
        for k in range(10):
            test_crop = build_10_crop_set_k(X_test, crop_size, k)
            pickle.dump(test_crop, open('./data/X_test_crop_{}_{}.pkl'.format(crop_size, k), 'wb'), 4)
            del test_crop

    #Please help me to debug
    for j, (train_idx, test_idx) in enumerate(tqdm(folds)):
        print_and_log('\n===================FOLD={}'.format(j), logger)
        X_holdout = X_train[test_idx]
        Y_holdout = target_train[test_idx]

        # Angle
        X_angle_hold = X_angle[test_idx]

        print_and_log("Crop val_{} set".format(j), logger)
        X_holdout_crop = build_10_crop_set(X_holdout, crop_size)
        print_and_log("val_{}_crop_shape:{} {}".format(j,len(X_holdout_crop),X_holdout_crop[0].shape), logger)

        galaxyModel = getVggAngleModel()
        for m in filelist:
            print_and_log("Model:: {}".format(m), logger)
            galaxyModel.load_weights(rootpath+m+"/%s_aug_model_weights.hdf5" % j)
            for i in range(10):
                print('crop {}'.format(i))
                test_crop = pickle.load(open('./data/X_test_crop_{}_{}.pkl'.format(crop_size, i), 'rb'))
                if i == 0:
                    pred_valid = galaxyModel.predict([X_holdout_crop[i], X_angle_hold])
                    pred_test = galaxyModel.predict([test_crop, X_test_angle])
                else:
                    pred_valid += galaxyModel.predict([X_holdout_crop[i], X_angle_hold])
                    pred_test += galaxyModel.predict([test_crop, X_test_angle])
            pred_valid = pred_valid / 10
            pred_test = pred_test / 10
            print_and_log(' Val_{} Log Loss Validation= {}'.format(j,log_loss(Y_holdout, pred_valid)), logger)
            # Submission for each fold.
            submission = pd.DataFrame()
            submission['id'] = test['id']
            submission['is_iceberg'] = pred_test
            if not os.path.exists('output/10-crop-' + modelname + "-" + cur_time +"/" + m):
                os.makedirs('output/10-crop-' + modelname + "-" + cur_time +"/" + m)
            sub_file = 'output/10-crop-' + modelname + "-" + cur_time  +"/" + m + "/sub"  + "-fold" + str(j) + ".csv"
            mkdir_if_missing(os.path.dirname(sub_file))
            submission.to_csv(sub_file, index=False)


myAngleCV(X_train, X_angle, X_test)

