# 3. Import libraries and modules
import numpy as np
import pandas as pd
np.random.seed(123)  # for reproducibility
import os
os.environ['KERAS_BACKEND']='theano'
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import theano
import struct
from keras.models import Sequential
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Convolution3D, MaxPooling3D,ZeroPadding3D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import np_utils
from IPython.core.debugger import Tracer
import matplotlib.pyplot as plt
working_path='/work/vsankar/projects/kaggle_segmented_predict/'
patients_folder='/work/vsankar/projects/lungCancer/'

def get_model(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(32, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1), 
                            input_shape=(1, 20, 512, 512)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool3'))

#     5th layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(2, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(2, activation='softmax', name='fc8'))
    if summary:
        print(model.summary())
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
def normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    image -= mean  # images should already be standardized, but just in case
    image /= std
    return image

model = get_model(summary=False)

working_path1 = '/work/vsankar/projects/Kaggle_segmented_predicted_cancer/'
working_path2 = '/work/vsankar/projects/Kaggle_segmented_predicted_nocancer/'

patient_list=os.listdir(patients_folder+'sample_images')

data = np.load(working_path1+'Pat_mean_full_%d.npy' %(0))
data = data[()]

n = len(data)
xt1 = np.ndarray([n,1,20,512,512],dtype=np.float32)
c=0
yt1 = np.empty([n, 1])
#Tracer()()
for i in data.keys():
#     xt[i,0] =np.nanmean(data[i][0][0:20,:,:,:],axis=0)
#     Tracer()()
    for j in range(20):
#         print(j)
        xt1[c,0,j] = np.nan_to_num(data[i][0][j,0,:,:])
        
    
    yt1[c] = data[i][1]
    c=c+1
    
yt1 =  np.array(yt1)
        
lenPat_full =6       
for i in range(lenPat_full):
    data = np.load(working_path1+'Pat_mean_full_%d.npy' %(i+1))
    data = data[()]
    #Tracer()()
    n = len(data)
    xti = np.ndarray([n,1,20,512,512],dtype=np.float32)
    c=0
    yti  = np.empty([n, 1])
    for i in data.keys():
        
#         xti[i,0] = np.nanmean(data[i][0][0:20,:,:,:],axis=0)
        
        
        for j in range(20):
            xti[c,0,j] = normalize(data[i][0][j,0,:,:])
        
        yti[c] = data[i][1]
        c=c+1
    
    yti = yti=np.array(yti)
        
    xt1 = np.concatenate((xt1, xti), axis=0)
    yt1 = np.concatenate((yt1,yti), axis=0)
    
    

data = np.load(working_path2+'Pat_mean_full_%d.npy' %(0))
data = data[()]

n = len(data)
xt2 = np.ndarray([n,1,20,512,512],dtype=np.float32)
#Tracer()()
yt2 = np.empty([n, 1])
c=0
for i in data.keys():
#     xt[i,0] =np.nanmean(data[i][0][0:20,:,:,:],axis=0)
    
    for j in range(20):
        xt2[c,0,j] = np.nan_to_num(data[i][0][j,0,:,:])
    
    yt2[c] = data[i][1]
    c=c+1
    
yt2 =  np.array(yt2)
        
lenPat_full = 5        
for i in range(lenPat_full):
    data = np.load(working_path2+'Pat_mean_full_%d.npy' %(i+1))
    data = data[()]
    #Tracer()()
    n = len(data)
    xti = np.ndarray([n,1,20,512,512],dtype=np.float32)
    c=0
    yti  = np.empty([n, 1])
    for i in data.keys():
        
#         xti[i,0] = np.nanmean(data[i][0][0:20,:,:,:],axis=0)
        
        
        for j in range(20):
            xti[c,0,j] = normalize(data[i][0][j,0,:,:])
        

        yti[c] = data[i][1]
        c=c+1
    
    yti = yti=np.array(yti)
        
    xt2 = np.concatenate((xt2, xti), axis=0)
    yt2 = np.concatenate((yt2,yti), axis=0)

xt =  np.concatenate((xt1, xt2), axis=0)
yt = np.concatenate((yt1,yt2), axis=0)

xt = np.nan_to_num(xt)


print('shape of train')
print(xt.shape)

if False:
    print('using exisitng weights')
    model.load_weights('/home/vsankar/bharat/KaggleSegmentedClassification_norm_both/weights_Classificaiton.04.hdf5')


model_checkpoint = ModelCheckpoint('/home/vsankar/bharat/KaggleSegmentedClassification_norm_both/weights_Classificaiton.{epoch:02d}.hdf5', monitor='loss',
                                       save_best_only=False)
print('fitting')

model.fit(xt, yt,
          batch_size=1, nb_epoch=100, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])
