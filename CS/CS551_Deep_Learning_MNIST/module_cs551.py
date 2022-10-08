#===============================================================
# IMPORTS ======================================================
#===============================================================
print('module_cs551')
import math
import os
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from sklearn.metrics import confusion_matrix
#import keract
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Model
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#===============================================================
# PERFORMANCE =================================================
#===============================================================
#------------------------------------------------------------------
# Confusion Matrix
#------------------------------------------------------------------
def _print_header(class_labels):
    g_HSTR=''   # HEADER ROW for printing confusing matrix
    for i in range(0,len(class_labels)):
        g_HSTR+='\t'+str(class_labels[i])
    return  g_HSTR
def _print_rows(cm_row, nos_labels):
    g_RSTR = ''
    for j in range(0,nos_labels):
        g_RSTR += '\t'+ str(cm_row[j])
    return g_RSTR
def print_conf_matrix(conf_matrix, suffix, class_labels):
    g_CMSTR=(suffix+'T\\P' + _print_header(class_labels)+'\n')
    nos_l=len(class_labels)
    for i in range(0,nos_l):
        g_CMSTR+=(suffix+str(class_labels[i]) + _print_rows(conf_matrix[i],nos_l )+'\n')
    return g_CMSTR

#------------------------------------------------------------------
# Performance Measures
#------------------------------------------------------------------
def _get_performance(conf_matrix):
    nos_class = len(conf_matrix[0,:]) # len of 0th row
    perf_measures_array = np.zeros((0,11),dtype ='float64')
    for i in range(0,nos_class):
        
        CLASS_ACTUAL = np.sum(conf_matrix[i,:]) #<---- ROW SUM = NOS TRUE CLASS
        CLASS_PRED = np.sum(conf_matrix[:,i])      #<---- COL SUM = NOS PRED CLASS
        MSUM = np.sum(conf_matrix)  # = TP + FN + FP + TN
        
        # compute TP, TN, FP, FN ---------------------------- 
        TP = conf_matrix[i,i]
        FP =  CLASS_PRED - TP
        FN = CLASS_ACTUAL - TP
        TN =  MSUM- FN - FP - TP #<------------ this methods counts more than true negative
        #TN = np.sum(conf_matrix[np.diag_indices(nos_class)]) - TP..

        #Accuracy #<-= how many samples correctly classifed out of all samples
        ACC = (TP+TN)   /   ( MSUM)  
        
        #Precision = TP/CLASS_PRED    #<- = how many samples correctly predicted as true out of all samples predicted as true
        PRE = (TP)      /   (TP+FP)         #Presicion
        
        #Sensetivity = TP/CLASS_ACTUAL #<- = how many samples correctly predicted as true out of all actually true samples
        SEN = (TP)      /   (TP+FN)         #Sensitivity/Recall
        
        #Specificity #<-= how many samples correctly predicted false out of all samples predicted as false
        SPF = (TN)      /   (TN+FP)         
        
        # F1-Score #<-= 2*TP / (CLASS_ACTUAL + CLASS_PRED) 
        F1S = 2*PRE*SEN /   (PRE+SEN)       #F1 score #<-= harmonic mean of Precision and Sensetivity

        prefi = np.array([CLASS_ACTUAL , CLASS_PRED, TP, FN, FP, TN, ACC, PRE, SEN, SPF, F1S])
        perf_measures_array = np.vstack((perf_measures_array,prefi))
        
    return perf_measures_array, nos_class
def print_performance(conf_matrix, class_labels, do_round=-1):
    #header_string = 'Class\tACC\tPRE\tSEN\tSPF\tF1S'
    header_string = 'Class\t#True\t#Pred\tTPs\tFNs\tFPs\tTNs\tACC\tPRE\tSEN\tSPF\tF1S'
    perf_measures, nos_class = _get_performance(conf_matrix)
    if len(class_labels)!=nos_class:
        print('WARNING:: Class label count mismatch!! Cannot print performance')
        return -1
    #nos_class = len(perf_measures[:,0])
    print('Performance for '+str(nos_class)+' classes')
    print (header_string)
    for i in range(0, nos_class):
        if do_round<0:
          perf_i = perf_measures [i,:]
        else:
          perf_i = np.round(perf_measures [i,:],do_round)
          
        print(
              str(class_labels[i])+'\t'+
              str(perf_i[0])+'\t'+
              str(perf_i[1])+'\t'+
              str(perf_i[2])+'\t'+
              str(perf_i[3])+'\t'+
              str(perf_i[4])+'\t'+
              str(perf_i[5])+'\t'+
              str(perf_i[6])+'\t'+
              str(perf_i[7])+'\t'+
              str(perf_i[8])+'\t'+
              str(perf_i[9])+'\t'+
              str(perf_i[10])
              )
    return nos_class

#===============================================================
# MNIST DATA HELPER ==============================================
#===============================================================
# Load Training Images
def _mnist_images(IMAGE_FILE):
  print('> Reading Images from',IMAGE_FILE )
  _FILE_IMAGE = open(IMAGE_FILE,"rb" ) 

  # first 4 byte = magic number
  _magic = int.from_bytes(_FILE_IMAGE.read(4), "big")
  print('Magic Number\t',_magic)

  # next 4 byte = nos items
  _nos_images = int.from_bytes(_FILE_IMAGE.read(4), "big")
  print('Nos Images\t',_nos_images)

  # next 4 byte = image dimension Rows
  _nos_rows= int.from_bytes(_FILE_IMAGE.read(4), "big")
  print('Nos Rows\t',_nos_rows)

  # next 4 byte = image dimension Cols
  _nos_cols = int.from_bytes(_FILE_IMAGE.read(4), "big")
  print('Nos Cols\t',_nos_cols)

  _nos_bytes = _nos_rows*_nos_cols
  print('Bytes per Image\t',_nos_bytes)

  # next onwards.... one image per _nos_bytes bytes
  _buffer = np.frombuffer(_FILE_IMAGE.read(_nos_bytes*_nos_images), 
                          dtype=np.uint8, count=_nos_bytes*_nos_images, offset=0)
  print('ImageBuffer:',_buffer.dtype, _buffer.shape)
  _buffer = _buffer.reshape(_nos_images,_nos_rows,_nos_cols)
  print('ImageBuffer Reshaped:',_buffer.dtype, _buffer.shape)

  _FILE_IMAGE.close()
  print('Done\n' )
  return _buffer, _nos_images, _nos_rows, _nos_cols

def _mnist_labels(LABEL_FILE):
  print('> Reading Labels from',LABEL_FILE )
  _FILE_LABEL = open(LABEL_FILE,"rb" ) 

  # first 4 byte = magic number
  _magic = int.from_bytes(_FILE_LABEL.read(4), "big")
  print('Magic Number\t',_magic)

  # next 4 byte = nos items
  _nos_labels = int.from_bytes(_FILE_LABEL.read(4), "big")
  print('Nos Labels\t',_nos_labels)

  # next onwards.... one image per _nos_bytes bytes
  _buffer = np.frombuffer(_FILE_LABEL.read(_nos_labels), 
                          dtype=np.uint8, count=_nos_labels, offset=0)
  print('LabelBuffer:',_buffer.dtype, _buffer.shape)

  _FILE_LABEL.close()
  print('Done\n' )
  return _buffer, _nos_labels

#===============================================================
# IMAGE TRANSFORMS =============================================
#===============================================================

# flips horizontal and vertical
def _aug_flip(imageA, horz=True, vert=True):
  res = imageA
  if vert:
    res = np.flip(res,1)
  if horz:
    res = np.flip(res,2)
  return res

def _aug_shift(imageA, shiftA):
  imageT = []
  for i in range(0, len(imageA)):
    imageT.append(np.roll(np.roll(imageA[i], shiftA[i,0],axis=0),shiftA[i,1],axis=1))
  return imageT

def _aug_rotate(imageA, radA, opx, opy):
  imageT = [] #<<--- appending to list is faster than vstacking
  cpx,cpy = int(opx/2), int(opy/2)
  for i in range(0, len(imageA)):
    imageT.append(transform.warp(imageA[i],
                                       transform.AffineTransform(matrix= _get_rotation_matrix_wrtp(radA[i], cpx,cpy)), 
                                       output_shape=(opx, opy)))
  return imageT

# tranformation matrices for translation and rotation 
def _get_translation_matrix(tX,tY):
  return np.array([[1,0,tX],[0,1,tY],[0,0,1]])

def _get_rotation_matrix_wrtc(tH):
  return np.array([[math.cos(tH), -math.sin(tH),  0],
                   [math.sin(tH), math.cos(tH),   0],
                   [0,            0,              1]])
  
def _get_rotation_matrix_wrtp(tH, tX, tY):
  # rotation wrt to a point = translate to center - rotate - translate back to poit
  t1 = _get_translation_matrix(-tX, -tY)
  tr = _get_rotation_matrix_wrtc(tH)
  t2 = _get_translation_matrix(tX, tY)
  return np.matmul(t2, np.matmul(tr,t1))

#===============================================================
# Model Definition =================================================
#===============================================================

def get_model(print_summary, model_name, conv_kernels, conv_kernel_size, dense_size):
    global _ishape, _nos_classes
  
    inputL = Input( shape=_ishape, name = "input" )

    conv_1 =  Conv2D(conv_kernels,                                       #kernels, 
                          kernel_size=conv_kernel_size,                  #kernel_size
                          strides=(1,1), 
                          padding='valid', 
                          data_format='channels_last', 
                          dilation_rate=1, 
                          activation=tf.nn.leaky_relu, 
                          use_bias=True, 
                          kernel_initializer='glorot_uniform', 
                          bias_initializer='zeros', 
                          kernel_regularizer=None, 
                          bias_regularizer=None, 
                          activity_regularizer=None, 
                          kernel_constraint=None, 
                          bias_constraint=None,
                          name='conv_1') (inputL) 
    
  
    norm_2 = tf.keras.layers.BatchNormalization(
                                        axis=-1,
                                        momentum=0.99,
                                        epsilon=0.001,
                                        center=True,
                                        scale=True,
                                        beta_initializer="zeros",
                                        gamma_initializer="ones",
                                        moving_mean_initializer="zeros",
                                        moving_variance_initializer="ones",
                                        beta_regularizer=None,
                                        gamma_regularizer=None,
                                        beta_constraint=None,
                                        gamma_constraint=None,
                                        renorm=False,
                                        renorm_clipping=None,
                                        renorm_momentum=0.99,
                                        fused=None,
                                        trainable=True,
                                        virtual_batch_size=None,
                                        adjustment=None,
                                        name='norm_2') (conv_1)
    

    flat_ = Flatten(data_format=None,name='flat_') (norm_2)
    den_3 = Dense(dense_size, activation=tf.nn.leaky_relu, name = "den_3")(flat_)

    outputL = Dense(_nos_classes, activation=tf.nn.softmax, name = "output")(den_3)

    model=Model(inputs=inputL, outputs=outputL, name=model_name)
    #-------------------------------------
    if print_summary:
        print(model.summary())
    return model
# =========================================================================================


print('Done!')
		
