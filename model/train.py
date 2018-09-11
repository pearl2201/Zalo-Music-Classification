import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import glob
import pickle
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
import itertools
import six
import keras
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout,BatchNormalization,Input,Concatenate,Activation,Embedding,Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import SMOTE
from resnet import build,basic_block

'''
define of f1 score as metric for imbalance data
'''
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def save_variable(data,file_name):
    output = open(file_name, 'wb')
    pickle.dump(data, output)
    output.close()
if not os.path.exists("model/checkpoint"):
    os.mkdir("model/checkpoint")
#read input
music_train_data = pickle.load(open("model/tmp/data.pkl","rb"))
name_csv_file = 'model/train.csv'
df = pd.read_csv(name_csv_file,header=None)
train_df = df.values
mfccs = []
labels = []
for i in range(len(train_df)):
    for fp in music_train_data:
        if fp["name"] == train_df[i][0]:
            mfccs.append(fp["mfccs"] )
            labels.append(train_df[i][1] - 1)        
del music_train_data

#padding mfcc feature
for i in range(len(mfccs)):    
    if np.shape(mfccs[i])[0]!=325:
        mfccs[i] = (keras.preprocessing.sequence.pad_sequences(np.array(mfccs[i]).T,325)).T 

# calculate label weights
mu =0.5
class_weights = {}
maxWeights = 0
for i in range(10):
    score = math.log(mu*len(labels)/float(len([y for y in labels if y == i])))
    class_weights[i] = score if score > 1.0 else 1.0
    maxWeights = class_weights[i] if maxWeights < class_weights[i] else maxWeights
for i in range(10):
    class_weights[i] = class_weights[i]/maxWeights        
        
#scale data
def minmaxScale(x):
    shape = np.shape(x)
    x = np.reshape(x, (shape[0],-1))
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    x = np.reshape(x,shape)
    return scaler, x
scaleMfccs, mfccs = minmaxScale(mfccs)
save_variable(scaleMfccs,'/model/checkpoint/scale.pkl') # save variable for test

#train test split
labels_train = []
labels_validation = []
xtr_train = []
xtr_validation = []
rate = 0.8

for i in range(10):
    
    indexes = []
    for j in range(np.shape(labels)[0]):
        if labels[j] == i:
            indexes.append(j)
    np.random.shuffle(indexes) 
    idxHighTrain = int(len(indexes)*rate)
    
    labels_train.extend([labels[k] for k in indexes[:idxHighTrain]])
    xtr_train.extend([mfccs[k] for k in indexes[:idxHighTrain]])

    labels_validation.extend([labels[k] for k in indexes[idxHighTrain:]])      
    xtr_validation.extend([mfccs[k] for k in indexes[idxHighTrain:]]) 
    



#Uppersample data with SMOTE
ros = SMOTE()
shape = np.shape(xtr_train)
xtr_train = np.reshape(xtr_train,(shape[0],-1))
xtr_rs_train, labels_rs_train = ros.fit_sample(xtr_train, labels_train)

xtr_rs_train = np.reshape(np.array(xtr_rs_train),(-1,325,40,1))
xtr_validation = np.reshape(np.array(xtr_validation),(-1,325,40,1))

ytr_rs_train = keras.utils.to_categorical(labels_rs_train, 10)
ytr_validation = keras.utils.to_categorical(labels_validation, 10)

model = build((325,40,1), 10, basic_block, [2, 2, 2, 2])
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',f1])


#set callback save model
filepath="model/checkpoint/zalo-music-resnet18-improvement-epoch: {epoch:02d}- val_loss: {val_loss:.2f}-vall_acc: {val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#fit model
model.fit(xtr_rs_train , ytr_rs_train,
          validation_data= (xtr_validation, ytr_validation),
          epochs=1,batch_size=128, callbacks=callbacks_list, class_weight=class_weights )

'''
plot best model performance
'''
'''
#load best model
#model_opt = load_model("best-model-name.hdf5", custom_objects={'f1': f1})
model_opt = model
ytr_val_prob = model_opt.predict(xtr_validation,batch_size=128)
ytr_val_pred = np.argmax(ytr_val_prob,axis=1)

# Compute confusion matrix
cnf_matrix = confusion_matrix(labels_validation, ytr_val_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['1','2','3','4','5','6','7','8','9','10'] ,title='Confusion matrix, without normalization')
'''