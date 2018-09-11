import pandas as pd
import numpy as np
import os
import pickle
import keras
from keras.models import load_model
from keras import backend as K


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

def scale(scaler,x):
    shape = np.shape(x)
    x = np.reshape(x, (shape[0],-1))
    x = scaler.transform(x)
    x = np.reshape(x,shape)
    return x

#load model
model_opt = load_model('model/models/model.h5', custom_objects={'f1': f1})
 

#load data has processed
mfccs_te = []
data_processed = pickle.load(open("model/tmp/data.pkl","rb"))   
for item in data_processed:
    mfccs_te.append(item['mfccs'])

#padding     
for i in range(len(mfccs_te)):
    if np.shape(mfccs_te[i])[0]!=325:
        mfccs_te[i] = (keras.preprocessing.sequence.pad_sequences(np.array(mfccs_te[i]).T,325)).T   
        
#scale data
scaleMfccs = pickle.load(open('model/scale.pkl',"rb")) 
mfccs_te = scale(scaleMfccs, mfccs_te)

mfccs_te = np.reshape(mfccs_te,(-1,325,40,1))

#predict
yte_prob = model_opt.predict(mfccs_te,batch_size=128)
yte_pred = np.argmax(yte_prob,axis=1)
yte_pred = [x + 1 for x in yte_pred]

#write data to csv
result = []
for i in range(len(data_processed)):
    result.append([data_processed[i]['name'],yte_pred[i]])
resultFrame = pd.DataFrame(result,columns=['filename', 'genre'])
if os.path.exists('result') is False:
    os.mkdir('result')
resultFrame.to_csv('result/submission.csv', sep=',',index=False)
