import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras import callbacks
#

df_features=pd.read_csv('/media/sanjay/HDD2/tflow/anneal_features.csv')
df_labels=pd.read_csv('/media/sanjay/HDD2/tflow/anneal_labels.csv')
df_labels2=pd.get_dummies(df_labels['class'])
## labels needs to be different shape.  get dummies
#
print("num classes: ",len(df_labels['class'].unique()))

anneal_model = keras.Sequential(
        [
            layers.Dense(64, activation="relu"),
            layers.Dense(5, activation="softmax"),
        ]
)

anneal_model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

earlystopping = callbacks.EarlyStopping(monitor ="loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)

history = anneal_model.fit(x=np.array(df_features),y=np.array(df_labels2),
                    epochs=200,
                    verbose=1,
                    callbacks =[earlystopping])

df_test_features=pd.read_csv('/media/sanjay/HDD2/tflow/features_anneal_test.csv')
df_labels_test=pd.read_csv('/media/sanjay/HDD2/tflow/labels_anneal_test.csv')
df_labels_test2=pd.get_dummies(df_labels['class'])

loss, accuracy = anneal_model.evaluate(x=np.array(df_features),y=np.array(df_labels_test2))
print("accuracy on test data: ", accuracy)

