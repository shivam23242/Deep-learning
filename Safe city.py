#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


# In[2]:


import pandas as pd
import io


# In[3]:


import csv
with open('E:/datasetforapp.csv', newline='', encoding='cp1252') as csvfile:
    csv_reader = list(csv.reader(csvfile, delimiter=','))

    print(csv_reader)


# In[4]:


df=pd.DataFrame(csv_reader,columns =['Category', 'Message'],)


# In[5]:


df.head(5)


# In[6]:


df.groupby('Category').describe()


# In[7]:


df.groupby('Category').describe()


# In[8]:


214/1285


# In[9]:


df_unsafe = df[df['Category']=='unsafe']
df_unsafe.shape


# In[10]:


df_safe = df[df['Category']=='safe']
df_safe.shape


# In[11]:


df_safe_downsampled = df_safe.sample(df_unsafe.shape[0])
df_safe_downsampled.shape


# In[12]:


df_balanced = pd.concat([df_safe_downsampled, df_unsafe])
df_balanced.shape


# In[13]:


df_balanced['Category'].value_counts()


# In[14]:


df_balanced['unsafe']=df_balanced['Category'].apply(lambda x: 1 if x=='unsafe' else 0)
df_balanced.sample(5)


# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_balanced['Message'],df_balanced['unsafe'], stratify=df_balanced['unsafe'])


# In[16]:


X_train.head(4)


# In[17]:


preprosesor_url='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encoder_url='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'


# In[18]:


bert_preprocessor=hub.KerasLayer(preprosesor_url)
bert_encoder=hub.KerasLayer(encoder_url)


# In[19]:


def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocessor(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']

get_sentence_embeding([
    "help me", 
    "what are you doing"]
)


# In[20]:


e = get_sentence_embeding([
    "help", 
    "save me",
    "why are you following me",
    "is anone there",
    "please let me go",
    "i will call police"
]
)


# In[21]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity([e[0]],[e[1]])


# In[22]:


cosine_similarity([e[0]],[e[3]])


# In[23]:


cosine_similarity([e[3]],[e[4]])


# In[24]:


text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocessor(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])


# In[25]:


model.summary()


# In[ ]:


len(X_train)


# In[30]:


METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)


# In[31]:


model.fit(X_train, y_train, epochs=12)


# In[32]:


model.evaluate(X_test, y_test)


# In[33]:


y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()


# In[34]:


import numpy as np

y_predicted = np.where(y_predicted > 0.8, 1, 0)
y_predicted


# In[35]:


from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_predicted)
cm 


# In[36]:


from matplotlib import pyplot as plt
import seaborn as sn
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[37]:


print(classification_report(y_test, y_predicted))


# In[38]:


from time import sleep
from speech_recognition import Microphone,Recognizer,AudioFile,UnknownValueError,RequestError


# In[ ]:


# Sampling frequency
freq = 44100


# Recording duration
duration = 10

 
def callback(recognizer,source):
    try:
        recognized=recognizer.recognize_google(source)
        data=[recognized]
        y_pred = model.predict(data)
        y_pred = np.where(y_pred > 0.8, 1, 0)
        y_pred
        print("you said: ",recognized+" ",y_pred)
        
    except RequestError as exc:
        print(exc)
        
    except UnknownValueError:
        print("unable to recognize")

def listen():
    recog=Recognizer()
    mic=Microphone()
    
    with mic:
        recog.adjust_for_ambient_noise(mic)
    
    print("talk")
    audio_stopper=recog.listen_in_background(mic,callback)
        
    
def foo():
    while True:
        print("Lisening")
        sleep(1)
        
listen()     
foo()

    


# In[ ]:





# In[ ]:




