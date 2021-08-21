#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2


# In[7]:


from deepface import DeepFace 


# In[2]:


img = cv2.imread('good.jpg')


# In[10]:


import matplotlib.pyplot as plt


# In[4]:


plt.imshow(img)


# In[5]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[7]:


predictions = DeepFace.analyze(img)


# In[8]:


predictions


# In[9]:


type(predictions)


# In[10]:


predictions['dominant_emotion']


# In[11]:


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# In[12]:


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,1.1,4)
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# In[13]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[14]:


font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,
           predictions['dominant_emotion'],
            (0,50),
            font,1,
            (0,0,255),
            2,
            cv2.LINE_4);
            


# In[15]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[12]:


img = cv2.imread('feared_man.jfif')


# In[13]:


plt.imshow(img)


# In[14]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[23]:


predictions = DeepFace.analyze(img)


# In[24]:


predictions


# In[25]:


font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,
           predictions['dominant_emotion'],
            (0,50),
            font,1,
            (0,0,255),
            2,
            cv2.LINE_4);


# In[ ]:





# In[ ]:





# In[34]:


import cv2
from deepface import DeepFace
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")
    
while True:
    ret,frame = cap.read()
    result = DeepFace.analyze(frame,actions=['emotion'])
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame,
               result['dominant_emotion'],
               (50,50),
               font,3,
               (0,0,255),
               2,
               cv2.LINE_4)
    cv2.imshow('demo vedio',frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:




