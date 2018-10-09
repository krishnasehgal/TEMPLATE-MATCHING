
# coding: utf-8

# # TEMPLATE MATCHING

# ###### CITED FROM NET
# 

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('/Users/krishna/Downloads/proj1_cse573/task3/pos_8.jpg')
#img= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)


# In[2]:


template=cv2.imread('/Users/krishna/Downloads/proj1_cse573/task3/temp3.png',0)
blur=cv2.GaussianBlur(img1,(3,3),0)

laplacian = cv2.Laplacian(blur,cv2.CV_32F)
laplacian_template= cv2.Laplacian(template,cv2.CV_32F)
final=np.array(laplacian, dtype=np.float32)
final1=np.array(laplacian_template, dtype=np.float32)


width, height = template.shape[::-1]


res = cv2.matchTemplate(final,final1,cv2.TM_CCOEFF_NORMED)
threshold = 0.33
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img1, pt, (pt[0] + width, pt[1] + height), (0,255,255), 2)



# In[ ]:


cv2.imshow('result',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

