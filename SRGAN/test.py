import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


###################################################################################
#Test - perform super resolution using saved generator model
from keras.models import load_model
from numpy.random import randint
from sklearn.model_selection import train_test_split



os.chdir('D:\Formations\Computer vision\projet\Detection-Reconnaissance-Matricules-Alg-riens\SRGAN\\train_images')

n=5
lr_list = os.listdir("..\\train_images\lr_images")[:n]

lr_images = []
for img in lr_list:
    img_lr = cv2.imread("..\\train_images\lr_images\\" + img)
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    lr_images.append(img_lr)   


hr_list = os.listdir("..\\train_images\hr_images")[:n]
   
hr_images = []
for img in hr_list:
    img_hr = cv2.imread("..\\train_images\hr_images\\" + img)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
    hr_images.append(img_hr)   

lr_images = np.array(lr_images)
hr_images = np.array(hr_images)


lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, test_size=0.33, random_state=42)




generator = load_model('gen.h5', compile=False)

ix = randint(0, len(lr_test))


sreeni_lr = lr_test[ix]
sreeni_hr = hr_test[ix]

#Change images from BGR to RGB for plotting. 
#Remember that we used cv2 to load images which loads as BGR.
sreeni_lr = cv2.cvtColor(sreeni_lr, cv2.COLOR_BGR2RGB)
sreeni_hr = cv2.cvtColor(sreeni_hr, cv2.COLOR_BGR2RGB)

sreeni_lr = sreeni_lr / 255.
sreeni_hr = sreeni_hr / 255.

sreeni_lr = np.expand_dims(sreeni_lr, axis=0)
sreeni_hr = np.expand_dims(sreeni_hr, axis=0)

generated_sreeni_hr = generator.predict(sreeni_lr)

# plot all three images
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('LR Image')
plt.imshow(sreeni_lr[0,:,:,:])
plt.subplot(232)
plt.title('Superresolution')
plt.imshow(generated_sreeni_hr[0,:,:,:])
plt.subplot(233)
plt.title('Orig. HR image')
plt.imshow(sreeni_hr[0,:,:,:])

plt.savefig('results.png')


