

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import config


train_dir = '..\\train_images' 

os.chdir('D:\Formations\Computer vision\projet\Detection-Reconnaissance-Matricules-Alg-riens\SRGAN\original_images_srgan')
path='..\original_images_srgan'


for img in os.listdir(path):

    img_array = cv2.imread(os.path.join(path,img))
    
    img_array = cv2.resize(img_array, (128,128))
    lr_img_array = cv2.resize(img_array,(32,32))
    print(train_dir+'\hr_images\\' + img)
    print(train_dir+'\lr_images\\'+ img)
    cv2.imwrite(train_dir+'\hr_images\\' + img, img_array)
    cv2.imwrite(train_dir+'\lr_images\\'+ img, lr_img_array)


    