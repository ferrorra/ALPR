from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from yolov3_tf2.models import YoloV3 #the model we gonna use
from yolov3_tf2.dataset import transform_images #resizing images for yolo
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing #non max sup
from deep_sort import nn_matching #deep association metrics
from deep_sort.detection import Detection #detect obj
from deep_sort.tracker import Tracker #write tracking info
from tools import generate_detections as gdet #import feature generation encoder (appearance features)

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')


#init deepsort params
max_cosine_distance = 0.5 #if two objects are the same or not
nn_budget = None #form gallery for each detection to use dnn to extract features
nms_max_overlap = 0.8 #avoiding too many detections for the same object

model_filename = 'model_data/mars-small128.pb' #deepsort pretrained model for pedestrians
encoder = gdet.create_box_encoder(model_filename, batch_size=1) #feature generator
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget) #association metric
tracker = Tracker(metric) #create our tracker

vid = cv2.VideoCapture('./data/video/onelane.mp4') #try on a video, replace with 0 to use cam


#saving the video inference
codec = cv2.VideoWriter_fourcc(*'XVID') #format
vid_fps =int(vid.get(cv2.CAP_PROP_FPS)) #frames per sec
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) #getting dimensions
out = cv2.VideoWriter('./data/video/results-onelane.avi', codec, vid_fps, (vid_width, vid_height))  #save
cv2.namedWindow("output", cv2.WINDOW_NORMAL)


while True: #capture all frmaes from vids
    ret, img = vid.read()
    cv2.resizeWindow('output', 1024, 768)

    if img is None:
        break

    #conversion for yolo prediction
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0) #transform from 3d array to 4d array, adding batch size as a dimension
    img_in = transform_images(img_in, 416) #img size for yolov3

    t1 = time.time() #starting timer

    boxes, scores, classes, nums = yolo.predict(img_in) #should be modified to use deepsort algo

    #boxes are 3d shape of (1,100,4) x,y center coords, width, height of box
    #scores are 2d shaped (1,100) matrice creuse of confidence
    #classes 2d shape (1,100), matrice creuse of class number
    #nums 1d shapes (1,) number of detected objects

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0]) #convert boxes into a list
    features = encoder(img, converted_boxes) #generate features

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)] #detect with deepsort

    #get info from deepsort detection
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores) #what should we takeoff
    detections = [detections[i] for i in indices] #detect again without redundancy

    tracker.predict() #propagate the track state distribution using filltering
    tracker.update(detections) #update appearance and diappearance of an object

    #visualisation 
    cmap = plt.get_cmap('tab20b') #mapping colors to numbers (id)
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    current_count = int(0)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1: #if filltering can't get track or if there was no update we skip
            continue
        bbox = track.to_tlbr() #we get bounding box
        #we display object detection, class id, color and bbox drawing
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        height, width, _ = img.shape
        cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (225, 255, 0), thickness=1)
        
        cv2.line(img, (0, int(3*height/6+height/2.5)), (width, int(3*height/6+height/2.5)), (225, 0, 0), thickness=1)


        center_y = int(((bbox[1])+(bbox[3]))/1.1)

        
        if class_name=="car" or class_name=="truck" or class_name=="bus":
            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2) #x,y,x+w,y+h

            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                        +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)



            if center_y <= int(3*height/6+height/20) and center_y >= int(3*height/6-height/20):
                #extraction d'image pour la detection du matricule
                cropped_image = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                #cv2.imshow('cropped',cropped_image)
                cv2.imwrite(f"./data/images/v1/Image{int(track.track_id)}.jpg", cropped_image)

                if center_y <= int(3*height/6+height/2.5) and center_y >= int(3*height/6-height/2.5):
                    #extraction d'image pour la detection du matricule
                    cropped_image = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    #cv2.imshow('cropped',cropped_image)
                    cv2.imwrite(f"./data/images/v2/Image{int(track.track_id)}.jpg", cropped_image)

    #fps = 1./(time.time()-t1)
    #cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)

    cv2.imshow('output', img)
    #cv2.resizeWindow('output', 1024, 768)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()