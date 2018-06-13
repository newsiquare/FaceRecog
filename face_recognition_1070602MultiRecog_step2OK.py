#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to use dlib's face recognition tool.  This tool maps
#   an image of a human face to a 128 dimensional vector space where images of
#   the same person are near to each other and images from different people are
#   far apart.  Therefore, you can perform face recognition by mapping faces to
#   the 128D space and then checking if their Euclidean distance is small
#   enough. 
#
#   When using a distance threshold of 0.6, the dlib model obtains an accuracy
#   of 99.38% on the standard LFW face recognition benchmark, which is
#   comparable to other state-of-the-art methods for face recognition as of
#   February 2017. This accuracy means that, when presented with a pair of face
#   images, the tool will correctly identify if the pair belongs to the same
#   person or is from different people 99.38% of the time.
#
#   Finally, for an in-depth discussion of how dlib's tool works you should
#   refer to the C++ example program dnn_face_recognition_ex.cpp and the
#   attendant documentation referenced therein.
#
#
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  This code will also use CUDA if you have CUDA and cuDNN
#   installed.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.

# Step2
# 讀取faceData.txt與label.txt, 自動打開WebCAM進行辨識 

import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2
import json


#图像的目录
data = np.zeros((1,128))

#定义一个128维的空向量data
label = []   

#定义空的list存放人脸的标签


if len(sys.argv) != 4:
    print(
        "Call this program like this:\n"
        "   ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
        "You can download a trained facial shape predictor and recognition model from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
        "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
    exit()

#predictor_path = sys.argv[1]
#face_rec_model_path = sys.argv[2]
#faces_folder_path = sys.argv[3]

predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
faces_folder_path = '/Users/siquare/Desktop/Code/dlib/dlib-master/python_examples/'
threshold = 0.54
# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


#win = dlib.image_window()
#
## Now process all the images
#for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
#    print("Processing file: {}".format(f))
#    img = io.imread(f)
#    
#    fileName = f
#    labelName = f.split('_')[0]                                                              
#    print('current image: ', f)
#    print('current label: ', labelName)
#
#    win.clear_overlay()
#    win.set_image(img)
#
#    # Ask the detector to find the bounding boxes of each face. The 1 in the
#    # second argument indicates that we should upsample the image 1 time. This
#    # will make everything bigger and allow us to detect more faces.
#    dets = detector(img, 1)
#    print("Number of faces detected: {}".format(len(dets)))
#
#    # Now process each face we found.
#    for k, d in enumerate(dets):
#        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#            k, d.left(), d.top(), d.right(), d.bottom()))
#        # Get the landmarks/parts for the face in box d.
#        shape = sp(img, d)
#        # Draw the face landmarks on the screen so we can see what face is currently being processed.
#        win.clear_overlay()
#        win.add_overlay(d)
#        win.add_overlay(shape)
#
#        # Compute the 128D vector that describes the face in img identified by
#        # shape.  In general, if two face descriptor vectors have a Euclidean
#        # distance between them less than 0.6 then they are from the same
#        # person, otherwise they are from different people. Here we just print
#        # the vector to the screen.
#        face_descriptor = facerec.compute_face_descriptor(img, shape)
#        print(face_descriptor)
#        
#        # It should also be noted that you can also call this function like this:
#        #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100)
#        # The version of the call without the 100 gets 99.13% accuracy on LFW
#        # while the version with 100 gets 99.38%.  However, the 100 makes the
#        # call 100x slower to execute, so choose whatever version you like.  To
#        # explain a little, the 3rd argument tells the code how many times to
#        # jitter/resample the image.  When you set it to 100 it executes the
#        # face descriptor extraction 100 times on slightly modified versions of
#        # the face and returns the average result.  You could also pick a more
#        # middle value, such as 10, which is only 10x slower but still gets an
#        # LFW accuracy of 99.3%.
#
#
#        dlib.hit_enter_to_continue()
#     
def findNearestClassForImage(face_descriptor, faceLabel):
    temp =  face_descriptor - data
    e = np.linalg.norm(temp,axis=1,keepdims=True)
    min_distance = e.min() 
    print('distance: ', min_distance)
    if min_distance > threshold:
        return 'other'
    index = np.argmin(e)
    return faceLabel[index]                                                                    #关闭所有的窗口

   
        

def recognition(img):
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        rec = dlib.rectangle(d.left(),d.top(),d.right(),d.bottom())
        print(rec.left(),rec.top(),rec.right(),rec.bottom())
        shape = sp(img, rec)
        face_descriptor = facerec.compute_face_descriptor(img, shape)        
        
        class_pre = findNearestClassForImage(face_descriptor, label)
        print('class_pre = ',class_pre)
        #cv2.rectangle(img, (rec.left(), rec.top()+10), (rec.right(), rec.bottom()), (0, 255, 0), 2)
        #cv2.putText(img, class_pre , (rec.left(),rec.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

        if class_pre == 'ChenLiangGee' or class_pre == 'LinWeiCheng':
            # Draw a box around the face
            cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (216, 187, 43), 2)
            # Draw a label with a name below the face
            cv2.rectangle(img, (rec.left(), rec.bottom() - 35), (rec.right(), rec.bottom()), (216, 187, 43), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX    
        else:
            # Draw a box around the face
            cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (105, 232, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(img, (rec.left(), rec.bottom() - 35), (rec.right(), rec.bottom()), (105, 232, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, class_pre, (rec.left() + 6, rec.bottom() - 6), font, 1.0, (255, 255, 255), 1)
            

    cv2.imshow('image', img) 


labelFile=open('label.txt','r')
label = json.load(labelFile)                                                   #载入本地人脸库的标签
labelFile.close()
    
data = np.loadtxt('faceData.txt',dtype=float)                                  #载入本地人脸特征向量

cap = cv2.VideoCapture(0)
fps = 10
size = (640,480)
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
videoWriter = cv2.VideoWriter('video.avi', fourcc, fps, size)

while(1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    #rgb_small_frame = frame[:, :, ::-1]
    recognition(frame)
    videoWriter.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
videoWriter.release()
cv2.destroyAllWindows()

