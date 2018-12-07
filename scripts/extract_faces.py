import cv2
import os
import dlib

import subprocess
import shutil
from shutil import copyfile
import sys

lastDetectedFace = None

def progressBar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

#detect face in image
def DetectFace(cascade, image, scale_factor=1.1):
    global lastDetectedFace
    #convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          
    #find face(s) in image
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=3, minSize=(110,110))
    
    #crop image to face region
    for face in faces:
        if face is None:
            return (lastDetectedFace, False)
        x,y,w,h = face
        lastDetectedFace = (x,y,w,h)
        #image = image[y:y+h, x:x+w]
        return (lastDetectedFace, True)
    return (lastDetectedFace, False)

def extractFramesFromVideo(path,savePath, faceDetectorPrecision,
                           missedFacesDirectory='D:\\Neil_TFS\\AR Emotion Research\\OMG-Empathy-Challenge\\data\\faces\\MissedFaces.txt', 
                           size=128):
    videos = os.listdir(path + "/")
    #haarcascade='H:\\haarcascade_frontalface_default.xml'
    haarcascade='D:\\Neil_TFS\\AR Emotion Research\\OMG-Empathy-Challenge\\omg-empathy\\scripts\\c++-scripts\\haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(haarcascade)
    lCountMissedFaces = 0
    fileMissedFaces = open(missedFacesDirectory, "w")
    fileMissedFaces.write("Missed Faces\n")
    
    for video in videos:

        videoPath = path + "/" + video
        print ("- Processing Video:", videoPath + " ...")
        

        #copyTarget = "/data/datasets/OMG-Empathy/clip1.mp4"
        copyTarget = "D:/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/temp.mp4"
        print ("--- Copying file:", videoPath + " ...")
        copyfile(videoPath, copyTarget)
        cap = cv2.VideoCapture(copyTarget)

        #cap = cv2.VideoCapture(videoPath)
        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        check = True
        imageNumber = 0
        print ("- Extracting Faces:", str(totalFrames) + " Frames ...")

        savePathSubject = savePath + "/" + video + "/Subject/"

        if not os.path.exists(savePathSubject):
            os.makedirs(savePathSubject)
            while (check):
                    check, img = cap.read()
                    if img is not None:
                        #Extract subject faces
                        imageSubject = img[0:720, 1080:2560]

                        try:                            
                            # detect subject faces
                            _, faceFound = DetectFace(cascade, imageSubject)
        
                            if not faceFound:
                                # update lCountMissedFaces and write to file if missed subject face
                                lCountMissedFaces += 1
                                msg = 'Face missed at {0}. Frame number {1}. Total missed faces: {2}\n'.format(savePathSubject, imageNumber, lCountMissedFaces)
                                fileMissedFaces.write(msg)
                            
                            x,y,w,h = lastDetectedFace
                            faceSubject = imageSubject[y:y+h, x:x+w]
                            scaledFace = cv2.resize(faceSubject, (size, size))
                            #write new cropped and scaled subject image to file
                            cv2.imwrite(savePathSubject + "/%d.png" % imageNumber, scaledFace)
                            
                        except:
                            print('error')
                            
                        imageNumber += 1
                        progressBar(imageNumber, totalFrames)

if __name__ == "__main__":


    #Path where the videos are
    #path ="/data/videos/Training/"
    #path = "D:/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/data/Training/Videos"
    path = "D:/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/Testing/Videos"

    #Path where the faces will be saved
    #savePath ="/data/faces/Training/"
    #savePath = "D:/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/data/faces/Training"
    savePath = "D:/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/omg-empathy/data/Testing/faces"
    
    # If 1, the face detector will act upon each of the frames. If 1000, the face detector update its position every 100 frames.
    faceDetectorPrecision = 3

    detector = dlib.get_frontal_face_detector()

    extractFramesFromVideo(path, savePath, faceDetectorPrecision, missedFacesDirectory='D:\\Neil_TFS\\AR Emotion Research\\OMG-Empathy-Challenge\\omg-empathy\\data\\Testing\\MissedFaces.txt', size=256)

    #path = "D:/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/data/Validation/Videos"
    #savePath = "D:/Neil_TFS/AR Emotion Research/OMG-Empathy-Challenge/data/faces/Validation"
    #extractFramesFromVideo(path, savePath, faceDetectorPrecision)