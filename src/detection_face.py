#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import imutils


Emotions = ["angry", "happy", "sad", "surprise"]


# LBPH - Face Recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create() # We create the function - LBPH
face_recognizer.read('/home/marco/pepper_sim_ws/src/dataset/modeloLBPHFace.xml')

# Opencv-Haar-Cascade Classifier - Face Detection
faceClassif = cv2.CascadeClassifier('/home/marco/pepper_sim_ws/src/dataset/haarcascade_frontalface_default.xml')


class Face_Recognition:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/pepper/camera/front/image_raw",Image,self.callback)

  def callback(self,data):
    
    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
    
    frame = cv_image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray,1.1,3)

    for (x,y,w,h) in faces:
      face = auxFrame[y:y+h,x:x+w]
      face = cv2.resize(face,(150,150),interpolation= cv2.INTER_CUBIC)
      result = face_recognizer.predict(rostro)
      cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
	
      if result[1] < 70:
        cv2.putText(frame,'{}'.format(Emotions[result[0]]),(x,y-25),2,0.5,(0,255,0),1,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
      else:
        cv2.putText(frame,'Unknown',(x,y-20),2,0.5,(0,0,255),1,cv2.LINE_AA)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

		
    cv2.imshow("Image window",frame)
    cv2.waitKey(1)
  
    

def main(args):
  ic = Face_Recognition()
  rospy.init_node('Face_Recognition', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
