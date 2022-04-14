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


faceClassif = cv2.CascadeClassifier('/home/marco/pepper_sim_ws/src/dataset/haarcascade_frontalface_default.xml')
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()


class image_converter:

  def __init__(self):
    #self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/pepper/camera/front/image_raw",Image,self.callback)

  def callback(self,data):
    
    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
    
    frame = cv_image
    frame =  imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray,1.1,3)

    for (x,y,w,h) in faces:
	if h*w < 1800: 
	    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)
	    # rostro = auxFrame[y:y+h,x:x+w]
	    #area=h*w
	    #cv2.putText(frame,str(area), (x, y), cv2.FONT_ITALIC, 0.75, (255, 0, 0), 2)

	# rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
		
    cv2.imshow("Image window",frame)
    cv2.waitKey(1)
  
    

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
