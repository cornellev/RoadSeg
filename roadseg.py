# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


FILEPATH = "" #fill with Local filepath to model parameters
CLASSES = "" #fill with Local filepath to class params
COLORS = "" #fill with Local filepath to color params (used for object vs. road vs. undetected)

def seg_callback(data):
    classes = CLASSES.read().strip().split("\n")
    colors = open(COLORS).read().strip().split("\n")
    net = cv2.dnn.readNet(FILEPATH)
    image = bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    image = imutils.resize(image, width=500)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0,
	swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()
    (numClasses, height, width) = output.shape[1:4]
    classMap = np.argmax(output[0], axis=0)
    
    return output

if __name__ == '__main__':
    while 1:
        try:
            rospy.init_node('RS_sender', anonymous=True)
            pub = rospy.Publisher('Segmenter', Bool, queue_size=1)
            depthsub = rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, seg_callback)
            rospy.spin()
        except rospy.ROSInterruptException:
            pass